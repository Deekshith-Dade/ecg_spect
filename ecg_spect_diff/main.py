import os
import argparse
from datetime import datetime
import logging

import torch
from accelerate import Accelerator
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from PIL import Image
from torch.utils.data import DataLoader
from ecg_spect_diff.dataset.dataset import getKCLTrainTestDataset
from transformers import CLIPTextModel, CLIPTokenizer
import torch.distributed as dist
from tqdm.auto import tqdm
import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
file_handler = logging.FileHandler(f'./ecg_spect_diff/out/finetune_{timestamp}.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def stack_images(images):
    offset = 4
    width, height = images[0].size
    total_height = (height + offset) * len(images)
    stacked_image = Image.new('RGB', (width, total_height))

    y_offset = 0
    for img in images:
        stacked_image.paste(img, (0, y_offset))
        y_offset += height + offset
    
    return stacked_image

def sample_images(pipeline, num_images=4):
    pipeline.unet.eval()

    generator = torch.Generator(device=pipeline.device).manual_seed(42)
    prompts = ["ECG with severe hyperkalemia", "ECG with healthy signs"] * int(num_images/2)
    
    with torch.no_grad():
        images = pipeline(
            prompts,
            num_inference_steps=50,
            generator=generator
        ).images
    
    stacked_image = stack_images(images)
    
    pipeline.unet.train()
    
    return stacked_image

def train_one_epoch(
    accelerator,
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    dataloader,
    optimizer,
    epoch,
    args
):
    unet.train()
    total_loss = 0.
    for step, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        with accelerator.accumulate(unet):
            latents = vae.encode(batch["image"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz, ), device=accelerator.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoding = tokenizer(batch["cond_inputs"]["text"],
                                 truncation=True,
                                 max_length=tokenizer.model_max_length,
                                 padding="max_length",
                                 return_tensors="pt")
            
            input_ids = encoding.input_ids.to(accelerator.device)
            encoder_hidden_states = text_encoder(input_ids)[0]
            
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").mean()
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.detach().item()
        
        if accelerator.is_main_process and step % args.logging_steps == 0:
            print(f"Running Sampling")
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=accelerator.unwrap_model(vae),
                tokenizer=tokenizer,
                scheduler=noise_scheduler
            )
            pipeline = pipeline.to(accelerator.device)
            val_images = sample_images(pipeline, 4)
            
            # Calculate average loss up to this point
            avg_loss = total_loss / (step + 1)
            
            # Log to wandb
            accelerator.log({
                "train/loss": avg_loss,
                "train/step": step,
                "train/epoch": epoch,
                "samples": [wandb.Image(val_images)]
            })
            
        if step % args.save_steps == 0:
            save_progress(accelerator, unet, args.output_dir, epoch, step)
            
            
        
    return  total_loss / len(dataloader)

def save_progress(accelerator, unet, output_dir, epoch, step):
    if accelerator.is_main_process:
        save_path = os.path.join(output_dir, f'checkpoint-{epoch}-{step}')
        accelerator.save_state(save_path)


def main(args, dataset_config):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
        log_with="wandb"
    )
    
    accelerator.init_trackers(
        project_name="spectrogram_diffusion",
        config= vars(args) | dataset_config,
        init_kwargs = {"wandb": {
            "entity": "deekshith",
            "resume": "allow",
            "name": args.project_name
        }}
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_main_process else logging.WARN
    )

    #################### LOAD MODELS ####################
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet")
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    train_dataset, test_datset = getKCLTrainTestDataset(dataset_config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    unet, vae, text_encoder, tokenizer, noise_scheduler, optimizer, train_dataloader = accelerator.prepare(
        unet, vae, text_encoder, tokenizer, noise_scheduler, optimizer, train_dataloader
    )

    #### TRAINING INFO ####
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("**** RUNNING TRAINING ****")
    logger.info(f" Num Examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    # Training Loop
    for epoch in range(args.num_train_epochs):
        train_loss = train_one_epoch(
            accelerator,
            unet,
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            train_dataloader,
            optimizer,
            epoch,
            args
        )

        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch}: Average loss = {train_loss}")
        
        # Checkpointing
        if epoch == args.num_train_epochs - 1 and accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=accelerator.unwrap_model(vae),
                tokenizer=tokenizer,
                scheduler=noise_scheduler
            )
            pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Stable Diffusion on ECG Spectrograms")
    parser.add_argument("--model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--save_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./ecg_spect_diff/checkpoints")
    parser.add_argument("--project_name", type=str, default="project")

    args = parser.parse_args()
    datasetConfig = {
        "data_path": "/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/ecg_spect_diff/parquet_patients/ecgs_patients.parquet",
        "dataDir": "/uufs/sci.utah.edu/projects/ClinicalECGs/AllClinicalECGs/",
        "timeCutOff": 960000,
        "lowerCutOff": 0,
        "randSeed": 7777,
        "scale_training_size": 1.0,
        "kcl_params": {
            "lowThresh": 4.0,
            "highThresh": 5.0,
            "highThreshRestrict": 11.0
        }
    }
    
    current_time = datetime.now()
    project_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = "/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/ecg_spect_diff"
    args.output_dir = f"{base_dir}/checkpoints/{current_time}"
    args.project_name = project_name 
    main(args, datasetConfig)