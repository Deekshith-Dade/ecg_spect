import os
import argparse
from datetime import datetime
import logging

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    UNet2DModel,
    VQModel,
    DDIMScheduler
)
from PIL import Image
from torch.utils.data import DataLoader
from ecg_spect_diff.dataset.dataset import getKCLTrainTestDataset
from transformers import CLIPTextModel, CLIPTokenizer, BertTokenizer, BertModel
import torch.distributed as dist
from tqdm.auto import tqdm

from ecg_spect_diff.train import Training

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
file_handler = logging.FileHandler(f'./ecg_spect_diff/out/finetune_{timestamp}.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def get_dataset_min_max(dataloader, max_samples=None):
    global_min = float('inf')
    global_max = float('-inf')
    
    samples = 0
    print("Calculating Min and Max over the dataset...")
    
    for batch in tqdm(dataloader):
        spectrograms = batch['image']
        samples += spectrograms.shape[0]
        
        batch_min = torch.min(spectrograms)
        batch_max = torch.max(spectrograms)
        
        if batch_min < global_min:
            global_min = batch_min.item()
        if batch_max > global_max:
            global_max = batch_max.item()
        
        if max_samples and samples > max_samples:
            break

    return global_min, global_max

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
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", subfolder="tokenizer")
    # text_encoder = BertModel.from_pretrained("bert-base-uncased", subfolder="bert")
    vae = AutoencoderKL.from_pretrained(args.model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet")

    # unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")
    # vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    # noise_scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler")

    global_min, global_max = None, None
    if args.vae_checkpoint and os.path.isfile(args.vae_checkpoint):
        checkpoint_dict = torch.load(args.vae_checkpoint, map_location="cpu")
        vae.load_state_dict(checkpoint_dict['vae_state_dict'])
        global_min = checkpoint_dict['global_min']
        global_max = checkpoint_dict['global_max']
    
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
    stats_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers
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

    if global_min is None and global_max is None:
        if accelerator.is_main_process: 
            global_min, global_max = get_dataset_min_max(stats_dataloader)

            stats_tensor = torch.tensor([global_min, global_max], device=accelerator.device)

        stats_tensor = broadcast(stats_tensor, from_process=0)
    
        global_min = stats_tensor[0].item()
        global_max = stats_tensor[1].item()
    
    # Training Loop
    training = Training(accelerator, unet, vae, text_encoder, tokenizer, noise_scheduler, 
                        train_dataloader, optimizer, args, global_min, global_max, 
                        plot_ecgs=True, 
                        checkpoint_path="spect_ecg_gan/checkpoints/2025-09-14_01-39-31/00200000/g_00200000")
    training.train()

    accelerator.end_training()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Stable Diffusion on ECG Spectrograms")
    parser.add_argument("--model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--save_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./ecg_spect_diff/checkpoints")
    parser.add_argument("--project_name", type=str, default="project")
    parser.add_argument("--vae_checkpoint", type=str, default=None)

    args = parser.parse_args()
    datasetConfig = {
        "data_path": "/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/ecg_spect_diff/parquet_patients/ecgs_patients.parquet",
        "dataDir": "/uufs/sci.utah.edu/projects/ClinicalECGs/AllClinicalECGs/",
        "timeCutOff": 1800,
        "lowerCutOff": 0,
        "randSeed": 7777,
        "scale_training_size": 1.0,
        "kcl_params": {
            "lowThresh": 4.0,
            "highThresh": 5.0,
            "highThreshRestrict": 8.5
        }
    }
    
    current_time = datetime.now()
    project_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = "/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/ecg_spect_diff"
    args.output_dir = f"{base_dir}/checkpoints/{current_time}"
    args.project_name = project_name 
    args.vae_checkpoint = "/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/spect_vae/checkpoints/2025-10-11_17-50-04/00015000/checkpoint_00015000.pt"
    main(args, datasetConfig)
