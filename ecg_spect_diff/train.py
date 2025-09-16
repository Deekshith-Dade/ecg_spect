import os
from PIL import Image
from diffusers.pipelines.deepfloyd_if import safety_checker
import torch

from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline

import wandb

from ecg_spect_diff.utils import plot_image_channels_grid, resize_and_stack_images

class Training:
    def __init__(self, accelerator, unet, vae, text_encoder, tokenizer, noise_scheduler, train_dataloader, optimizer, args, global_min, global_max):
        self.accelerator = accelerator
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.dataloader = train_dataloader
        self.optimizer = optimizer
        self.args = args
        self.global_min = global_min
        self.global_max = global_max
        print(f"Values {self.global_min}, {self.global_max} from {self.accelerator.device}")
    
    def _normalize_spectrogram(self, spec):
        spec_0_1 = (spec - self.global_min) / (self.global_max - self.global_min)
        spec_neg1_1 = spec_0_1 * 2.0 - 1.0
        return spec_neg1_1
    
    def _denormalize_spectrogram(self, spec):
        spec_0_1 = (spec + 1.0) / 2.0
        original_spec = spec_0_1 * (self.global_max - self.global_min) + self.global_min
        return original_spec
    
    def train(self):
        for epoch in range(self.args.num_train_epochs):
            train_loss = self.train_one_epoch(epoch)
            
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch}: Average Loss = {train_loss}")
            
            # Checkpointing
            if epoch == self.args.num_train_epochs - 1 and self.accelerator.is_main_process:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.args.model_name_or_path,
                    unet=self.accelerator.unwrap_model(self.unet),
                    text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                    vae=self.accelerator.unwrap_model(self.vae),
                    tokenizer=self.tokenizer,
                    scheduler=self.noise_scheduler
                )
                pipeline.save_pretrained(self.args.output_dir)

    
    def train_one_epoch(self, epoch):
        self.unet.train()
        total_loss = 0.
        for step, batch in enumerate(tqdm(self.dataloader, desc=f"Training Epoch {epoch}", disable=not self.accelerator.is_main_process)):
            with self.accelerator.accumulate(self.unet):
                data = self._normalize_spectrogram(batch['image'])
                latents = self.vae.encode(data).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz, ), device=self.accelerator.device)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                encoding = self.tokenizer(batch["cond_inputs"]["text"],
                                    truncation=True,
                                    max_length=self.tokenizer.model_max_length,
                                    padding="max_length",
                                    return_tensors="pt")
                
                input_ids = encoding.input_ids.to(self.accelerator.device)
                encoder_hidden_states = self.text_encoder(input_ids)[0]
                
                noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").mean()
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.detach().item()
            
            if self.accelerator.is_main_process and step % self.args.logging_steps == 0:
                print(f"Running Sampling")
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.args.model_name_or_path,
                    unet=self.accelerator.unwrap_model(self.unet),
                    text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                    vae=self.accelerator.unwrap_model(self.vae),
                    tokenizer=self.tokenizer,
                    scheduler=self.noise_scheduler,
                    safety_checker = None,
                    requires_safety_checker = False
                )
                pipeline = pipeline.to(self.accelerator.device)
                val_images = self.sample_images(pipeline, 4)
                figs = plot_image_channels_grid(val_images)
                
                # Calculate average loss up to this point
                avg_loss = total_loss / (step + 1)
                
                # Log to wandb
                self.accelerator.log({
                    "train/loss": avg_loss,
                    "train/step": step,
                    "train/epoch": epoch,
                    "samples": [wandb.Image(figs)]
                })
                
            if step % self.args.save_steps == 0:
                self.save_progress(epoch, step)
                
        return  total_loss / len(self.dataloader)
    
    def save_progress(self, epoch, step):
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.args.output_dir, f'checkpoint-{epoch}-{step}')
            self.accelerator.save_state(save_path)


    def sample_images(self, pipeline, num_images=4):
        pipeline.unet.eval()

        generator = torch.Generator(device=pipeline.device).manual_seed(42)
        prompts = ["ECG with severe hyperkalemia", "ECG with healthy signs"] * int(num_images/2)
        
        with torch.no_grad():
            images = pipeline(
                prompts,
                num_inference_steps=50,
                generator=generator
            ).images
       
        images = resize_and_stack_images(images)
        images = self._denormalize_spectrogram(images)
        
        
        pipeline.unet.train()
        
        return images