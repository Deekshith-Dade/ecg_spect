import os
import torch
from tqdm.auto import tqdm

import wandb
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from spect_ecg_gan.models import Generator
from ecg_spect_diff.utils import (
    plot_image_channels_grid, 
    resize_and_stack_images, 
    load_generator_model, 
    plot_overlapping_ecgs
)
from ecg_spect_diff.dataset.dataset import SpectrogramExtractor


class Training:
    def __init__(self, accelerator, unet, vae, text_encoder, tokenizer, noise_scheduler, train_dataloader, optimizer, args, global_min, global_max, plot_ecgs=True, checkpoint_path=None):
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
        self.plot_ecgs = plot_ecgs
        self.checkpoint_path = checkpoint_path
        print(f"Values {self.global_min}, {self.global_max} from {self.accelerator.device}")
        self.generator = None
        self.spectrogram_extractor = None
        if self.accelerator.is_main_process and self.plot_ecgs:
            if self.checkpoint_path is None:
                raise "Checkpoing path Not Found Error"
            else:
                self.generator = load_generator_model(self.checkpoint_path)
                self.generator = self.generator.to(self.accelerator.device)
                self.generator.eval()
                self.spectrogram_extractor = SpectrogramExtractor()
            
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
                # latents = self.vae.encode(data).latents
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

                # pipeline = DiffusionPipeline.from_pretrained(
                #     "CompVis/ldm-celebahq-256",
                #     unet=self.accelerator.unwrap_model(self.unet),
                #     vae=self.accelerator.unwrap_model(self.vae),
                #     scheduler=self.noise_scheduler,
                #     safety_checker=None,
                #     requires_safety_checker = False
                # )
                pipeline = pipeline.to(self.accelerator.device)
                val_images = self.sample_images(pipeline, 4)
                full_figs = plot_image_channels_grid(val_images)

                # Plot Original Spectrogram to Compare
                og_figs = plot_image_channels_grid(batch['image'][:2])
                
                
               # Calculate average loss up to this point 
                avg_loss = total_loss / (step + 1)
                training_log = {
                    "train/loss": avg_loss,
                    "train/step": step,
                    "train/epoch": epoch,
                }
                if self.plot_ecgs:
                    ecg_plots = self.get_ecg_plots(val_images)
                    for i, fig in enumerate(ecg_plots):
                        training_log[f'ecg_sample_{i+1}'] = fig

                    # Plot original ecgs to compare
                    og_ecgs = batch['ecgs'][:2] #(2, 8, 2500)
                    recon_ecgs = self._get_ecgs_from_spectrograms(batch['image'][:2])
                    for i in range(og_ecgs.shape[0]):
                        ecg_original = og_ecgs[i].unsqueeze(0).detach().cpu()
                        recon_ecg = recon_ecgs[i].detach().cpu()
                        combined = torch.vstack([recon_ecg, ecg_original])
                        fig = plot_overlapping_ecgs(combined, f'og_recon_ecg_{i+1}') 
                        training_log[f'og_recon_ecg_{i+1}'] = fig
                    
                    # Plot ecgs from spectrograms to compare Generator Quality
                    # ecg_plots = self.get_ecg_plots(batch['image'][:2])
                    # for i, fig in enumerate(ecg_plots):
                    #     training_log[f'ecg_spect_{i+1}'] = fig

                for i, fig in enumerate(full_figs):
                    training_log[f'sample_{i+1}'] = wandb.Image(fig)
                
                for i, fig in enumerate(og_figs):
                    training_log[f'original_{i+1}'] = wandb.Image(fig)
                
                # Log to wandb
                self.accelerator.log(training_log)
                
            if step % self.args.save_steps == 0:
                self.save_progress(epoch, step)
                
        return  total_loss / len(self.dataloader)
    
    def _get_ecgs_from_spectrograms(self, images):
        B = images.shape[0]
        IMG_SIZE = 256
        images = images.view(-1, IMG_SIZE, IMG_SIZE).to(self.accelerator.device) # B*3, 256, 256
        out = self.spectrogram_extractor(images) # B * 3, 8, 65, 126
        out = out.view(-1, 65, 126) # B * 3 * 8, 65, 126
        with torch.no_grad():
            res = self.generator(out) # B * 3 * 8, 1, 2500
        final_ecgs = res.view(B, -1, 8, 2500).detach().cpu() # B , 3, 8, 2500
        return final_ecgs

    def get_ecg_plots(self, images):
        final_ecgs = self._get_ecgs_from_spectrograms(images)
        B = final_ecgs.shape[0]
        out = []
        for i in range(B):
            fig = plot_overlapping_ecgs(final_ecgs[i], f"sample_{i+1}")
            out.append(fig)
            
        return out
        
    
    def save_progress(self, epoch, step):
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.args.output_dir, f'checkpoint-{epoch}-{step}')
            self.accelerator.save_state(save_path)

    def sample_images(self, pipeline, num_images=2):
        pipeline.unet.eval()
        IMG_SIZE = 256

        generator = torch.Generator(device=pipeline.device).manual_seed(42)
        prompts = ["ECG that shows severe stage hyperkalemia", "ECG that shows normal signs of hyperkalemia"] * int(num_images/2)
        # prompts = ["an ECG spectrogram", "an ECG spectrogram"] * int(num_images/2)
        with torch.no_grad():
            images = pipeline(
                prompts,
                height=IMG_SIZE,
                width=IMG_SIZE,
                num_inference_steps=100,
                # generator=generator
            ).images
       
        images = resize_and_stack_images(images)
        images = images * 2.0 - 1.0
        images = self._denormalize_spectrogram(images)
        
        
        pipeline.unet.train()
        
        return images