import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from ecg_spect_diff.utils import plot_image_channels_grid
from ecg_spect_diff.dataset.dataset import SpectrogramExtractor
from ecg_spect_diff.utils import (
    load_generator_model,
    plot_overlapping_ecgs
)

from tqdm import tqdm
import wandb


class VAETrainer:
    def __init__(self, config, accelerator, vae, discriminator, lpips_fn, optimizer, 
                 optimizer_disc, scheduler, scheduler_disc, train_loader, val_loader,
                 global_min, global_max, checkpoint_path=None):
        self.config = config
        self.accelerator = accelerator
        self.step = 0

        self.vae = vae
        self.discriminator = discriminator
        self.lpips_fn = lpips_fn

        self.optimizer = optimizer
        self.optimizer_disc = optimizer_disc
        self.scheduler = scheduler
        self.scheduler_disc = scheduler_disc
        
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.global_min = global_min
        self.global_max = global_max
        
        self.vocoder = None
        self.spectrogram_extractor = None
        
        if self.accelerator.is_main_process:
            if checkpoint_path is None:
                raise "Checkpoint path Not Found"
            else:
                self.generator = load_generator_model(checkpoint_path)
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

    def _get_ecgs_from_spectrograms(self, images):
        B = images.shape[0]
        IMG_SIZE = 256
        images = images.view(-1, IMG_SIZE, IMG_SIZE).to(self.accelerator.device) # B*3, 256, 256
        out = self.spectrogram_extractor(images) # B*3, 8, 65, 126
        out = out.view(-1, 65, 126) # B * 3 * 8, 65, 126
        with torch.no_grad():
            res = self.generator(out) # B * 3 * 8, 1, 2500
        final_ecgs = res.view(B, -1, 8, 2500).detach().cpu() # B ,3, 8, 2500
        return final_ecgs

    def encode_decode(self, images):
        if hasattr(self.vae, 'module'):
            vae_model = self.vae.module
        else:
            vae_model = self.vae
            
        with torch.no_grad():
            latents = vae_model.encode(images).latent_dist.mode()
        reconstruction = vae_model.decode(latents).sample
        return reconstruction
    
    def compute_vae_loss(self, images, reconstruction):
        loss_l2 = F.mse_loss(reconstruction, images)
        
        images_lpips = images * 2 - 1
        recon_lpips = reconstruction * 2 - 1
        loss_lpips = self.lpips_fn(recon_lpips, images_lpips).mean()
        
        # Adversarial Loss
        disc_fake_scores = self.discriminator(reconstruction)
        loss_disc = F.softplus(-disc_fake_scores).mean()
        
        disc_weight = self.config.cost_disc if self.step >= self.config.disc_loss_skip_steps else 0.0
        
        loss = (
            loss_l2 * self.config.cost_l2 +
            loss_lpips * self.config.cost_lpips +
            loss_disc * disc_weight
        )

        return loss, {
            'loss_l2': loss_l2.item(),
            'loss_lpips': loss_lpips.item(),
            'loss_disc': loss_disc.item(),
            'loss_total': loss.item()
        }
    
    def compute_discriminator_loss(self, real_images, fake_images):
        """Compute discriminator loss with R1 gradient penalty"""
        # Detach fake images
        fake_images = fake_images.detach()
        
        # Get discriminator scores
        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)
        
        # Standard GAN loss
        loss_real = F.softplus(-real_scores).mean()
        loss_fake = F.softplus(fake_scores).mean()
        loss_gan = loss_real + loss_fake
        
        # R1 gradient penalty
        real_images.requires_grad_(True)
        real_scores_for_grad = self.discriminator(real_images)
        grad_real = torch.autograd.grad(
            outputs=real_scores_for_grad.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_penalty = (grad_real ** 2).sum(dim=[1, 2, 3]).mean()
        
        loss = loss_gan + self.config.cost_gradient_penalty * grad_penalty
        
        return loss, {
            'disc_loss_real': loss_real.item(),
            'disc_loss_fake': loss_fake.item(),
            'disc_grad_penalty': grad_penalty.item(),
            'disc_loss_total': loss.item(),
            'disc_pred_real': torch.sigmoid(real_scores).mean().item(),
            'disc_pred_fake': torch.sigmoid(-fake_scores).mean().item(),
        }
    
    def train_step(self, batch):
        images = self._normalize_spectrogram(batch['image'])

        with self.accelerator.accumulate(self.vae):
            reconstruction = self.encode_decode(images)
            loss_vae, metrics_vae = self.compute_vae_loss(images, reconstruction)

            self.accelerator.backward(loss_vae)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.vae.parameters(),
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        
        with self.accelerator.accumulate(self.discriminator):
            reconstruction = self.encode_decode(images)
            loss_disc, metrics_disc = self.compute_discriminator_loss(images, reconstruction)
            
            self.accelerator.backward(loss_disc)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.discriminator.parameters(), 
                    self.config.max_grad_norm
                )
            
            self.optimizer_disc.step()
            self.scheduler_disc.step()
            self.optimizer_disc.zero_grad()
        
        metrics = {**metrics_vae, **metrics_disc}
        
        metrics.update({
            'lr': self.scheduler.get_last_lr()[0],
            'lr_disc': self.scheduler_disc.get_last_lr()[0],
        })
        
        
        if self.accelerator.sync_gradients:
            self.step += 1
        
        return metrics, reconstruction
    
    def evaluate(self):
        self.vae.eval()
        losses = []
        
        with torch.no_grad():
            for batch in tqdm(
                self.val_loader, 
                desc="Evaluating",
                disable=not self.accelerator.is_local_main_process
            ):
                images = self._normalize_spectrogram(batch['image'])
                reconstruction = self.encode_decode(images)
                loss = F.mse_loss(reconstruction, images)
                
                # Gather losses from all processes
                gathered_loss = self.accelerator.gather(loss)

                if isinstance(gathered_loss, list):
                    losses.append(torch.stack(gathered_loss).mean())
                else:
                    losses.append(gathered_loss)
        
        self.vae.train()
        
        avg_loss = torch.stack(losses).mean().item()
        return {'val_loss': avg_loss}
            
    def log_images(self, num_images=4, prefix="val"):
        """Log reconstruction images to W&B using spectrogram channel visualization (only on main process)"""
        if not self.accelerator.is_main_process:
            return
        
        self.vae.eval()
        collected_images = []  # Array to collect individual concatenated images

        recon_collected_images = []
        og_collected_ecgs = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                if len(collected_images) >= num_images:
                    break
                
                images = self._normalize_spectrogram(batch['image'])
                reconstruction = self.encode_decode(images)
                reconstruction = self._denormalize_spectrogram(reconstruction)
                ecgs = batch['ecgs']
                
                images_cpu = images.cpu()
                reconstruction_cpu = reconstruction.cpu()
                
                original_single_channel = images_cpu[:, :1, :, :] # First Channel of Original
                unn_single_channel = batch['image'][:,:1,:,:]
                concatenated_images = torch.cat([original_single_channel, reconstruction_cpu], dim=1)
                
                # Add individual images to our collection array
                batch_size = concatenated_images.shape[0]
                for i in range(batch_size):
                    if len(collected_images) >= num_images:
                        break
                    collected_images.append(concatenated_images[i:i+1]) 

                    recon_collected_images.append(reconstruction[i:i+1])
                    og_collected_ecgs.append(ecgs[i:i+1])
                
                del images, reconstruction, images_cpu, reconstruction_cpu, original_single_channel, concatenated_images, unn_single_channel
                torch.cuda.empty_cache()
        
        if collected_images:
            images_tensor = torch.cat(collected_images, dim=0)  # [num_images, 4, height, width]

            figures = plot_image_channels_grid(images_tensor)

            # Log images to wandb
            image_logs = {}
            for i, fig in enumerate(figures):
                image_logs[f"{prefix}_spectrogram_{i}"] = wandb.Image(
                    fig, 
                    caption=f"Spectrogram {i}: Channel 1 (Original, leftmost), Channels 2,3,4 (VAE Reconstruction)"
                )
            
            if image_logs:
                self.accelerator.print(f"Logging {len(image_logs)} spectrogram images to W&B at step {self.step}")
                wandb.log({**image_logs, "step": self.step})
                
                # Clean up figures
                for fig in figures:
                    plt.close(fig)
            else:
                self.accelerator.print(f"No images to log at step {self.step}")
        else:
            self.accelerator.print(f"No images collected to log at step {self.step}")
        
        if og_collected_ecgs:
            og_ecgs = torch.cat(og_collected_ecgs, dim=0)
            recon_collected_images = torch.cat(recon_collected_images, dim=0)
            recon_ecgs = self._get_ecgs_from_spectrograms(recon_collected_images)

            image_logs = {}
            for i in range(og_ecgs.shape[0]):
                ecg_original = og_ecgs[i].unsqueeze(0).detach().cpu()
                recon_ecg = recon_ecgs[i].detach().cpu()
                combined = torch.vstack([recon_ecg, ecg_original])
                fig = plot_overlapping_ecgs(combined, f'og_recon_ecg_{i+1}') 
                image_logs[f"recon_ecg_{i+1}"] = fig
            
            if image_logs:
                self.accelerator.print(f"Logging {len(image_logs)} ecg recons to W&B at step {self.step}")
                wandb.log({**image_logs, "step": self.step})

        self.vae.train()
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:

            unwrapped_vae = self.accelerator.unwrap_model(self.vae)
            unwrapped_disc = self.accelerator.unwrap_model(self.discriminator)
            
            self.accelerator.save({
                'step': self.step,
                'vae_state_dict': unwrapped_vae.state_dict(),
                'discriminator_state_dict': unwrapped_disc.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scheduler_disc_state_dict': self.scheduler_disc.state_dict(),
                'global_min': self.global_min,
                'global_max': self.global_max,
                'config': vars(self.config),
            }, path)
            
            self.accelerator.print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load model states
        unwrapped_vae = self.accelerator.unwrap_model(self.vae)
        unwrapped_disc = self.accelerator.unwrap_model(self.discriminator)
        
        unwrapped_vae.load_state_dict(checkpoint['vae_state_dict'])
        unwrapped_disc.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load optimizer states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
        
        # Load scheduler states
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scheduler_disc.load_state_dict(checkpoint['scheduler_disc_state_dict'])
        
        self.step = checkpoint['step']

        self.global_min = checkpoint['global_min']
        self.global_max = checkpoint['global_max']
        
        self.accelerator.print(f"Checkpoint loaded from {path}, resuming from step {self.step}")
    
    def train(self):
        self.accelerator.print("Strating Training....")
        self.vae.train()
        self.discriminator.train()
        
        pbar = tqdm(
            total = self.config.total_steps,
            desc="Training",
            disable=not self.accelerator.is_local_main_process
        )

        epoch = 0
        while self.step < self.config.total_steps:
            for batch in self.train_loader:
                if self.step >= self.config.total_steps:
                    break
                
                metrics, _ = self.train_step(batch)
                
                if self.step % self.config.log_steps == 0 and self.accelerator.is_main_process:
                    wandb.log({**metrics, 'step': self.step, 'epoch': epoch})
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in list(metrics.items())[:4]})
                
                if self.step % self.config.eval_steps == 0:
                        self.accelerator.wait_for_everyone()  
                        val_metrics = self.evaluate()
                        
                        if self.accelerator.is_main_process:
                            wandb.log({**val_metrics, 'step': self.step})
                            # Only log validation images to reduce memory usage
                            self.log_images(num_images=self.config.num_log_images)
                        
                        self.accelerator.wait_for_everyone()
                    
                # Save checkpoint
                if self.step % self.config.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    self.save_checkpoint(self.config.output_dir / f"checkpoint_{self.step}.pt")
                
                pbar.update(1)   
            
            epoch += 1
        
        pbar.close()
        
        # Final save
        self.accelerator.wait_for_everyone()
        self.save_checkpoint(self.config.output_dir / "final_checkpoint.pt")
        
        # Save only the VAE in HuggingFace format (on main process)
        if self.accelerator.is_main_process:
            unwrapped_vae = self.accelerator.unwrap_model(self.vae)
            unwrapped_vae.save_pretrained(self.config.output_dir / "vae_finetuned")
            self.accelerator.print("Training complete!")
        
        self.accelerator.end_training()
