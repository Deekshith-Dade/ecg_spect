from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from diffusers import AutoencoderKL
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, broadcast
from tqdm import tqdm
import lpips
import argparse
from datetime import datetime

from ecg_spect_diff.main import get_dataset_min_max
from spect_vae.dataset import get_datasets
from spect_vae.train import VAETrainer


class Config:
    def __init__(self, config_dict=None):
        d = {}
        if config_dict is not None:
            d.update(config_dict)
        self.__dict__.update(d)
        # ensure output_dir exists (if it's a Path)
        if isinstance(self.output_dir, Path):
            self.output_dir.mkdir(exist_ok=True)
    def to_dict(self):
        return dict(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

class SimpleDiscriminator(nn.Module):
    """Simplified StyleGAN-style discriminator"""
    
    def __init__(self, image_size=256, base_channels=64):
        super().__init__()
        
        channels = [3, base_channels, base_channels*2, base_channels*4, 
                   base_channels*8, base_channels*8]
        
        layers = []
        for i in range(len(channels)-1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        
        self.convs = nn.Sequential(*layers)
        
        # Calculate final spatial size
        final_size = image_size // (2 ** (len(channels)-1))
        self.final = nn.Linear(channels[-1] * final_size * final_size, 1)
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="VAE Fine-tuning with Accelerate")
    parser.add_argument("--output_dir", type=str, default="output-dir-vae")
    parser.add_argument("--vae_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--total_steps", type=int, default=150000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    return parser.parse_args()


def main(args, config):

        
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_dir=config.logging_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    # Set seed consistently across all ranks
    set_seed(config.seed)
    accelerator.wait_for_everyone()

    accelerator.init_trackers(
        project_name="vae-finetune",
        config= vars(args) | config.to_dict(),
        init_kwargs = {"wandb": {
            "entity": "deekshith",
            "resume": "allow",
            "name": args.project_name
        }}
    )

    train_ds, val_ds = get_datasets(1.0)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # VAE
    accelerator.print("Initializing VAE")
    vae = AutoencoderKL.from_pretrained(config.vae_model_id, subfolder="vae")
    for param in vae.encoder.parameters():
            param.requires_grad = False
    for param in vae.quant_conv.parameters():
        param.requires_grad = False
    
    trainable_params = (
        list(vae.decoder.parameters())+
        list(vae.post_quant_conv.parameters())
    )
    
    # Discriminator
    accelerator.print("Initializing Discriminator")
    discriminator = SimpleDiscriminator()

    # LPIPS
    accelerator.print("Loading LPIPS")
    lpips_fn = lpips.LPIPS(net="vgg")
    for param in lpips_fn.parameters():
        param.requires_grad = False
    
    # optimizers
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    optimizer_disc = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.learning_rate_disc,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Schedulers
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps
    )
    scheduler_disc = get_linear_schedule_with_warmup(
        optimizer_disc,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps
    )

    
    #### Prepare Accelerator #####
    vae, discriminator, lpips_fn, optimizer, optimizer_disc, scheduler, scheduler_disc, train_loader, val_loader = accelerator.prepare(
        vae, discriminator, lpips_fn, optimizer, optimizer_disc, scheduler, scheduler_disc, train_loader, val_loader
    )
    
    stats_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    stats_tensor = torch.zeros(2, device=accelerator.device)

    if accelerator.is_main_process:
        global_min, global_max = get_dataset_min_max(stats_dataloader, max_samples=200000)

        stats_tensor = torch.tensor([global_min, global_max], device=accelerator.device)

    stats_tensor = broadcast(stats_tensor, from_process=0)
    
    global_min = stats_tensor[0].item()
    global_max = stats_tensor[1].item()
    
    if accelerator.is_main_process:
            trainable_count = sum(p.numel() for p in trainable_params)
            disc_count = sum(p.numel() for p in discriminator.parameters())
            accelerator.print(f"Trainable parameters: {trainable_count:,}")
            accelerator.print(f"Discriminator parameters: {disc_count:,}")
            accelerator.print(f"Number of processes: {accelerator.num_processes}")
            accelerator.print(f"Mixed precision: {config.mixed_precision}")

    trainer = VAETrainer(config, accelerator, vae, discriminator,
                         lpips_fn, optimizer, optimizer_disc, scheduler, scheduler_disc,
                         train_loader, val_loader, global_min, global_max,
                         checkpoint_path="spect_ecg_gan/checkpoints/2025-09-14_01-39-31/00200000/g_00200000")
    
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # Train
    trainer.train()

default_config_dict = {
    # Paths
    "output_dir": Path("output-dir-vae"),
    "vae_model_id": "runwayml/stable-diffusion-v1-5",

    # Training hyperparameters
    "learning_rate": 1e-5,
    "learning_rate_disc": 1e-4,
    "batch_size": 8,  
    "gradient_accumulation_steps": 1,
    "num_epochs": 10,
    "total_steps": 150_000,
    "warmup_steps": 4000,
    "max_grad_norm": 1.0,  # Gradient clipping

    # Loss weights
    "cost_l2": 0.5,
    "cost_lpips": 5.0,
    "cost_disc": 0.005,
    "cost_gradient_penalty": 10.0,

    # Training strategy
    "disc_loss_skip_steps": 1000,

    # Logging
    "log_steps": 10,
    "eval_steps": 1000,
    "save_steps": 10000,
    "num_log_images": 2,

    # Hardware
    "num_workers": 4,
    "seed": 42,
    "mixed_precision": "fp16",  # "no", "fp16", "bf16"

    # Accelerate specific
    "logging_dir": "logs",
    "report_to": "wandb",  # "wandb", "tensorboard", "all"
}

 
if __name__ == "__main__":
    args = parse_args()
    
    # Update config with args
    config = Config(default_config_dict)
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    current_time = datetime.now()
    project_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = "/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/spect_vae"
    config.output_dir = f"{base_dir}/checkpoints/{current_time}"
    args.project_name = project_name 
    main(args, config)