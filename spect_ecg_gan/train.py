import itertools
from ntpath import isdir
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import wandb

from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss, generator_loss, feature_loss
from dataset import get_datasets, convert_to_spectrogram
from utils import plot_ecg_leads, plot_spectrogram

class Training:
    def __init__(self, a, h):
        self.h = h
        self.a = a
        
        self.use_ddp = "LOCAL_RANK" in os.environ
        self.gpu_id = 0 if "LOCAL_RANK" not in os.environ else int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.checkpoint_interval = a.checkpoint_interval
        self.validation_interval = a.validation_interval
        self.checkpoint_path = a.checkpoint_path
        self.log_interval = a.logging_interval  # Fix undefined variable
        self.stdout_interval = a.logging_interval  # Fix undefined variable


        self.generator = Generator(self.h).to(self.device)
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)
        self.steps = 0
        self.last_epoch = -1
        
        if self.gpu_id == 0:
            # print(self.generator)
            os.makedirs(a.checkpoint_path, exist_ok=True)
            print(f"Checkpoints Directory: {a.checkpoint_path}")
        
        # Optimizer and Scheduler    
        self.optim_g = torch.optim.AdamW(self.generator.parameters(), self.h.learning_rate, betas=[self.h.adam_b1, self.h.adam_b2])
        self.optim_d = torch.optim.AdamW(itertools.chain(self.msd.parameters(),
                                                         self.mpd.parameters()),
                                         self.h.learning_rate, betas=[self.h.adam_b1, self.h.adam_b2])
        
        # Checkpoint Loading
        if os.path.isdir(self.checkpoint_path) and h['prev_model_step'] is not None:
            step = f"{h['prev_model_step']:08d}"
            state_dict_g = torch.load(f"{self.checkpoint_path}/{step}/g_{step}", map_location=self.device)
            state_dict_do = torch.load(f"{self.checkpoint_path}/{step}/do_{step}", map_location=self.device)
            self.generator.load_state_dict(state_dict_g['generator'])
            self.mpd.load_state_dict(state_dict_do['mpd'])
            self.msd.load_state_dict(state_dict_do['msd'])
            self.steps = state_dict_do['steps'] + 1
            self.last_epoch = state_dict_do['epoch']
            self.optim_g.load_state_dict(state_dict_do['optim_g'])
            self.optim_d.load_state_dict(state_dict_do['optim_d'])

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=self.h.lr_decay)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=self.h.lr_decay)
        
        if self.use_ddp:
            self.generator = DistributedDataParallel(self.generator, device_ids=[self.gpu_id]).to(self.device)
            self.mpd = DistributedDataParallel(self.mpd, device_ids=[self.gpu_id]).to(self.device)
            self.msd = DistributedDataParallel(self.msd, device_ids=[self.gpu_id]).to(self.device)
        
        # Data prep
        self.train_dataset, self.val_dataset = get_datasets(1.0)

        self.train_sampler = DistributedSampler(self.train_dataset) if self.use_ddp else None
        self.train_loader = DataLoader(self.train_dataset, num_workers=self.h.num_workers, shuffle=False,
                                  sampler=self.train_sampler,
                                  batch_size=self.h.batch_size,
                                  pin_memory=True,
                                  drop_last=True)
        if self.gpu_id == 0:
            self.validation_loader = DataLoader(self.val_dataset, num_workers=2, shuffle=False,
                                           sampler=None, batch_size=int(self.h.batch_size), pin_memory=True, drop_last=True)
    
    def _save_checkpoint(self, filepath, obj):
        print(f"Saving Checkpoint to {filepath}")
        torch.save(obj, filepath)
        print("Complete")
        
    def _checkpoint(self, epoch):
        folder_name = f"{self.steps:08d}"
        os.makedirs(f"{self.checkpoint_path}/{folder_name}", exist_ok=True)
        checkpoint_path = f"{self.checkpoint_path}/{folder_name}/g_{self.steps:08d}" 
        self._save_checkpoint(checkpoint_path, 
                             {'generator': (self.generator.module if self.use_ddp else self.generator).state_dict()})
        
        checkpoint_path = f"{self.checkpoint_path}/{folder_name}/do_{self.steps:08d}"
        self._save_checkpoint(checkpoint_path,
                              {
                                  'mpd': (self.mpd.module if self.use_ddp else self.mpd).state_dict(),
                                  'msd': (self.msd.module if self.use_ddp else self.msd).state_dict(),
                                  'optim_g': self.optim_g.state_dict(),
                                  'optim_d': self.optim_d.state_dict(),
                                  'steps': self.steps,
                                  'epoch': epoch
                              })
        
    
    def train(self):
        for epoch in range(max(0, self.last_epoch), self.a.training_epochs):
            if self.gpu_id == 0:
                start = time.time()
                print(f"Epoch: {epoch+1}")
            
            if self.use_ddp:
                self.train_sampler.set_epoch(epoch)
            loss_gen_total = 0.0
            loss_disc_total = 0.0
            loss_mel_total = 0.0
            for i, batch in enumerate(self.train_loader):
                
                self.generator.train()
                self.mpd.train()
                self.msd.train()
                
                if self.gpu_id == 0:
                    start_b = time.time()
                
                x, y = batch
                x  = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                y = y.unsqueeze(1)
                
                y_g_hat = self.generator(x)
                y_g_hat_mel = convert_to_spectrogram(y_g_hat.squeeze(1))

                self.optim_d.zero_grad()

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
                
                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_df_hat_g)
                
                loss_disc_all = loss_disc_s + loss_disc_f
                
                loss_disc_all.backward()
                self.optim_d.step()
                
                
                # Generator
                self.optim_g.zero_grad()
                
                # L1 Mel Spectrogram Loss
                loss_mel = F.l1_loss(x, y_g_hat_mel)   # Check this again 

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * 45

                
                loss_gen_total += loss_gen_all.item()
                loss_disc_total += loss_disc_all.item()
                loss_mel_total += loss_mel.item()
                
                loss_gen_all.backward()
                self.optim_g.step()
                
                # Add synchronization barrier to prevent NCCL timeouts
                if self.use_ddp:
                    dist.barrier()

                # Evaluation and Logging
                if self.gpu_id == 0:
                    training_log = None 
                                        
                    # Checkpointing 
                    if self.steps % self.checkpoint_interval == 0:
                        self._checkpoint(epoch)
                        print("Saving Done")
                    
                    # Validation
                    validation_string = None
                    if self.steps % self.validation_interval == 0:
                        print("Validation Starting")
                        if training_log is None:
                            training_log = {}
                        validation_plot = False
                        self.generator.eval()
                        torch.cuda.empty_cache()
                        val_err_total = 0
                        print("Validation Loop Started")
                        
                        with torch.no_grad():
                            for j, batch in enumerate(self.validation_loader):
                                print(f"Validation number {j+1}/{len(self.validation_loader)}", end="\r")
                                x_test, y_test = batch
                                y_g_hat_test = self.generator(x_test.to(self.device))
                                y_mel_test = x_test
                                y_g_hat_mel_test = convert_to_spectrogram(y_g_hat_test.squeeze(1))
                                
                                val_err_total += F.l1_loss(y_mel_test.to(self.device), y_g_hat_mel_test).item()

                                if not validation_plot:
                                    fig = plot_spectrogram(x_test[0].detach().cpu(), y_g_hat_mel_test[0].detach().cpu(), prefix="val")
                                    training_log['val_spectrograms'] = fig
                                    
                                    fig = plot_ecg_leads(y_test[0].squeeze().detach().cpu(), y_g_hat_test[0].squeeze().detach().cpu(), prefix="val")
                                    training_log['val_ecgs'] = fig

                                    validation_plot = True
                            
                            val_err = val_err_total / (j+1)
                            training_log['val_err'] = val_err
                            validation_string = f" Validation Error: {val_err:4.6f}"
                                
                    # Wandb Logging        
                    if self.steps % self.log_interval == 0:
                        # wandb log
                        if training_log is None:
                            training_log = {}
                        
                        training_log['steps'] = self.steps
                        training_log['loss_disc'] = loss_disc_total / self.log_interval 
                        training_log['loss_gen'] = loss_gen_total / self.log_interval 
                        training_log['loss_mel_recon'] = loss_mel_total / self.log_interval 
                        loss_disc_total, loss_gen_total, loss_mel_total = 0.0, 0.0, 0.0
                        
                        fig = plot_spectrogram(x[0].detach().cpu(), y_g_hat_mel[0].detach().cpu(), prefix="train")
                        training_log['spectrograms'] = fig
                        
                        fig = plot_ecg_leads(y[0].squeeze().detach().cpu(), y_g_hat[0].squeeze().detach().cpu(), prefix="train") 
                        training_log['ecgs'] = fig
                        
                    if training_log:
                        wandb.log(training_log)
                    
                    # STDOUT Logging
                    if self.steps % self.stdout_interval == 0:
                        out_str =f'Steps: {self.steps}, Gen Loss Total : {loss_gen_all:4.3f}, Mel-Spec. Error: {loss_mel:4.3f}, s/b: {(time.time() - start_b):4.3f}' 
                        out_str = out_str if validation_string is None else out_str + validation_string
                        print(out_str)

                    self.generator.train()
                
                
                
                self.steps += 1
            
            self.scheduler_d.step()
            self.scheduler_g.step()
            
            if self.gpu_id == 0:
                print(f"Time Taken for epoch {epoch+1} is {int(time.time()-start)} sec\n")

                