import datetime
import os
import json
import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group
import wandb

from spect_ecg_gan.env import AttrDict, build_env
from spect_ecg_gan.train import Training


def ddp_setup():
    # Set NCCL environment variables for better stability
    os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30 minutes timeout
    os.environ.setdefault('NCCL_DEBUG', 'INFO')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')  # Disable InfiniBand if causing issues
    os.environ.setdefault('NCCL_SOCKET_IFNAME', '^docker0,lo')  # Exclude problematic interfaces
    
    init_process_group(backend='nccl')

def main():
    print('Intializaing Training Process')

    ##### Arguments Setup #######
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./spect_ecg_gan/config_v1.json')
    parser.add_argument('--training_epochs', default=3000, type=int)
    parser.add_argument('--logtowandb', default=False, action="store_true")
    parser.add_argument('--logging_interval', default=50, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int) 
    
    a = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        ddp_setup()
    
   ################### Config Setup ############################# 
    with open(a.config) as f:
        data = f.read()
    
    json_config = json.loads(data)
    h = AttrDict(json_config)
    

    #################### Checkpoint Setup #################### 
    baseDir = "/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/spect_ecg_gan/checkpoints"
    # h.checkpoint is in format (formattedtime, wandbrunid, step)
    wandbrunid = None
    prev_model_step = None
    if len(h.prev_model) :
       project_name = h.prev_model[0] 
       wandbrunid = h.prev_model[1]
       if len(h.prev_model) > 2:
           prev_model_step = int(h.prev_model[2])
           step = f"{prev_model_step:08d}"
           assert os.path.isfile(f"{baseDir}/{project_name}/{step}/do_{step}"), "File mentioned in the config doesn't exist"
       else:
            # Find the biggest checkpoint available in the folder
           checkpoint_dir = f"{baseDir}/{project_name}"
           if os.path.exists(checkpoint_dir):
               # Get all directories in the checkpoint folder
               checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) 
                                if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.isdigit()]
               if checkpoint_dirs:
                   # Find the directory with the highest number
                   prev_model_step = max(int(d) for d in checkpoint_dirs)
                   print(f"Using highest checkpoint: {prev_model_step}")
               else:
                   print("No checkpoint directories found")
                   return
           else:
               print(f"Checkpoint directory {checkpoint_dir} does not exist")
               return

    else:
        current_time = datetime.datetime.now()
        project_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    a.checkpoint_path = f"{baseDir}/{project_name}"
    print(f"Checkpoint Path is {a.checkpoint_path}")
    h['prev_model_step'] = prev_model_step
    
    build_env(a.config, 'config.json', a.checkpoint_path)

    #################### Seed and Launch Training ####################
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print(f'Batch Size per GPU: {h.batch_size}')
    else:
        pass
    
    zero_process = False
    if "LOCAL_RANK" in os.environ:
        if int(os.environ["LOCAL_RANK"]) == 0:
            zero_process = True
    else:
        zero_process = True
    
    if zero_process and a.logtowandb:
        wandbrun = wandb.init(
            project = "vocoder",
            notes = "Adapting HifiGAN to ECGs",
            entity = "deekshith",
            reinit = True,
            config = h | vars(a),
            name = project_name,
            resume = "allow",
            id=wandbrunid
        )
        run_id = wandbrun.id
        print(f"Run ID: {run_id}")
        
    try:
        trainer = Training(a, h)
        trainer.train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if "LOCAL_RANK" in os.environ:
            destroy_process_group()
    
if __name__ == "__main__":
    main()