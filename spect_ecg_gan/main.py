import datetime
import os
import json
import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group
import wandb

from env import AttrDict, build_env
from train import Training


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

    parser.add_argument('--config', default='config_v1.json')
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
    
    baseDir = "/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/spect_ecg_gan/checkpoints"
    # h.checkpoint is in format (formattedtime, step, wandbrunid)
    wandbrunid = None
    checkpoint_step = None
    if len(h.checkpoint) :
       project_name = h.checkpoint_path[0] 
       checkpoint_step = h.checkpoint_path[1]
       wandbrunid = h.checkpoint_path[2]
    else:
        current_time = datetime.datetime.now()
        project_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    a.checkpoint_path = f"{baseDir}/{project_name}"
    h['checkpoint_step'] = checkpoint_step
    print(f"Checkpoint Path is {a.checkpoint_path}")
    
    build_env(a.config, 'config.json', a.checkpoint_path)

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