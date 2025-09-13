import os
import json
import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group
import wandb

from env import AttrDict, build_env
from train import Training


def ddp_setup():
    init_process_group(backend='nccl')

def main():
    print('Intializaing Training Process')

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config_v1.json')
    parser.add_argument('--training_epochs', default=3000, type=int)
    parser.add_argument('--checkpoint_path', default="./")
    parser.add_argument('--logtowandb', default=False, action="store_true")
    
    a = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        ddp_setup()
    
    with open(a.config) as f:
        data = f.read()
    
    json_config = json.loads(data)
    h = AttrDict(json_config)
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
    
    if zero_process:
        wandbrun = wandb.init(
            project = "vocoder",
            notes = "Adapting HifiGAN to ECGs",
            entity = "deekshith",
            reinit = True,
            config = h,
            name = "project",
            resume = "allow"
        )
        run_id = wandbrun.id
        print(f"Run ID: {run_id}")
        
    trainer = Training(a, h)
    trainer.train()
    
    if "LOCAL_RANK" in os.environ:
        destroy_process_group()
    
if __name__ == "__main__":
    main()