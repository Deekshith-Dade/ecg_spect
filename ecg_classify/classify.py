import os
import logging
from datetime import datetime
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from ecg_classify.dataset.data_utils import dataprepKCL, splitKCLPatients
from ecg_classify.training import trainNetwork_balancedClassification, loss_bce_kcl
from ecg_classify.networks import ECG_SpatioTemporalNet1D
from ecg_classify.parameters import spatioTemporalParams_1D

def setup_logger(log_file, exp_name):
    logger = logging.getLogger(f'classify_{exp_name}')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def ddp_setup():
    """Initialize distributed training.
    
    When using torchrun, the process group is automatically initialized.
    This function sets NCCL environment variables and ensures the device is set correctly.
    """
    # Set NCCL environment variables for better stability
    os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30 minutes timeout
    os.environ.setdefault('NCCL_DEBUG', 'INFO')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')  # Disable InfiniBand if causing issues
    os.environ.setdefault('NCCL_SOCKET_IFNAME', '^docker0,lo')  # Exclude problematic interfaces
    
    # torchrun automatically initializes the process group, but if we're not using torchrun,
    # we need to initialize it manually
    if not dist.is_initialized():
        # Manual initialization (e.g., when using srun or other launchers)
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '12355')
        
        init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_addr}:{master_port}',
            rank=rank,
            world_size=world_size
        )
    
    # Set the device for this process
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def main():
    # Initialize DDP if running in distributed mode
    ddp_enabled = "LOCAL_RANK" in os.environ
    if ddp_enabled:
        ddp_setup()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f"cuda:{local_rank}")
        is_main = is_main_process()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
    
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
    
    # Can be a single path (string) or list of paths (list of strings)
    # The ECG_Gen_DatasetLoader will combine data from all paths into a single dataset
    gen_data_path = [
        "generated_ecgs/25k_gen.pth",
        "generated_ecgs/50k_gen.pth"
    ]  # You can also use a single path: "generated_ecgs/50k_gen.pth"
    model_save_dir = "classify_models"
    log_dir = "classify_logs"
    
    # Only main process creates directories
    if is_main:
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
    # Wait for main process to create directories
    if ddp_enabled:
        dist.barrier()
    
    seeds = [42, 123, 456]
    use_augmented = [True]  # Can be changed to [False, True] or [False] for different experiments
    num_gen_samples_list = [75000]  # Number of generated ECGs to add: 75K (combining 25K + 50K files)
    
    lossParams = {
        'lowThresh': 4.0,
        'highThresh': 5.0
    }
    
    num_epochs = 15
    # Batch size is per GPU, so total batch size = batch_size * world_size
    batch_size_per_gpu = 32 * 3
    learning_rate = 0.0001 / 3
    leads = list(range(8))
    num_workers = 4
    
    splits_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'splits')
    if is_main:
        os.makedirs(splits_dir, exist_ok=True)
    
    # Wait for main process to create splits directory
    if ddp_enabled:
        dist.barrier()
    
    # Only main process creates splits
    if is_main:
        for seed in seeds:
            split_file = os.path.join(splits_dir, f'{seed}_test_patients.pkl')
            if not os.path.exists(split_file):
                if is_main:
                    print(f"Creating splits for seed {seed}")
                splitKCLPatients(seed, datasetConfig)
                if is_main:
                    print(f"Splits created for seed {seed}")
    
    # Wait for splits to be created
    if ddp_enabled:
        dist.barrier()
    
    for seed in seeds:
        for use_aug in use_augmented:
            # If using augmentation, iterate over different numbers of generated samples
            # Otherwise, just run one experiment without augmentation
            samples_to_iterate = num_gen_samples_list if use_aug else [None]
            
            for num_gen_samples in samples_to_iterate:
                # Add human-readable timestamp to distinguish experiments
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if use_aug:
                    exp_name = f"{timestamp}_seed_{seed}_aug_True_gen_{num_gen_samples//1000}k"
                else:
                    exp_name = f"{timestamp}_seed_{seed}_aug_False"
                log_file = os.path.join(log_dir, f"{exp_name}.log")
                
                # Only main process sets up logger
                if is_main:
                    logger = setup_logger(log_file, exp_name)
                    logger.info(f"Starting experiment: {exp_name}")
                    logger.info(f"Seed: {seed}, Use Augmented: {use_aug}")
                    if use_aug:
                        logger.info(f"Generated samples: {num_gen_samples}")
                    logger.info(f"DDP enabled: {ddp_enabled}, World size: {world_size}, Rank: {rank}")
                    logger.info(f"Batch size per GPU: {batch_size_per_gpu}, Total batch size: {batch_size_per_gpu * world_size}")
                else:
                    logger = logging.getLogger(f'classify_{exp_name}')
                    logger.setLevel(logging.WARNING)  # Suppress non-main process logs
                
                # Prepare data loaders with DDP support
                gen_path = gen_data_path if use_aug else None
                num_gen = num_gen_samples if use_aug else None
                
                trainNormalLoader, trainAbnormalLoader, testLoader = dataprepKCL(
                    seed=seed,
                    dataDir=datasetConfig['dataDir'],
                    batch_size=batch_size_per_gpu,
                    kclTaskParams=datasetConfig['kcl_params'],
                    gen_data_path=gen_path,
                    num_gen_samples=num_gen,
                    num_workers=num_workers,
                    ddp_enabled=ddp_enabled,
                    rank=rank,
                    world_size=world_size
                )
                
                if is_main:
                    logger.info(f"Train Normal samples: {len(trainNormalLoader.dataset)}")
                    logger.info(f"Train Abnormal samples: {len(trainAbnormalLoader.dataset)}")
                    logger.info(f"Test samples: {len(testLoader.dataset)}")
                
                network = ECG_SpatioTemporalNet1D(
                    **spatioTemporalParams_1D,
                    classification=True,
                    avg_embeddings=True
                )
                network = network.to(device)
                
                # Wrap model with DDP if enabled
                if ddp_enabled:
                    network = DDP(network, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
                    model_for_saving = network.module  # Access the underlying model for saving
                else:
                    model_for_saving = network
                
                optimizer = optim.Adam(network.parameters(), lr=learning_rate)
                
                # Only main process initializes wandb
                log_to_wandb = False
                if is_main:
                    try:
                        import wandb
                        wandb.init(
                            project="ecg-classify",
                            name=exp_name,
                            id=exp_name,
                            # resume="never",
                            config={
                                'seed': seed,
                                'use_augmented': use_aug,
                                'num_gen_samples': num_gen_samples if use_aug else None,
                                'batch_size_per_gpu': batch_size_per_gpu,
                                'total_batch_size': batch_size_per_gpu * world_size,
                                'world_size': world_size,
                                'learning_rate': learning_rate,
                                'num_epochs': num_epochs
                            }
                        )
                        log_to_wandb = True
                    except Exception as e:
                        log_to_wandb = False
                        logger.info(f"wandb not available or failed to initialize: {e}, skipping wandb logging")
                
                try:
                    best_auc = trainNetwork_balancedClassification(
                        network=network,
                        trainDataLoader_normals=trainNormalLoader,
                        trainDataLoader_abnormals=trainAbnormalLoader,
                        testDataLoader=testLoader,
                        numEpoch=num_epochs,
                        optimizer=optimizer,
                        lossFun=loss_bce_kcl,
                        lossParams=lossParams,
                        leads=leads,
                        modelSaveDir=model_save_dir,
                        modelName=exp_name,
                        logger=logger,
                        logToWandB=log_to_wandb,
                        device=device,
                        ddp_enabled=ddp_enabled,
                        is_main_process=is_main,
                        model_for_saving=model_for_saving
                    )
                    
                    if is_main:
                        logger.info(f"Experiment {exp_name} completed. Best AUC: {best_auc}")
                except Exception as e:
                    if is_main:
                        logger.error(f"Experiment {exp_name} failed with error: {e}", exc_info=True)
                    raise
                finally:
                    if log_to_wandb and is_main:
                        try:
                            import wandb
                            wandb.finish()
                        except:
                            pass
    
    # Cleanup DDP
    if ddp_enabled:
        destroy_process_group()

if __name__ == "__main__":
    main()

