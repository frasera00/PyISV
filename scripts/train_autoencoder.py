# This script trains an autoencoder model using PyTorch and the PyISV library.

import os
import datetime
import time 
import logging
import datetime
import numpy as np
import warnings
import json
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional

from PyISV.training_utils import *
from PyISV.set_architecture import import_config # type: ignore
from PyISV.neural_network import NeuralNetwork 
from PyISV.validation_utils import Validator   

# Set paths to directories and ID 
from PyISV.define_root import PROJECT_ROOT as root_dir

# Profiling utilities for performance analysis
import torch.utils.bottleneck
import torch.profiler

class Trainer():
    def __init__(self, config_file: str, params: Optional[dict] = None) -> None:
        self.config = import_config(config_file) if params is None else params

        self.device = get_device(device=self.config['GENERAL']['device'])
        self.run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = None
        self.best_val_loss = None

        # Dir paths
        self.model_dir = f"{root_dir}/models/{self.run_id}"
        self.logs_dir = f"{root_dir}/logs/{self.run_id}"
        self.outputs_dir = f"{root_dir}/outputs/{self.run_id}"
        self.norms_dir = f"{root_dir}/norms/{self.run_id}"
        self.data_dir = f"{root_dir}/datasets"

        # File names
        self.model_file = f"{self.model_dir}/model.pt"
        self.stats_file = f"{self.model_dir}/stats/stats.dat"
        self.arch_file = f"{self.model_dir}/arch.dat"

        self.norm_files = {
            "divval_inputs": f"input_autoen_scaler_subval.npy",
            "subval_inputs": f"input_autoen_scaler_divval.npy",
            "divval_targets": f"output_autoen_scaler_subval.npy",
            "subval_targets": f"output_autoen_scaler_divval.npy"
        }

        # Save config file in the model directory
        with open(f'{self.model_dir}/config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

        # Set validator, DDP, and run
        self.validator = Validator(self.config)
        self._setup_ddp()

        if is_main_process():
            print(f"\nTraining on {self.device} with run ID: {self.run_id}\n")
            
    def run_training(self) -> None:
        logging.info("Starting training process...")
        self.prepare_data()
        logging.info("Data prepared.")
        self.prepare_model()
        logging.info("Model prepared.")
        self.train()
        self._cleanup()
        logging.info("Training process completed.")

    def _cleanup(self) -> None:
        if self.config['GENERAL']['use_ddp']:
            dist.destroy_process_group()
            torch.cuda.empty_cache()
    
    def _setup_ddp(self) -> None:
        if self.config['GENERAL']['use_ddp']:
            # Get rank and world size from SLURM environment variables if available
            if 'SLURM_PROCID' in os.environ:  # We're running under SLURM
                local_rank = int(os.environ.get('SLURM_LOCALID', 0))
                global_rank = int(os.environ.get('SLURM_PROCID', 0))
                world_size = int(os.environ.get('SLURM_NTASKS', 1))
                # Set PyTorch distributed environment variables that might not be set by SLURM
                os.environ['RANK'] = str(global_rank)
                os.environ['LOCAL_RANK'] = str(local_rank)
                os.environ['WORLD_SIZE'] = str(world_size)
                # Use SLURM_NODELIST to get master address if not already set
                if 'MASTER_ADDR' not in os.environ:
                    import subprocess
                    result = subprocess.run(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], 
                                           stdout=subprocess.PIPE, text=True)
                    os.environ['MASTER_ADDR'] = result.stdout.strip().split('\n')[0]
                # Use a unique port per SLURM job to avoid collisions
                if 'MASTER_PORT' not in os.environ:
                    os.environ['MASTER_PORT'] = str(2**15 + (int(os.environ.get('SLURM_JOBID', '0')) % 2**15))
            else:
                # Fallback for non-SLURM execution (e.g., direct torchrun calls)
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                global_rank = int(os.environ.get('RANK', 0))
                world_size = int(os.environ.get('WORLD_SIZE', 1))
            
            print(f"Using DDP with local rank {local_rank} and world size {world_size}")
            
            # Try to use CCL if available, else fall back to nccl/gloo
            try:
                import intel_extension_for_pytorch
                import oneccl_bindings_for_pytorch
                backend = "ccl"
                if is_main_process():
                    print("Using CCL backend for DDP.")
            except ImportError:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                if is_main_process():
                    print(f"Using backend: {backend}")

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f'cuda:{local_rank}')
            else:
                # For CPU DDP, set device to cpu and optionally set thread affinity
                self.device = torch.device('cpu')
                num_threads = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 1))
                print(f"Using DDP, with {num_threads} CPU threads.")
                torch.set_num_threads(num_threads)
                torch.set_num_interop_threads(2)
            dist.init_process_group(backend=backend)
            torch.distributed.barrier()
        else:
            num_threads = os.cpu_count()
            print(f"Using {num_threads} CPU threads.")
            torch.set_num_threads(num_threads) # type: ignore
            torch.set_num_interop_threads(2)

    def prepare_data(self) -> None:
        input_file = f"{self.data_dir}/RDFs/{self.config['INPUTS']['dataset']}"
        target_file = f"{self.data_dir}/RDFs/{self.config['INPUTS']['target']}" if self.config['INPUTS'].get('target_path') else None
        input_data = load_tensor(input_file)
        target_data = load_tensor(target_file) if target_file else input_data.clone()

        dataset = Dataset(
            input_data, target_data,
            norm_inputs=True, norm_targets=True,
            norm_mode=self.config['TRAINING']['normalization']
        )

        # Save normalization params
        for key, value in self.norm_files.items():
            ds = getattr(dataset, key)
            np.save(f"{self.norms_dir}/{self.run_id}_{value}", ds.numpy())

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_valid, _, _ = train_test_split(
            dataset.inputs, dataset.targets,
            train_size=self.config['TRAINING']['train_size'],
            random_state=self.config['GENERAL']['seed'],
            shuffle=True,
        )
        # Loads the data using pre-computed normalization
        self.train_dataset = PreloadedDataset(X_train, X_train)
        self.valid_dataset = PreloadedDataset(X_valid, X_valid)

        # Save the validation dataset
        torch.save(self.valid_dataset.inputs.detach().cpu(), 
                   f"{self.outputs_dir}/{self.run_id}_input_validation_data.pt")
        # Samplers and DataLoaders (robust to DDP)
        use_ddp = self.config['GENERAL']['use_ddp']

        loader_params = {
            'batch_size': self.config['TRAINING']['batch_size'],
            'num_workers': self.config['TRAINING']['num_workers'],
            'pin_memory': self.config['TRAINING']['pin_memory'],
            'drop_last': use_ddp,
        }

        self.train_loader = get_data_loader(
            self.train_dataset,
            use_ddp=use_ddp,
            shuffle=True,
            **loader_params
        )
        self.valid_loader = get_data_loader(
            self.valid_dataset,
            use_ddp=use_ddp,
            shuffle=False,
            **loader_params
        )

    def prepare_model(self, evaluation_mode=False) -> None:
        # Model
        self.model = NeuralNetwork(self.config['MODEL']).to(self.device)
        
        # Use PyTorch optimizations for CPU
        if self.device.type == "cpu":
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model)
                if is_main_process():
                    print("Using IPEX optimizations for CPU model")
            except ImportError:
                if is_main_process():
                    print("IPEX not available, using standard CPU optimizations")
                # Apply other CPU optimizations
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
            
        # DDP/DataParallel
        use_ddp = self.config['GENERAL']['use_ddp']
        local_rank = int(os.environ.get('LOCAL_RANK', 0)) if use_ddp else 0
        if use_ddp and dist.is_initialized():
            print(f"Using DDP with local rank {local_rank}")
            print(f"Device: {self.device}")
            if self.device.type == "cuda":
                self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=False)
            else:
                self.model = DDP(self.model, find_unused_parameters=False)
        elif (torch.cuda.is_available() and torch.cuda.device_count() > 1 and str(self.device).startswith("cuda")):
            self.model = DataParallel(self.model)

        # Loss and optimizer
        self.loss_function = self.config['TRAINING']['loss_function']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['LEARNING']['learning_rate'])

        # Callbacks
        if not evaluation_mode:
            self.save_best_model = SaveBestModel(best_model_name=self.model_file)
            self.early_stopping = EarlyStopping(
                patience=self.config['TRAINING']['early_stopping']['patience'],
                min_delta=self.config['TRAINING']['early_stopping']['delta']
            )

    def train_loop(self, start_epoch: int = 0) -> None:
        gradient_clipping = self.config['TRAINING'].get('gradient_clipping', None)
        min_epochs = self.config['TRAINING']['min_epochs']
        max_epochs = self.config['TRAINING']['max_epochs']
        use_ddp = self.config['GENERAL']['use_ddp']
        lr_scheduler = setup_lr_scheduler_with_warmup(
            optimizer=self.optimizer,
            scheduled_lr=self.config['LEARNING']['scheduled_lr'],
            lr_warmup_epochs=self.config['LEARNING']['lr_warmup_epochs'],
            milestones=self.config['LEARNING']['milestones'],
            gamma=self.config['LEARNING']['gamma'],
        )
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda')) # type: ignore
        learn_rate = []
        log_interval = 10

        def log_epoch(epoch: int, total_loss: float, val_loss: float, current_lr: float, elapsed_time: float) -> None:
            if self.writer is not None:
                self.writer.add_scalar('LearningRate', current_lr, epoch)
            log_main(logging.INFO, f'- END EPOCH {epoch} -\nTrain loss = {total_loss:.4f}, Validation loss = {val_loss:.4f}')
            log_main(logging.INFO, f'Learning rate = {current_lr:.6f}, Elapsed time = {elapsed_time:.2f}s')

        def step_lr(lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]) -> float:
            if self.config['LEARNING']['scheduled_lr'] and lr_scheduler is not None:
                current_lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            return current_lr

        def check_save_best_model(epoch: int, val_loss: float, total_loss: float) -> None:
            if (self.best_val_loss is None) or (val_loss < self.best_val_loss):
                if is_main_process():
                    self.save_best_model(
                        model=self.model,
                        current_valid_loss=val_loss,
                        current_train_loss=total_loss,
                        epoch=epoch,
                        optimizer=self.optimizer
                    )
                    log_main(logging.INFO, f'Best model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}')
                self.best_val_loss = val_loss

        for epoch in range(start_epoch, max_epochs):
            t0 = time.time()
            total_loss = train_epoch_step(
                model=self.model,
                optimizer=self.optimizer,
                scaler=scaler,
                epoch=epoch,
                device=self.device,
                data_loader=self.train_loader,
                loss_function=self.loss_function,
                gradient_clipping=gradient_clipping,
                use_ddp=use_ddp
            )

            if (epoch + 1) % log_interval == 0:
                log_main(logging.INFO, f'END TRAIN LOOP EPOCH {epoch}, Loss: {total_loss:.4f}')

            val_loss = self.validator.validate_epoch(
                model=self.model,
                device=self.device,
                data_loader=self.valid_loader,
                loss_function=self.loss_function,
                emb_file=f"{self.outputs_dir}/{self.run_id}_embeddings.pt",
                out_file=f"{self.outputs_dir}/{self.run_id}_outputs_validation_data.pt"
            )

            log_and_save_metrics(
                epoch=epoch,
                total_loss=total_loss,
                val_loss=val_loss,
                writer=self.writer,
                stats_file=self.stats_file
            )

            if ((epoch + 1) % log_interval == 0):
                check_save_best_model(epoch, val_loss, total_loss)

            if epoch >= min_epochs and self.early_stopping(val_loss):
                log_main(logging.INFO, f'Early stopping at epoch {epoch + 1}')
                break

            current_lr = step_lr(lr_scheduler)
            learn_rate.append(current_lr)
            elapsed_time = time.time() - t0

            log_epoch(epoch, total_loss, val_loss, current_lr, elapsed_time)

        if self.writer is not None:
            self.writer.close()

    def train(self) -> None:
        # Optionally set up TensorBoard
        if self.config['GENERAL']['use_tensorboard']:
            self.writer, tb_log_dir = setup_tensorboard_writer(self.logs_dir, self.run_id) 
            log_main(logging.INFO, f'TensorBoard logging enabled at {tb_log_dir}')

        # Save stats/model arch
        with open(self.stats_file, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
        with open(self.arch_file, "w") as f:
            f.write(str(self.model))

        # Run training
        self.train_loop(start_epoch=0)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train an autoencoder model using PyTorch.')
    parser.add_argument('--config', '-c', type=str, 
                       default=f"{root_dir}/config_autoencoder_cpu.json",
                       help='Path to the JSON configuration file')
    args = parser.parse_args()

    # Call Trainer
    trainer = Trainer(args.config)
    trainer.run_training()

    if is_main_process():
        print("Training completed.")