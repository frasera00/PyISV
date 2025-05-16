# This script trains an autoencoder model using PyTorch and the PyISV library.

import os
import datetime
import time 
import logging
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional

from PyISV.training_utils import *
from PyISV.set_architecture import import_config
from PyISV.neural_network import NeuralNetwork 
from PyISV.validation_utils import Validator   

# Set paths to directories and ID 
from PyISV.define_paths import (
    MODELS_DIR as models_dir, DATA_DIR as data_dir, OUTPUTS_DIR as outputs_dir,
    NORMS_DIR as norms_dir, STATS_DIR as stats_dir, LOGS_DIR as logs_dir,
    PROJECT_ROOT as root_dir
)

# Profiling utilities for performance analysis
import torch.utils.bottleneck
import torch.profiler

class Trainer():
    def __init__(self, params: dict) -> None:
        self.config = params
        self.device = get_device(device=self.config['GENERAL']['device'])
        self.run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = None
        self.best_val_loss = None

        # File paths
        self.model_file = f"{models_dir}/{self.run_id}_model.pt"
        self.stats_file = f"{stats_dir}/{self.run_id}_stats.dat"
        self.arch_file = f"{models_dir}/{self.run_id}_arch.dat"
        
        self.norm_files = {
            "divval_inputs": f"input_autoen_scaler_subval.npy",
            "subval_inputs": f"input_autoen_scaler_divval.npy",
            "divval_targets": f"output_autoen_scaler_subval.npy",
            "subval_targets": f"output_autoen_scaler_divval.npy"
        }

        # Set validator
        self.validator = Validator(self.config)
        self._setup_ddp()

        if is_main_process():
            print(f"Trainer initialized with run ID: {self.run_id}")
            setup_logging(f"{logs_dir}/{self.run_id}_train.log")
        self._run()

    def _run(self) -> None:
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
            #print(f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
            #print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
            #print(f"World size: {os.environ.get('WORLD_SIZE')}")
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            torch.distributed.barrier()
            #print(f"Process {local_rank} using torch.cuda.current_device()={torch.cuda.current_device()}, self.device={self.device}")
            
    def prepare_data(self) -> None:
        input_file = f"{data_dir}/RDFs/{self.config['INPUTS']['dataset']}"
        target_file = f"{data_dir}/RDFs/{self.config['INPUTS']['target']}" if self.config['INPUTS'].get('target_path') else None
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
            np.save(f"{norms_dir}/{self.run_id}_{value}", ds.numpy())

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
                   f"{outputs_dir}/{self.run_id}_input_validation_data.pt")
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

    def prepare_model(self) -> None:
        # Model
        self.model = NeuralNetwork(self.config['MODEL']).to(self.device)

        # DDP/DataParallel
        use_ddp = self.config['GENERAL']['use_ddp']
        local_rank = int(os.environ.get('LOCAL_RANK', 0)) if use_ddp else 0
        if use_ddp and dist.is_initialized():
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=False)
        elif (torch.cuda.is_available() and torch.cuda.device_count() > 1 and str(self.device).startswith("cuda")):
            self.model = DataParallel(self.model)

        # Loss and optimizer
        self.loss_function = self.config['TRAINING']['loss_function']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['LEARNING']['learning_rate'])

        # Callbacks
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
                emb_file=f"{outputs_dir}/{self.run_id}_embeddings.pt",
                out_file=f"{outputs_dir}/{self.run_id}_outputs_validation_data.pt"
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
            self.writer, tb_log_dir = setup_tensorboard_writer(logs_dir, self.run_id) 
            log_main(logging.INFO, f'TensorBoard logging enabled at {tb_log_dir}')

        # Save stats/model arch
        with open(self.stats_file, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
        with open(self.arch_file, "w") as f:
            f.write(str(self.model))

        # Train
        self.train_loop(start_epoch=0)

if __name__ == '__main__':
    params = import_config(json_path=f"{root_dir}/config_autoencoder.json")
    trainer = Trainer(params)

    if is_main_process():
        print("Training completed.")