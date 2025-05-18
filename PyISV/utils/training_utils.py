
# Training utilities for PyISV
# This module contains functions to train models, including autoencoders and classifiers.
# It also includes functions for early stopping, learning rate scheduling, and logging.

import re, os, logging, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader # type: ignore
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

class RegexFilter(logging.Filter):
    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = re.compile(pattern)
    def filter(self, record: logging.LogRecord) -> bool:
        return not self.pattern.search(record.getMessage())

# Loss functions
class RMSELoss(nn.Module):
    def __init__(self) -> None:
        super(RMSELoss,self).__init__()
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

class BCEWithLogitsLoss(nn.Module):
    def __init__(self) -> None:
        super(BCEWithLogitsLoss,self).__init__()
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(x, y)
        return loss

class BCELoss(nn.Module):
    def __init__(self) -> None:
        super(BCELoss,self).__init__()
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        criterion = nn.BCELoss()
        loss = criterion(x, y)
        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super(CrossEntropyLoss,self).__init__()
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(x, y)
        return loss

class MSELoss(nn.Module):
    def __init__(self) -> None:
        super(MSELoss,self).__init__()
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        criterion =  nn.MSELoss()
        loss = criterion(x, y)
        return loss

class HuberLoss(torch.nn.SmoothL1Loss):
    def __init__(self, delta: float = 1.0) -> None:
        # PyTorch's SmoothL1Loss is the Huber loss with beta=delta
        super().__init__(beta=delta, reduction='mean')
    def forward(self, x: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        # Ignore delta argument for compatibility, use self.beta
        return super().forward(x, y)
    
# -- Dataset class -- #
class Dataset(Dataset): # type: ignore
    def __init__(self,
                 inputs: torch.Tensor,
                 targets: torch.Tensor,
                 norm_inputs: bool = False,
                 norm_targets: bool = False,
                 norm_mode: str = "minmax",
                 norm_threshold_inputs: float = 1e-6,
                 norm_threshold_targets: float = 1e-6) -> None:

        self.num_inputs = inputs.shape[0]
        self.num_inputs_features = inputs.shape[-1]
        self.num_targets = targets.shape[0]
        self.num_targets_features = targets.shape[-1]

        self.inputs = inputs
        self.targets = targets
        
        self.norm_threshold_inputs = norm_threshold_inputs
        self.norm_threshold_targets = norm_threshold_targets

        self.norm_mode = norm_mode # 'minmax' or 'gaussian'
        if ((self.norm_mode != "gaussian") and (self.norm_mode != "minmax")):
            raise ValueError(f"Normalization mode {self.norm_mode} not supported. Use 'minmax' or 'gaussian'.")
        
        if (norm_inputs==True):  
            self.inputs = self.set_norm_inputs(inputs)
        if (norm_targets==True):
            self.targets = self.set_norm_targets(targets)
            
    def __len__(self) -> int:
        return self.num_inputs

    def set_norm_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if (self.norm_mode == "minmax"):
            self.maxval_inputs = torch.max(x,axis=0).values # type: ignore
            self.subval_inputs = torch.min(x,axis=0).values # type: ignore
            self.divval_inputs = (self.maxval_inputs - self.subval_inputs)
            self.divval_inputs[self.divval_inputs < self.norm_threshold_inputs] = 1
        if (self.norm_mode == "gaussian"):
            self.subval_inputs = torch.mean(x, axis=0) # type: ignore
            self.divval_inputs = torch.std(x,axis=0) # type: ignore
            self.divval_inputs[self.divval_inputs < self.norm_threshold_inputs] = 1            

        x = self.rescale(x, self.subval_inputs, self.divval_inputs)
        return x

    def set_norm_targets(self, x: torch.Tensor) -> torch.Tensor:
        if (self.norm_mode == "minmax"):
            self.maxval_targets = torch.max(x, axis=0).values # type: ignore
            self.subval_targets = torch.min(x, axis=0).values # type: ignore
            self.divval_targets = (self.maxval_targets - self.subval_targets)
            self.divval_targets[self.divval_targets < self.norm_threshold_targets] = 1
        if (self.norm_mode == "gaussian"):
            self.subval_targets = torch.mean(x, axis=0) # type: ignore
            self.divval_targets = torch.std(x, axis=0) # type: ignore
            self.divval_targets[self.divval_targets < self.norm_threshold_targets] = 1            

        x = self.rescale(x, self.subval_targets, self.divval_targets)
        return x    

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]

    def rescale(self,
                x: torch.Tensor, 
                minval: torch.Tensor,
                rangeval: torch.Tensor) -> torch.Tensor:
        dataset_size = x.shape[0]
        sample_size = x.shape[1:]
        subval = torch.Tensor(minval).unsqueeze(0).expand(dataset_size, *sample_size)
        divval = torch.Tensor(rangeval).unsqueeze(0).expand(dataset_size, *sample_size)
        return torch.Tensor(x).sub(subval).div(divval)  

# -- PreloadedDataset class -- #
class PreloadedDataset(Dataset):
    def __init__(self,
                 inputs: torch.Tensor,
                 targets: torch.Tensor,
                 norm_inputs: bool = False,
                 norm_targets: bool = False,
                 norm_mode: str = "minmax",
                 norm_threshold_inputs: float = 1e-6,
                 norm_threshold_targets: float = 1e-6) -> None:
        # Use the standard Dataset initialization
        super().__init__(inputs, targets, norm_inputs, norm_targets, norm_mode, norm_threshold_inputs, norm_threshold_targets)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]: 
        # Directly use the parent class's __getitem__ method
        return super().__getitem__(idx)

# -- Callbacks -- #
class SaveBestModel():
    # see: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def __init__(self,
                 best_valid_loss: float = float('inf'),
                 best_train_loss: float = float('inf'),
                 best_model_name: str = "best_model"):
        self.best_valid_loss = best_valid_loss
        self.best_train_loss = best_train_loss
        self.best_model_name = best_model_name
        self.best_epoch = None

    def __call__(self, current_valid_loss: float, current_train_loss: float, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_train_loss = current_train_loss
            self.best_epoch = epoch
            print(f"  Saving best model at epoch: {epoch + 1}, Best validation loss: {self.best_valid_loss}")
            torch.save({
                'epoch': self.best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': self.best_valid_loss,
                'best_train_loss': self.best_train_loss
                }, self.best_model_name)

# -- Early Stopping -- #
class EarlyStopping:
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001) -> None:
        """
        Args:
            patience (int): How many epochs to wait before stopping when loss stops improving.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.best_loss: float = float('inf')
        self.counter: int = 0

    def __call__(self, loss: float) -> bool:
        improved = loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = loss
            self.counter = 0  # Reset counter if improvement
        else:
            self.counter += 1  # Increment counter if no improvement
        # Log after updating, and show the actual comparison
        logging.info(
            f"EarlyStopping: val_loss={loss:.8f}, best_loss={self.best_loss:.8f}, min_delta={self.min_delta:.8f}, "
            f"compare={self.best_loss + self.min_delta:.8f}, improved={improved}, counter={self.counter}"
        )
        return self.counter >= self.patience

def log_gpu_memory_usage(device: torch.device | None = None, prefix: str = "") -> None:
    """Logs current, reserved, and max allocated GPU memory for the given device (default: current device)."""
    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.cuda.current_device() # type: ignore
    else:
        device = device.index if hasattr(device, 'index') else device # type: ignore
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    msg = (
        f"{prefix}GPU Memory (device {device}): "
        f"allocated={allocated:.2f}MB, reserved={reserved:.2f}MB, max_allocated={max_allocated:.2f}MB"
    )
    logging.info(msg)

def clip_gradients(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if getattr(model, 'grad_clip', None) is not None:
        if getattr(model, 'amp', False):
            model.scaler.unscale_(optimizer) # type: ignore
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip) # type: ignore


def scheduler_step(model: torch.nn.Module, val_loss: float) -> None:
    if model.scheduler is not None:
        model.scheduler.step() # type: ignore

def log_epoch(
       model: torch.nn.Module,
       epoch: int,
       total_loss: float,
       correct: int,
       total: int) -> None:
    if model.model_type == "classifier":
        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}/{model.max_epochs}, Training Loss: {total_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
    else:
        print(f"Epoch {epoch + 1}/{model.max_epochs}, Training Loss: {total_loss:.4f}")

def early_stop_and_checkpoint(
        model: torch.nn.Module,
        val_loss: float,
        epoch: int) -> bool:
    stop = False
    if model.early_stopping is not None and val_loss is not None:
        if model.early_stopping(val_loss): # type: ignore
            print(f"Early stopping at epoch {epoch + 1}")
            stop = True
    if model.checkpoint_path is not None and val_loss is not None:
        torch.save(model.state_dict(), model.checkpoint_path) # type: ignore
    return stop

def apply_jit(
        model: torch.nn.Module,
        example_input: torch.Tensor) -> torch.nn.Module:
    """Applies JIT tracing to the model and returns the traced model."""
    traced_model = torch.jit.trace(model, example_input)
    return traced_model  # type: ignore

def is_main_process() -> bool:
    # For torch.distributed, rank 0 is the main process
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # For SLURM or other multi-process setups, check environment variables
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"]) == 0
    # Fallback: single process
    return True
    
def setup_logging(log_file: Optional[str] = None, log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
    """Set up logging configuration. If log_file is provided, logs will be written to that file.
    Removes all existing handlers to ensure this config takes effect."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename="train.log" if log_file is None else log_file,
        level=log_level,
        format=log_format
    )

def log_main(level: int, msg: str) -> None:
    if is_main_process():
        logging.log(level, msg)

def load_tensor(file_path: str) -> torch.Tensor:
    if file_path.endswith('.npy'):
        arr = np.load(file_path)
        tensor = torch.from_numpy(arr).float()
    else:
        tensor = torch.load(file_path).float()
    # Only unsqueeze if 2D
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    return tensor

def setup_tensorboard_writer(logs_dir: Path | str, run_id: str | int) -> tuple[SummaryWriter, str | Path]:
    from torch.utils.tensorboard import SummaryWriter
    tb_log_dir = f"{logs_dir}/tensorboard_{run_id}"
    writer = SummaryWriter(log_dir=tb_log_dir)
    return writer, tb_log_dir

def run_lr_finder(
       model: torch.nn.Module,
       train_loader: torch.utils.data.DataLoader,
       device: torch.device,
       loss_function: torch.nn.Module,
       out_file: str,
       min_lr: float = 1e-6,
       max_lr: float = 1,
       num_iters: int = 100) -> None:
    """Simple learning rate finder: increases LR exponentially and records loss."""
    import copy
    
    model_copy = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=min_lr)
    lr_mult = (max_lr / min_lr) ** (1 / num_iters)
    lrs, losses, lr = [], [], min_lr

    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda')) # type: ignore
    model_copy.train()
    iter_loader = iter(train_loader)
    for i in range(num_iters):
        try:
            x, _ = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            x, _ = next(iter_loader)
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast( # type: ignore
            device_type='cuda' if device.type == 'cuda' else 'cpu',
            dtype=torch.bfloat16 if device.type == 'cuda' else torch.float16):
            recon, embed = model_copy(x)
            loss = loss_function(recon, x)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lrs.append(lr)
        losses.append(loss.item())
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # Plot
    plt.figure()
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plot_path = out_file
    plt.savefig(plot_path)
    plt.close()
    print(f"Learning rate finder plot saved to {plot_path}")

def setup_lr_scheduler_with_warmup(
       optimizer: torch.optim.Optimizer,
       scheduled_lr: bool,
       lr_warmup_epochs: int,
       milestones: list[int],
       gamma: float) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Sets up a learning rate scheduler with optional warmup."""
    if not scheduled_lr:
        logging.info("Learning rate scheduler disabled")
        return None
        
    milestones = milestones or [100, 250]
    main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )
    
    if lr_warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=lr_warmup_epochs
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[lr_warmup_epochs]
        )
        logging.info(f"Learning rate scheduler enabled: LinearLR warmup for {lr_warmup_epochs} epochs, then MultiStepLR.")
    else:
        lr_scheduler = main_scheduler
        logging.info(f"Learning rate scheduler enabled: MultiStepLR with milestones {milestones} and gamma {gamma}")
        
    return lr_scheduler # type: ignore

def get_data_loader(dataset, batch_size, use_ddp, shuffle=True, num_workers=0, pin_memory=False, drop_last=False):
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle_flag = False
    else:
        sampler = None
        shuffle_flag = shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

def train_epoch_step(
        model: torch.nn.Module, 
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler, # type: ignore
        loss_function: torch.nn.Module,
        device: torch.device,
        use_ddp: bool = False,
        gradient_clipping: float | None = None,
        epoch: int =0) -> float:
    """Train model for a single epoch."""
    if use_ddp and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)   # type: ignore  
    model.train()
    total_loss = 0
    num_batches = 0

    import time
    data_time = 0.0
    compute_time = 0.0
    
    # Pre-fetch batches to reduce GIL contention
    batches = []
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        logging.info(f"Pre-fetching data for epoch {epoch}...")
    t_prefetch_start = time.time()
    for batch in data_loader:
        batches.append(batch)
    prefetch_time = time.time() - t_prefetch_start
    
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        logging.info(f"Prefetch completed in {prefetch_time:.2f}s for {len(batches)} batches")
    
    # Process batches
    for batch_idx, (x, _) in enumerate(batches):
        t0 = time.time()
        x = x.to(device, non_blocking=True)
        t1 = time.time()
        
        # Zero gradients more efficiently
        for param in model.parameters():
            param.grad = None
        
        # Use autocast for mixed precision
        with torch.amp.autocast( # type: ignore
            device_type='cuda' if device.type == 'cuda' else 'cpu',
            dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32): 
            recon, embed = model(x)
            loss = loss_function(recon, x)

        # Backward pass
        scaler.scale(loss).backward()

        if gradient_clipping is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        scaler.step(optimizer)
        scaler.update()

        t2 = time.time()
        data_time += (t1 - t0)
        compute_time += (t2 - t1)

        total_loss += loss.item()
        num_batches += 1
        
        # Periodically force synchronization between processes
        if use_ddp and batch_idx % 10 == 0:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Only rank 0 prints timing info DEBUG
    #if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
    #    print(f"[Epoch {epoch}] Data loading time: {data_time:.2f}s, Compute time: {compute_time:.2f}s")
    
    # Free memory
    del batches
    
    if num_batches > 0:
        return total_loss / num_batches
    else:
        return 0.0

def move_to_device(
        model: torch.nn.Module,
        device: torch.device) -> torch.nn.Module:
    """Move the model to the specified device."""
    model.to(device)
    return model

def get_device(device: str) -> torch.device:
    """Get the device for the model."""
    if device not in ("cuda","cpu"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

def log_and_save_metrics(
        stats_file: str,
        epoch: int,
        total_loss: float,
        val_loss: float,
        writer: torch.utils.tensorboard.SummaryWriter | None = None) -> None: # type: ignore
    """Log metrics to console and save them to stats file."""
    # TensorBoard logging
    if writer is not None:
        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

    # Console logging
    log_main(logging.INFO, f"END VALIDATION EPOCH {epoch}, Validation Loss: {val_loss:.4f}")

    # Save metrics to stats file
    if is_main_process():
        with open(stats_file, "a") as f:
            f.write(f"{epoch + 1},{total_loss:.4f},{val_loss:.4f}\n")
            log_main(logging.INFO, f"Epoch {epoch + 1} stats saved to {stats_file}")

