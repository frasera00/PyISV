
# Training utilities for PyISV
# This module contains functions to train models, including autoencoders and classifiers.
# It also includes functions for early stopping, learning rate scheduling, and logging.

import matplotlib.pyplot as plt
import logging
from typing import Optional

import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
    
class VAELoss(nn.Module):
    def __init__(self, reconstruction_weight=1.0, kl_weight=0.01, beta=1.0):
        super(VAELoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.beta = beta
    
    def forward(self, recon_x, x, mu, logvar):
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size
        
        # Total loss
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.kl_weight * self.beta * kl_loss)
        
        return total_loss

class Dataset(TorchDataset): 
    def __init__(self,
                 inputs: torch.Tensor,
                 targets: torch.Tensor,
                 norm_inputs: bool = False,
                 norm_targets: bool = False,
                 norm_mode: str = "minmax",
                 norm_threshold_inputs: float = 1e-6,
                 norm_threshold_targets: float = 1e-6,
                 device: Optional[torch.device] = None) -> None:

        self.num_inputs = inputs.shape[0]
        self.num_inputs_features = inputs.shape[-1]
        self.num_targets = targets.shape[0]
        self.num_targets_features = targets.shape[-1]

        self.inputs = inputs
        self.targets = targets
        self.device = device
        
        self.norm_threshold_inputs = norm_threshold_inputs
        self.norm_threshold_targets = norm_threshold_targets

        self.norm_mode = norm_mode # 'minmax' or 'gaussian'
        if ((self.norm_mode != "gaussian") and (self.norm_mode != "minmax")):
            raise ValueError(f"Normalization mode {self.norm_mode} not supported. Use 'minmax' or 'gaussian'.")
        
        if (norm_inputs==True):  
            self.inputs = self.set_norm_inputs(inputs)
        if (norm_targets==True):
            self.targets = self.set_norm_targets(targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]
    
    def __len__(self) -> int:
        return self.num_inputs
            
    def to_device(self, device: torch.device) -> None:
        """Move dataset tensors to specified device"""
        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)
        self.device = device
        return 

    def set_norm_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if (self.norm_mode == "minmax"):
            self.maxval_inputs = torch.max(x, dim=0).values # type: ignore
            self.subval_inputs = torch.min(x, dim=0).values # type: ignore
            self.divval_inputs = (self.maxval_inputs - self.subval_inputs)
            self.divval_inputs[self.divval_inputs < self.norm_threshold_inputs] = 1
        if (self.norm_mode == "gaussian"):
            self.subval_inputs = torch.mean(x, dim=0) # type: ignore
            self.divval_inputs = torch.std(x, dim=0) # type: ignore
            self.divval_inputs[self.divval_inputs < self.norm_threshold_inputs] = 1            

        x = self.rescale(x, self.subval_inputs, self.divval_inputs)    
        return x 
    
    def set_norm_targets(self, y: torch.Tensor) -> torch.Tensor:
        if (self.norm_mode == "minmax"):
            self.maxval_targets = torch.max(y, dim=0).values
            self.subval_targets = torch.min(y, dim=0).values
            self.divval_targets = (self.maxval_targets - self.subval_targets)
            self.divval_targets[self.divval_targets < self.norm_threshold_targets] = 1
        if (self.norm_mode == "gaussian"):
            self.subval_targets = torch.mean(y, dim=0)
            self.divval_targets = torch.std(y, dim=0)
            self.divval_targets[self.divval_targets < self.norm_threshold_targets] = 1
        
        y = self.rescale(y, self.subval_targets, self.divval_targets)
        return y
    
    def rescale(self, x: torch.Tensor, minval: torch.Tensor, rangeval: torch.Tensor) -> torch.Tensor:
        dataset_size = x.shape[0]
        sample_size = x.shape[1:]
        subval = torch.Tensor(minval).unsqueeze(0).expand(dataset_size, *sample_size)
        divval = torch.Tensor(rangeval).unsqueeze(0).expand(dataset_size, *sample_size)
        return torch.Tensor(x).sub(subval).div(divval)  

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

class SaveBestModel():
    # see: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def __init__(self, best_valid_loss: float = float('inf'), best_train_loss: float = float('inf'), 
                 best_model_name: str = "best_model") -> None:
        self.best_valid_loss = best_valid_loss
        self.best_train_loss = best_train_loss
        self.best_model_name = best_model_name
        self.best_epoch = None

    def __call__(self, current_valid_loss: float, current_train_loss: float, epoch: int, 
                 model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_train_loss = current_train_loss
            self.best_epoch = epoch
            print(f" --- 💾 Saving best model at epoch: {epoch + 1} --- ")
            torch.save({
                'epoch': self.best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': self.best_valid_loss,
                'best_train_loss': self.best_train_loss
                }, self.best_model_name)

class EarlyStopping:
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001) -> None:
        """ Class for early stopping based on validation loss."""
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

def clip_gradients(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if getattr(model, 'grad_clip', None) is not None:
        if getattr(model, 'amp', False):
            model.scaler.unscale_(optimizer) # type: ignore
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip) # type: ignore

def scheduler_step(model: torch.nn.Module, val_loss: float) -> None:
    if model.scheduler is not None:
        model.scheduler.step() # type: ignore

def early_stop_and_checkpoint(model: torch.nn.Module, val_loss: float) -> bool:
    stop = False
    if model.early_stopping is not None and val_loss is not None:
        if model.early_stopping(val_loss): # type: ignore
            stop = True
    if model.checkpoint_path is not None and val_loss is not None:
        torch.save(model.state_dict(), model.checkpoint_path) # type: ignore
    return stop

def apply_jit(model: torch.nn.Module, example_input: torch.Tensor) -> torch.nn.Module:
    """Applies JIT tracing to the model and returns the traced model."""
    traced_model = torch.jit.trace(model, example_input)
    return traced_model  # type: ignore
    
def run_lr_finder(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
       device: torch.device, loss_function: torch.nn.Module, out_file: str,
       min_lr: float = 1e-6, max_lr: float = 1, num_iters: int = 100) -> None:
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
            inp, targ = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            inp, targ = next(iter_loader)
        inp = inp.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast( # type: ignore
            device_type='cuda' if device.type == 'cuda' else 'cpu',
            dtype=torch.bfloat16 if device.type == 'cuda' else torch.float16):
            loss = loss_function(model(inp), targ)
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

def setup_lr_scheduler_with_warmup(optimizer: torch.optim.Optimizer,
       params: dict) -> torch.optim.lr_scheduler.LRScheduler:
    """Sets up a learning rate scheduler with optional warmup."""
    
    try:
        lr_warmup_epochs = params['lr_warmup_epochs']
        milestones = params['milestones']
        gamma = params['gamma']
    except KeyError as e:
        raise ValueError(f"Missing required parameter: {e}. Please provide 'lr_warmup_epochs', 'milestones', and 'gamma'.")

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
        
    return lr_scheduler

def get_data_loader(dataset: Dataset, batch_size: int,
                    use_ddp: bool = False, shuffle: bool = True,
                    num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            sampler=sampler,
            shuffle=False,
            drop_last=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            shuffle=shuffle,
            drop_last=False
        )

def train_epoch_step(
        model: torch.nn.Module, 
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler, # type: ignore
        loss_function: torch.nn.Module,
        device: torch.device,
        use_ddp: bool = False,
        epoch: int = 0) -> float:
    
    if use_ddp and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch) # type: ignore

    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Process batches directly from DataLoader (no prefetching)
    for batch_idx, (inp, targ) in enumerate(data_loader):
        inp = inp.to(device)
        targ = targ.to(device)

        optimizer.zero_grad()
        # Use autocast for mixed precision
        #with torch.amp.autocast( device_type='cuda' if device.type == 'cuda' else 'cpu')): 
        loss = loss_function(model(inp), targ)

        # Scale the loss for mixed precision training
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track full-magnitude loss for reporting
        total_loss += loss.item()
        num_batches += 1
        
        # Synchronize less frequently in distributed mode
        if use_ddp and batch_idx % 50 == 0:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    return total_loss / num_batches if total_loss > 0 else 0.0
 
def get_device(device: str) -> torch.device:
    """Get the device for the model."""
    if device not in ("cuda","cpu"):
        device = "cpu" # Default to CPU if not specified
    return torch.device(device)

def apply_cuda_optimizers(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere+ GPUs
        torch.backends.cudnn.allow_tf32 = True