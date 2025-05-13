# Helper functions for training models in PyISV.

import os
import logging
import torch
import matplotlib.pyplot as plt

def log_gpu_memory_usage(device=None, prefix=""):
    """Logs current, reserved, and max allocated GPU memory for the given device (default: current device)."""
    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.cuda.current_device()
    else:
        device = device.index if hasattr(device, 'index') else device
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    msg = (
        f"{prefix}GPU Memory (device {device}): "
        f"allocated={allocated:.2f}MB, reserved={reserved:.2f}MB, max_allocated={max_allocated:.2f}MB"
    )
    logging.info(msg)
    
def clip_gradients(model):
    if getattr(model, 'grad_clip', None) is not None:
        if getattr(model, 'amp', False):
            model.scaler.unscale_(model.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip)

def train_classifier_batch(model, x, y):
    outputs = model.forward(x)
    loss = model.criterion(outputs, y)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y).sum().item()
    total = y.size(0)
    return loss, correct, total

def train_autoencoder_batch(model, x, y):
    reconstructed, _ = model.forward(x)
    loss = model.criterion(reconstructed, y if model.use_separate_target else x)
    return loss, None, None

def scheduler_step(model, val_loss):
    if model.scheduler is not None:
        if hasattr(model.scheduler, 'step') and 'ReduceLROnPlateau' in model.scheduler.__class__.__name__:
            if val_loss is not None:
                model.scheduler.step(val_loss)
        else:
            model.scheduler.step()

def log_epoch(model, epoch, total_loss, correct, total):
    if model.model_type == "classifier":
        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}/{model.max_epochs}, Training Loss: {total_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
    else:
        print(f"Epoch {epoch + 1}/{model.max_epochs}, Training Loss: {total_loss:.4f}")

def early_stop_and_checkpoint(model, val_loss, epoch):
    stop = False
    if model.early_stopping is not None and val_loss is not None:
        if model.early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            stop = True
    if model.checkpoint_path is not None and val_loss is not None:
        torch.save(model.state_dict(), model.checkpoint_path)
    return stop

def check_convolution_layers(encoder_channels, decoder_channels):
    if not encoder_channels or not decoder_channels:
        raise ValueError("encoder_channels and decoder_channels must be specified in the config file.")
    if len(encoder_channels) != len(decoder_channels):
        raise ValueError("encoder_channels and decoder_channels must have the same length.")
    if len(encoder_channels) < 2:
        raise ValueError("encoder_channels and decoder_channels must have at least 2 elements.")

def apply_jit(model, example_input):
    """Applies JIT tracing to the model and returns the traced model."""
    traced_model = torch.jit.trace(model, example_input)
    return traced_model

def is_main_process():
    try:
        import torch.distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True

def log_main(level, msg):
    if is_main_process():
        logging.log(level, msg)

def setup_tensorboard_writer(logs_dir, run_id):
    from torch.utils.tensorboard import SummaryWriter
    tb_log_dir = os.path.join(logs_dir, f"tensorboard_{run_id}")
    writer = SummaryWriter(log_dir=tb_log_dir)
    return writer, tb_log_dir

def run_lr_finder(model, train_loader, device, loss_function, logs_dir, run_id, min_lr=1e-6, max_lr=1, num_iters=100):
    """Simple learning rate finder: increases LR exponentially and records loss."""
    import copy
    
    model_copy = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=min_lr)
    lr_mult = (max_lr / min_lr) ** (1 / num_iters)
    lrs, losses, lr = [], [], min_lr

    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
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
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
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
    plot_path = os.path.join(logs_dir, f'lr_finder_{run_id}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Learning rate finder plot saved to {plot_path}")
    return plot_path

