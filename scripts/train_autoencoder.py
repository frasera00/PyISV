# This script trains an autoencoder model using PyTorch and the PyISV library.

# ---------------- Load modules -------------------- #

# Import model class and training utilities
from PyISV.neural_network import NeuralNetwork
from PyISV.training_utilities import (
    Dataset, MSELoss, SaveBestModel, 
    EarlyStopping, PreloadedDataset, log_gpu_memory_usage
)
# Import helper functions 
from PyISV.training_helpers import (
    check_convolution_layers, apply_jit, is_main_process, log_main,
    run_lr_finder, setup_tensorboard_writer
)
# Import general libraries
import datetime
import os
import time 
import yaml
import logging
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Import PyTorch libraries
import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# Profiling utilities for performance analysis
import torch.utils.bottleneck
import torch.profiler

# ------------ Set paths to directories ------------- #

# Get the absolute path to the PyISV root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYISV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Build paths relative to the PyISV root:
models_dir = os.path.join(PYISV_ROOT, 'models')
data_dir = os.path.join(PYISV_ROOT, 'data')
outputs_dir = os.path.join(PYISV_ROOT, 'outputs')
norms_dir = os.path.join(PYISV_ROOT, 'norm_vals')
stats_dir = os.path.join(PYISV_ROOT, 'stats')
logs_dir = os.path.join(PYISV_ROOT, 'logs')

# ---------- Load config and setup logging ---------- #

with open(f"{PYISV_ROOT}/config_autoencoder.yaml", "r") as file:
    config = yaml.safe_load(file)

# Logging setup
log_file = os.path.join(logs_dir, config['output']['log_file'])
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    
# Generate a unique run ID for this experiment
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_main(logging.INFO, f"Run ID: {RUN_ID}")

# General settings with fallback default values
GENERAL_CONFIG = config["general"]
device = GENERAL_CONFIG.get("device", "auto")
seed = GENERAL_CONFIG.get("seed", 42)
apply_jit_tracing = GENERAL_CONFIG.get("apply_jit_tracing", False)

# Model settings with fallback default values
MODEL_CONFIG = config["model"]
model_type = MODEL_CONFIG.get("type") # "autoencoder" or "classifier"
kernel_size = MODEL_CONFIG.get("kernel_size", 5)
embed_dim = MODEL_CONFIG.get("embed_dim", 2)
use_pooling = MODEL_CONFIG.get("use_pooling", False)
encoder_channels = MODEL_CONFIG.get("encoder_channels", None)
decoder_channels = MODEL_CONFIG.get("decoder_channels", None)
activation_fn = getattr(torch.nn, MODEL_CONFIG["activation_fn"], torch.nn.ReLU)
check_convolution_layers(encoder_channels, decoder_channels) # Check if channels are valid

# Training settings with fallback default values
TRAINING_CONFIG = config["training"]
batch_size = TRAINING_CONFIG.get("batch_size", 32)
train_fraction = TRAINING_CONFIG.get("train_fraction", 0.8)
min_epochs = TRAINING_CONFIG.get("min_epochs", 10)
max_epochs = TRAINING_CONFIG.get("max_epochs", 100)
lrate = TRAINING_CONFIG.get("learning_rate", 0.001)
early_stopping_config = TRAINING_CONFIG.get("early_stopping", False)
num_workers = TRAINING_CONFIG.get("num_workers", 1)
pin_memory = TRAINING_CONFIG.get("pin_memory", False)
use_ddp = TRAINING_CONFIG.get("use_ddp", False)
gradient_clipping = TRAINING_CONFIG.get("gradient_clipping", None)
lr_warmup_epochs = TRAINING_CONFIG.get("lr_warmup_epochs", 0)

# Optional: Learning rate finder and TensorBoard
use_lr_finder = TRAINING_CONFIG.get("use_lr_finder", False)
use_tensorboard = TRAINING_CONFIG.get("use_tensorboard", False)

# TensorBoard setup using helper
if use_tensorboard:
    writer, tb_log_dir = setup_tensorboard_writer(logs_dir, RUN_ID)
    log_main(logging.INFO, f"TensorBoard logging enabled at {tb_log_dir}")
else:
    writer = None

# Data, input and output settings with fallback default values
DATA_CONFIG = config["input"]
OUTPUT_CONFIG = config["output"]
INPUT_FILE = os.path.join(f"{data_dir}/RDFs", DATA_CONFIG.get("dataset"))
TARGET_FILE = os.path.join(f"{data_dir}/RDFs", DATA_CONFIG.get("target")) if DATA_CONFIG.get("target_path") else None
INPUT_DATA = torch.load(INPUT_FILE).float()

if TARGET_FILE:
    TARGET_DATA = torch.load(TARGET_FILE).float()  # If provided, load it
else:
    TARGET_DATA = INPUT_DATA.clone()  # If not provided (pure autoencoder), use input data as target

# ----------- Device setup, DDP optional ------------ #

# Check if we are using DDP (Distributed Data Parallel)
local_rank = 0
if use_ddp and __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log_main(logging.INFO, f"Running on device: {device}")

# Setting seed for reproducibility
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    log_main(logging.INFO, "Using deterministic mode for CUDA")
torch.manual_seed(seed)

# --------------- Reconstruct dataset -------------- #

# Create the dataset object
dataset = Dataset(
    INPUT_DATA,
    TARGET_DATA,
    norm_inputs=True,
    norm_targets=True,
    norm_mode=DATA_CONFIG.get("normalization", "minmax")
)

# Save normalization parameters for input and target data
NORM_FILES = OUTPUT_CONFIG.get("normalization_params")
for key, value in NORM_FILES.items():
    ds = getattr(dataset, key)
    np.save(os.path.join(norms_dir, f"{RUN_ID}_{value}"), ds.numpy())

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_valid, _, _ = train_test_split(
    dataset.inputs, 
    dataset.targets,
    train_size=train_fraction,
    shuffle=True,
    random_state=seed
)

# use PreloadedDataset to efficiently serve already-normalized, pre-split data to model,
# avoiding repeated normalization or transformation
train_dataset = PreloadedDataset(X_train, X_train, norm_inputs=False, norm_targets=False)
valid_dataset = PreloadedDataset(X_valid, X_valid, norm_inputs=False, norm_targets=False)

# Only use DistributedSampler if DDP is initialized
if use_ddp and dist.is_initialized():
    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
else:
    train_sampler = None
    valid_sampler = None

# Create DataLoader objects for training and validation
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=use_ddp  # Ensure all ranks have the same number of batches in DDP
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=use_ddp  # For strict DDP validation, though not always required
)

# ---------------- Model initialization ---------------- #

# Check if input data is 3D (batch, channels, features)
input_shape = tuple(MODEL_CONFIG.get("input_shape", [1, INPUT_DATA.shape[-1]]))
if input_shape[-1] != INPUT_DATA.shape[-1]:
    logging.warning(f"input_shape[-1] ({input_shape[-1]}) does not match input data shape ({INPUT_DATA.shape[-1]}). Overriding input_shape to match data.")
    input_shape = (input_shape[0], INPUT_DATA.shape[-1])

# Initialize the model
model = NeuralNetwork({
    "type": model_type,
    "input_shape": input_shape,
    "embed_dim": embed_dim,
    "encoder_channels": encoder_channels,
    "decoder_channels": decoder_channels,
    "activation_fn": activation_fn,
    "kernel_size": kernel_size,
    "use_pooling": use_pooling,
    "device": device,               # should be a torch.device or string
})
model.to(device)

# DDP setup: enable find_unused_parameters to avoid hangs when some submodules are unused
if use_ddp and dist.is_initialized():
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
    logging.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = DataParallel(model)

# Define loss function
loss_function = MSELoss()

# Callbacks and utilities
save_best_model = SaveBestModel(best_model_name=f"{models_dir}/{RUN_ID}_best_autoencoder_model.pt",)
early_stopping = EarlyStopping(patience=early_stopping_config["patience"], min_delta=early_stopping_config["delta"])

# Statistics file
stats_file = f"{stats_dir}/{RUN_ID}_train_stats.txt"
with open(stats_file, "w") as f:
    f.write("epoch,train_loss,val_loss\n")  # Write header

# Save model architecture to file
model_architecture_file = OUTPUT_CONFIG["model_architecture_file"]
with open(os.path.join(models_dir, f"{RUN_ID}_model_architecture_file"), "w") as f:
    f.write(str(model))  # Save the model architecture

# ----------------- Training function -------------- #

def train_autoencoder(model, train_loader, valid_loader, training_params, utilities, device, loss_function, start_epoch=0):
    """Encapsulates the training logic for the autoencoder."""
    max_epochs = training_params['max_epochs']
    min_epochs = training_params['min_epochs']
    lrate = training_params['lrate']
    scheduled_lr = training_params['scheduled_lr']
    early_stopping = utilities['early_stopping']
    save_best_model = utilities['save_best_model']

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    # --- Learning Rate Scheduler with Warmup --- #
    lr_scheduler = None
    if scheduled_lr:
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250], gamma=0.5)
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
            logging.info("Learning rate scheduler enabled: MultiStepLR with milestones [100, 250] and gamma 0.5")
    else:
        logging.info("Learning rate scheduler disabled")

    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))  # Only enable if CUDA
    learn_rate = []  # Initialize list to track learning rates
    log_interval = 10  # Log every 10 batches
    for epoch in range(start_epoch, max_epochs):
        if use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()  # Start timer for epoch
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)  # Zero gradients

            # Use autocast for mixed precision
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                recon, embed = model(x)
                loss = loss_function(recon, x)

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()

            # --- Gradient Clipping --- #
            if gradient_clipping is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # TensorBoard: log training loss
        if writer is not None:
            writer.add_scalar('Loss/train', total_loss, epoch)

        if (epoch + 1) % log_interval == 0:
            log_main(logging.INFO, f"END TRAIN LOOP EPOCH {epoch}, Loss: {total_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in valid_loader:
                x = x.to(device)
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    recon, embed = model(x)
                    loss = loss_function(recon, x)
                val_loss += loss.item()

        # TensorBoard: log validation loss
        if writer is not None:
            writer.add_scalar('Loss/val', val_loss, epoch)

        log_main(logging.INFO, f"END VALIDATION EPOCH {epoch}, Validation Loss: {val_loss:.4f}")


        # Save metrics to stats file
        if is_main_process():
            with open(stats_file, "a") as f:
                f.write(f"{epoch + 1},{total_loss:.4f},{val_loss:.4f}\n")
                log_main(logging.INFO, f"Epoch {epoch + 1} stats saved to {stats_file}")

        # Log GPU memory usage (main process only, if CUDA)
        #if is_main_process() and torch.cuda.is_available():
        #    log_gpu_memory_usage(device, prefix=f"END EPOCH {epoch}: ")

        # Save best model only if validation loss improves
        if (not hasattr(train_autoencoder, "best_val_loss")) or (val_loss < train_autoencoder.best_val_loss):
            if is_main_process():
                save_best_model(
                    current_valid_loss=val_loss,
                    current_train_loss=total_loss,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer
                )
                log_main(logging.INFO, f"Best model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
            train_autoencoder.best_val_loss = val_loss

        # Early stopping
        if epoch >= min_epochs:
            if early_stopping(val_loss):
                log_main(logging.INFO, f"Early stopping at epoch {epoch + 1}")
                break

        # Update learning rate and track it
        if scheduled_lr:
            current_lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()
        else:
            current_lr = optimizer.param_groups[0]['lr']

        learn_rate.append(current_lr)
        elapsed_time = time.time() - t0

        # TensorBoard: log learning rate
        if writer is not None:
            writer.add_scalar('LearningRate', current_lr, epoch)

        log_main(logging.INFO, f"END EPOCH {epoch}: Learning rate = {current_lr:.6f}, Elapsed time = {elapsed_time:.2f}s")

    # TensorBoard: close writer
    if writer is not None:
        writer.close()
        


# ---------------- Main execution ------------------ #

if __name__ == "__main__":
    try:
        # Optionally run learning rate finder before training
        if use_lr_finder:
            run_lr_finder(model, train_loader, device, loss_function, logs_dir, RUN_ID)

        start_epoch = 0

        # Check if we are continuing from a checkpoint
        continue_from_checkpoint = False
        if continue_from_checkpoint:
            checkpoint_path = f"{models_dir}/checkpoint.pt"  # or your specific checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = torch.optim.Adam(model.parameters())  # Use same optimizer type as before
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

        # Train the autoencoder
        train_autoencoder(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            training_params={
                'max_epochs': max_epochs,
                'min_epochs': min_epochs,
                'lrate': lrate,
                'scheduled_lr': TRAINING_CONFIG.get("scheduled_lr", True),
                'start_epoch': start_epoch
            },
            utilities={
                'early_stopping': early_stopping,
                'save_best_model': save_best_model
            },
            device=device,
            loss_function=loss_function
        )

        # Apply JIT tracing after training (only on rank 0 for DDP)
        if apply_jit_tracing:
            dummy_input = torch.randn(1, *input_shape).to(device)

            # JIT trace using the underlying module if wrapped in DataParallel/DDP
            base_model = model.module if (isinstance(model, DataParallel) or isinstance(model, DDP)) else model
            traced_model = apply_jit(base_model, dummy_input)

            # Save traced model
            traced_model_path = os.path.join(models_dir, f"traced_autoencoder_model_{RUN_ID}.pt")
            traced_model.save(traced_model_path)
            log_main(logging.INFO, f"Traced model saved to {traced_model_path}")

    # Clean up DDP
    finally:
        if use_ddp and dist.is_initialized():
            dist.destroy_process_group()

    # --------------------------------------------------- #
