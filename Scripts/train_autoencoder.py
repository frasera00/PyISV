# -*- coding: utf-8 -*-

# Autoencoder training script
from PyISV.neural_network import NeuralNetwork
from PyISV.train_utils import Dataset, MSELoss, SaveBestModel, EarlyStopping, PreloadedDataset

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

# -- Load autoencoder configuration file -- #
with open("config_autoencoder.yaml", "r") as file:
    config = yaml.safe_load(file)

# -- Logging setup -- #
log_file = config["output"]["log_file"]
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Generate a unique run ID for this experiment
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.info(f"Run ID: {RUN_ID}")

# -- Device setup, DDP optional -- #
use_ddp = int(config.get("use_ddp", 0))  # Default to 0
if use_ddp:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f"cuda:{local_rank}")
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running on device: {DEVICE}")

# -- Setting seed -- #
if DEVICE == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    logging.info("Using deterministic mode for CUDA")
else:
    seed = config("seed", 42)  # Default seed value
    torch.manual_seed(seed)

# -- Model settings -- #
MODEL_CONFIG = config["model"]
model_type = MODEL_CONFIG["type"]
activation_fn = getattr(torch.nn, MODEL_CONFIG["activation_fn"], torch.nn.ReLU) # Default to ReLU if not defined
kernel_size = MODEL_CONFIG.get("kernel_size", 5)               # Default to 5 if not specified
embed_dim = MODEL_CONFIG["embed_dim"]
encoder_channels = MODEL_CONFIG["encoder_channels"]
decoder_channels = MODEL_CONFIG["decoder_channels"]

# -- Training settings -- #
TRAINING_CONFIG = config["training"]
batch_size = TRAINING_CONFIG.get("batch_size", 32)
train_fraction = TRAINING_CONFIG.get("train_fraction", 0.8)  # Default to 80% if not specified
min_epochs = TRAINING_CONFIG.get("min_epochs", 10)           # Default to 10 if not specified
max_epochs = TRAINING_CONFIG.get("max_epochs", 100)          # Default to 100 if not specified
lrate = TRAINING_CONFIG.get("learning_rate", 0.001)          # Default to 0.001 if not specified
early_stopping_config = TRAINING_CONFIG.get("early_stopping", False)  # Default to False if not specified
num_workers = TRAINING_CONFIG.get("num_workers", 1)          # Default to 1 if not specified
pin_memory = TRAINING_CONFIG.get("pin_memory", False)        # Default to False if not specified

# -- Data, input and output settings -- #
DATA_CONFIG = config["input"]
OUTPUT_CONFIG = config["output"]
INPUT_PATH = os.path.abspath(DATA_CONFIG["path"])
OUTPUT_PATH = os.path.abspath(DATA_CONFIG["target_path"]) if DATA_CONFIG["target_path"] else None
INPUT_DATA = torch.load(INPUT_PATH).float()

if OUTPUT_PATH:
    TARGET_DATA = torch.load(OUTPUT_PATH).float()  # If provided, load it
else:
    TARGET_DATA = INPUT_DATA.clone()  # If not provided (pure autoencoder), use input data as target

# Create dataset
dataset = Dataset(
    INPUT_DATA,
    TARGET_DATA,
    norm_inputs=True,
    norm_targets=True,
    norm_mode=DATA_CONFIG.get("normalization", "minmax") # Default to minmax if not specified
)

# Save normalization parameters for input and target data
np.save(OUTPUT_CONFIG["normalization_params"]["target_scaler_subval"], dataset.subval_targets.numpy())
np.save(OUTPUT_CONFIG["normalization_params"]["target_scaler_divval"], dataset.divval_targets.numpy())
np.save(OUTPUT_CONFIG["normalization_params"]["input_scaler_subval"], dataset.subval_inputs.numpy())
np.save(OUTPUT_CONFIG["normalization_params"]["input_scaler_divval"], dataset.divval_inputs.numpy())

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_valid, _, _ = train_test_split(
    dataset.inputs, 
    dataset.targets,
    train_size=train_fraction,
    shuffle=True,
    random_state=seed
)

# -- use PreloadedDataset to efficiently serve already-normalized, pre-split data to model, 
#  avoiding repeated normalization or transformation. -- #
train_dataset = PreloadedDataset(X_train, X_train, norm_inputs=False, norm_targets=False)
valid_dataset = PreloadedDataset(X_valid, X_valid, norm_inputs=False, norm_targets=False)

train_sampler = DistributedSampler(train_dataset) if use_ddp else None
valid_sampler = DistributedSampler(valid_dataset, shuffle=False) if use_ddp else None

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    drop_last=use_ddp  # Ensure all ranks have the same number of batches in DDP
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=4,
    pin_memory=True,
    drop_last=use_ddp  # For strict DDP validation, though not always required
)

# -- Model initialization -- #
input_shape = tuple(MODEL_CONFIG.get("input_shape", [1, INPUT_DATA.shape[-1]]))
if input_shape[-1] != INPUT_DATA.shape[-1]:
    logging.warning(f"input_shape[-1] ({input_shape[-1]}) does not match input data shape ({INPUT_DATA.shape[-1]}). Overriding input_shape to match data.")
    input_shape = (input_shape[0], INPUT_DATA.shape[-1])

model = NeuralNetwork(
    model_type=model_type,
    input_shape=input_shape,
    embed_dim=embed_dim,
    encoder_channels=encoder_channels,
    decoder_channels=decoder_channels,
    activation_fn=activation_fn,
    kernel_size=kernel_size,  # Pass kernel size to the model
    use_pooling=True,
    device=DEVICE,
)
model.to(DEVICE)

# -- DDP or DataParallel setup -- #
if use_ddp:
    model = DDP(model, device_ids=[local_rank])
elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
    logging.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = DataParallel(model)

# -- Define loss function --#
LOSS_FUNCTION = MSELoss()

# -- Callbacks and utilities -- #
model_name_base, model_name_ext = os.path.splitext(OUTPUT_CONFIG["model_name"])
model_save_path = f"{model_name_base}_{RUN_ID}{model_name_ext}"
save_best_model = SaveBestModel(best_model_name=model_save_path)
early_stopping = EarlyStopping(patience=early_stopping_config["patience"], min_delta=early_stopping_config["delta"])

# -- Create output directory for reconstructed outputs -- #
output_folder = "./data/reconstructed_outputs"
os.makedirs(output_folder, exist_ok=True)

stats_file = OUTPUT_CONFIG["stats_file"]
with open(stats_file, "w") as f:
    f.write("epoch,train_loss,val_loss\n")  # Write header

model_architecture_file = OUTPUT_CONFIG["model_architecture_file"]
with open(model_architecture_file, "w") as f:
    f.write(str(model))  # Save the model architecture

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

    # Initialize learning rate scheduler
    if scheduled_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250], gamma=0.5)
        logging.info("Learning rate scheduler enabled: MultiStepLR with milestones [100, 250] and gamma 0.5")
    else:
        lr_scheduler = None
        logging.info("Learning rate scheduler disabled")

    scaler = torch.amp.GradScaler()  # Initialize gradient scaler for mixed precision
    learn_rate = []  # Initialize list to track learning rates

    for epoch in range(start_epoch,max_epochs):
        if use_ddp:
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
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        logging.info(f"Epoch {epoch + 1}/{max_epochs}, Loss: {total_loss:.4f}")

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

        logging.info(f"Validation Loss: {val_loss:.4f}")

        # Save metrics to stats file
        with open(stats_file, "a") as f:
            f.write(f"{epoch + 1},{total_loss:.4f},{val_loss:.4f}\n")

        # Save best model only if validation loss improves
        if (not hasattr(train_autoencoder, "best_val_loss")) or (val_loss < train_autoencoder.best_val_loss):
            save_best_model(
                current_valid_loss=val_loss,
                current_train_loss=total_loss,
                epoch=epoch,
                model=model,
                optimizer=optimizer
            )
            train_autoencoder.best_val_loss = val_loss

        # Early stopping
        if epoch >= min_epochs:
            if early_stopping(val_loss):
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Update learning rate and track it
        if scheduled_lr:
            current_lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()
        else:
            current_lr = optimizer.param_groups[0]['lr']

        learn_rate.append(current_lr)
        elapsed_time = time.time() - t0

        logging.info(f"Epoch {epoch + 1}: Learning rate = {current_lr:.6f}, Elapsed time = {elapsed_time:.2f}s")

    logging.info("Training completed.")
    logging.info(f"Best Validation Loss: {train_autoencoder.best_val_loss:.4f}")

# Run the training function
if __name__ == "__main__":

    start_epoch = 0
    continue_from_checkpoint = False
    if continue_from_checkpoint:
        # Path to your checkpoint file
        checkpoint_path = "models/best_autoencoder_model.pt"  # or your specific checkpoint

        # Load checkpoint (use map_location if on CPU)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        # Load model and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters())  # Use same optimizer type as before
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set the starting epoch
        start_epoch = checkpoint['epoch'] + 1

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
        device=DEVICE,
        loss_function=LOSS_FUNCTION
    )

    # -- Apply JIT tracing after training (only on rank 0 for DDP) -- #
    if not use_ddp or (use_ddp and (not dist.is_initialized() or dist.get_rank() == 0)):
        dummy_input = torch.randn(1, *input_shape).to(DEVICE)

        # JIT trace using the underlying module if wrapped in DataParallel/DDP
        base_model = model.module if (isinstance(model, DataParallel) or isinstance(model, DDP)) else model
        traced_model = base_model.apply_jit(dummy_input)

        # Save traced model
        traced_model_path = os.path.join("./models", f"traced_autoencoder_model_{RUN_ID}.pt")
        traced_model.save(traced_model_path)
        logging.info(f"Traced model saved to {traced_model_path}")

    # Clean up DDP
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()
