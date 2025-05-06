import torch
from torch.utils.data import DataLoader
from PyISV.neural_network import NeuralNetwork
from PyISV.train_utils import Dataset, MSELoss, SaveBestModel, EarlyStopping, PreloadedDataset
import numpy as np
import yaml
import logging
import torch.utils.bottleneck
import torch.profiler
from torch.cuda.amp import autocast, GradScaler
import os
import matplotlib.pyplot as plt
import time  # Import time module

# Load the autoencoder-specific configuration file
with open("config_autoencoder.yaml", "r") as file:
    config = yaml.safe_load(file)

# Setup logging
log_file = config["output"]["log_file"]
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# -- General settings -- # 
device = torch.device(config["device"])
torch.manual_seed(config["seed"])

# -- Model settings -- #
model_config = config["model"]
model_type = config["model"]["type"]
activation_fn = getattr(torch.nn, model_config["activation_fn"], torch.nn.ReLU) # Default to ReLU if not defined
kernel_size = model_config.get("kernel_size", 5)  # Default to 5 if not specified
embed_dim = model_config["embed_dim"]
encoder_channels = model_config["encoder_channels"]
decoder_channels = model_config["decoder_channels"]

# -- Training settings -- #
training_config = config["training"]
batch_size = training_config["batch_size"]
train_fraction = training_config["train_fraction"]
min_epochs = training_config["min_epochs"]
max_epochs = training_config["max_epochs"]
lrate = training_config["learning_rate"]
early_stopping_config = training_config["early_stopping"]

# -- Data settings -- #
data_config = config["input"]
input_data_path = os.path.abspath(data_config["path"])
target_data_path = os.path.abspath(data_config["target_path"])

# Load input data
input_data = torch.load(input_data_path).float()

if target_data_path:
    # If target data is provided, load it
    target_data = torch.load(target_data_path).float()
else:
    # If no target data is provided (pure autoencoder), use input data as target
    target_data = input_data.clone()

# Save normalization parameters for input data
# Save normalization parameters for target data
dataset = Dataset(
    input_data,
    target_data,
    norm_inputs=True,
    norm_targets=True,
    norm_mode=data_config.get("normalization", "minmax")
)
np.save(config["output"]["normalization_params"]["target_scaler_subval"], dataset.subval_targets.numpy())
np.save(config["output"]["normalization_params"]["target_scaler_divval"], dataset.divval_targets.numpy())
np.save(config["output"]["normalization_params"]["input_scaler_subval"], dataset.subval_inputs.numpy())
np.save(config["output"]["normalization_params"]["input_scaler_divval"], dataset.divval_inputs.numpy())

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_valid, _, _ = train_test_split(
    dataset.inputs, dataset.targets, train_size=train_fraction, shuffle=True, random_state=config["seed"]
)

# Use PreloadedDataset instead of Dataset
train_dataset = PreloadedDataset(X_train, X_train, norm_inputs=False, norm_targets=False)
valid_dataset = PreloadedDataset(X_valid, X_valid, norm_inputs=False, norm_targets=False)

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=False)

valid_loader = DataLoader(valid_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=False)

# -- Model initialization -- #
model = NeuralNetwork(
    model_type=model_type,
    input_shape=(1, input_data.shape[-1]),
    embed_dim=embed_dim,
    encoder_channels=encoder_channels,
    decoder_channels=decoder_channels,
    activation_fn=activation_fn,
    kernel_size=kernel_size,  # Pass kernel size to the model
    use_pooling=True,
    device=device,
)
model.to(device)

# Define loss function
loss_function = MSELoss()

# Initialize utilities
save_best_model = SaveBestModel(best_model_name=config["output"]["model_name"])
early_stopping = EarlyStopping(patience=early_stopping_config["patience"], min_delta=early_stopping_config["delta"])

# Create a folder to save reconstructed outputs
output_folder = "./data/reconstructed_outputs"
os.makedirs(output_folder, exist_ok=True)

stats_file = config["output"]["stats_file"]
with open(stats_file, "w") as f:
    f.write("epoch,train_loss,val_loss\n")  # Write header

# Save the model architecture to a file
model_architecture_file = config["output"]["model_architecture_file"]
with open(model_architecture_file, "w") as f:
    f.write(str(model))  # Save the model architecture

def train_autoencoder(model, train_loader, valid_loader, training_params, utilities, device, loss_function):
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

    scaler = GradScaler()  # Initialize gradient scaler for mixed precision
    learn_rate = []  # Initialize list to track learning rates

    for epoch in range(max_epochs):
        t0 = time.time()  # Start timer for epoch
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)  # Zero gradients

            # Use autocast for mixed precision
            with autocast():
                reconstructed, _ = model(x)
                loss = loss_function(reconstructed, x)

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
                with autocast():
                    reconstructed, _ = model(x)
                    loss = loss_function(reconstructed, x)
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
    train_autoencoder(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        training_params={
            'max_epochs': max_epochs,
            'min_epochs': min_epochs,
            'lrate': lrate,
            'scheduled_lr': training_config.get("scheduled_lr", True)
        },
        utilities={
            'early_stopping': early_stopping,
            'save_best_model': save_best_model
        },
        device=device,
        loss_function=loss_function
    )

    # Apply JIT tracing after training
    example_input = torch.randn(1, 1, input_data.shape[-1]).to(device)  # Adjust shape as needed
    traced_model = model.apply_jit(example_input)
    traced_model_path = config["output"].get("traced_model_name", "./models/traced_autoencoder_model.pt")
    traced_model.save(traced_model_path)
    logging.info(f"Traced model saved to {traced_model_path}")