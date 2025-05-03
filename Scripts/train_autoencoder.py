import torch
from torch.utils.data import DataLoader
from PyISV.neural_network import NeuralNetwork
from PyISV.train_utils import Dataset, MSELoss, SaveBestModel, EarlyStopping, PreloadedDataset
import numpy as np
import yaml
import logging
import torch.utils.bottleneck

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Setup logging
log_file = config["output"]["log_file"]
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# General settings
device = torch.device(config["device"])
torch.manual_seed(config["seed"])

# Model settings
model_config = config["model"]
activation_fn = getattr(torch.nn, model_config["activation_fn"], torch.nn.ReLU)

# Training settings
training_config = config["training"]
batch_size = training_config["batch_size"]
train_fraction = training_config["train_fraction"]
min_epochs = training_config["min_epochs"]
max_epochs = training_config["max_epochs"]
lrate = training_config["learning_rate"]
early_stopping_config = training_config["early_stopping"]

# Data settings
data_config = config["input"]
input_data_path = data_config["autoencoder"]["path"]
target_data_path = data_config["autoencoder"]["target_path"]

# Load input data
input_data = torch.load(input_data_path).float()

if target_data_path:
    # If target data is provided, load it
    target_data = torch.load(target_data_path).float()
else:       
    # If no target data is provided, use input data as target
    target_data = input_data.clone()  

# Initialize dataset
dataset = Dataset(
    input_data,
    target_data,  # Autoencoder uses input as target
    norm_inputs=True,
    norm_targets=True,
    norm_mode=data_config["autoencoder"].get("normalization", "minmax")
)

# Save normalization parameters
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

# Add num_workers to DataLoader
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
# Initialize model
model = NeuralNetwork(
    model_type="autoencoder",
    input_shape=(1, input_data.shape[-1]),
    embed_dim=model_config["embed_dim"],
    encoder_channels=model_config["encoder_channels"],
    activation_fn=activation_fn,
    use_pooling=True
)
model.to(device)

# Define loss function
loss_function = MSELoss()

# Initialize utilities
save_best_model = SaveBestModel(best_model_name=config["output"]["model_name"])
early_stopping = EarlyStopping(patience=early_stopping_config["patience"], min_delta=early_stopping_config["delta"])

def train_autoencoder(model, train_loader, valid_loader, max_epochs, min_epochs, lrate, device, loss_function, early_stopping, save_best_model):
    """Encapsulates the training logic for the autoencoder."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    # Add a debug log to confirm max_epochs is respected
    logging.debug(f"Starting training with max_epochs={max_epochs}")

    # Ensure the loop respects max_epochs
    for epoch in range(max_epochs):
        logging.debug(f"Starting epoch {epoch + 1}/{max_epochs}")
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(x)
            loss = loss_function(reconstructed, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logging.info(f"Epoch {epoch + 1}/{max_epochs}, Loss: {total_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in valid_loader:
                x = x.to(device)
                reconstructed, _ = model(x)
                loss = loss_function(reconstructed, x)
                val_loss += loss.item()

        logging.info(f"Validation Loss: {val_loss:.4f}")

        # Early stopping
        if epoch >= min_epochs:
            if early_stopping(val_loss):
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Save best model
        save_best_model(
            current_valid_loss=val_loss,
            current_train_loss=total_loss,
            epoch=epoch,
            model=model,
            optimizer=optimizer
        )

    logging.info("Training completed.")

# Call the encapsulated function if running as a script
if __name__ == "__main__":
    train_autoencoder(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        lrate=lrate,
        device=device,
        loss_function=loss_function,
        early_stopping=early_stopping,
        save_best_model=save_best_model
    )

    # Apply JIT tracing after training
    example_input = torch.randn(1, 1, input_data.shape[-1]).to(device)  # Adjust shape as needed
    traced_model = model.apply_jit(example_input)
    traced_model_path = config["output"].get("traced_model_name", "traced_model.pt")
    traced_model.save(traced_model_path)
    logging.info(f"Traced model saved to {traced_model_path}")