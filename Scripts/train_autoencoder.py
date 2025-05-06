import torch
from torch.utils.data import DataLoader
from PyISV.neural_network import NeuralNetwork
from PyISV.train_utils import Dataset, MSELoss, SaveBestModel, EarlyStopping, PreloadedDataset
import numpy as np
import yaml
import logging
import torch.utils.bottleneck
import torch.profiler
import os
import argparse

# --- Model and Utility Initialization Encapsulation --- #
def initialize_autoencoder_components(config, input_data):
    model_config = config["model"]
    model_config["input_shape"] = (1, input_data.shape[-1])
    model = NeuralNetwork(model_config)
    model.to(model.device)

    # Loss function
    loss_function = MSELoss()

    # Utilities
    save_best_model = SaveBestModel(best_model_name=config["output"]["model_name"])
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping"]["patience"],
        min_delta=config["training"]["early_stopping"]["delta"]
    )

    # Stats/log file setup
    stats_file = config["output"]["stats_file"]
    with open(stats_file, "w") as f:
        f.write("epoch,train_loss,val_loss\n")
    model_architecture_file = config["output"]["model_architecture_file"]
    with open(model_architecture_file, "w") as f:
        f.write(str(model))

    # Output folder
    output_folder = "./data/reconstructed_outputs"
    os.makedirs(output_folder, exist_ok=True)

    return model, loss_function, early_stopping, save_best_model

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
activation_fn =  model_config["activation_fn"]
kernel_size = model_config.get("kernel_size", 5)  # Default to 5 if not specified
embed_dim = model_config["embed_dim"]
encoder_channels = model_config["encoder_channels"]
decoder_channels = model_config["decoder_channels"]


# --- Data Preparation Encapsulation --- #
from sklearn.model_selection import train_test_split
def prepare_autoencoder_data(config):
    data_config = config["input"]
    input_data_path = os.path.abspath(data_config["path"])
    target_data_path = os.path.abspath(data_config["target_path"]) if data_config["target_path"] else None

    # Load input data
    input_data = torch.load(input_data_path).float()

    if target_data_path:
        target_data = torch.load(target_data_path).float()
    else:
        target_data = input_data.clone()

    # Save normalization parameters for input and target data
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
    X_train, X_valid, _, _ = train_test_split(
        dataset.inputs, dataset.targets, train_size=config["training"]["train_fraction"], shuffle=True, random_state=config["seed"]
    )

    # Use PreloadedDataset instead of Dataset
    train_dataset = PreloadedDataset(X_train, X_train, norm_inputs=False, norm_targets=False)
    valid_dataset = PreloadedDataset(X_valid, X_valid, norm_inputs=False, norm_targets=False)

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset,
                              batch_size=config["training"]["batch_size"],
                              shuffle=True,
                              num_workers=0,
                              pin_memory=False)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=config["training"]["batch_size"],
                              shuffle=False,
                              num_workers=0,
                              pin_memory=False)

    return train_loader, valid_loader, input_data



# --- Data Preparation --- #
train_loader, valid_loader, input_data = prepare_autoencoder_data(config)

# --- Model initialization and utilities --- #
model_config["input_shape"] = (1, input_data.shape[-1])
model = NeuralNetwork(model_config)
model.to(model.device)

loss_function = MSELoss() # Define loss function
save_best_model = SaveBestModel(best_model_name=config["output"]["model_name"]) # Save model utility
early_stopping = EarlyStopping(patience=config["training"]["early_stopping"]["patience"],
                            min_delta=config["training"]["early_stopping"]["delta"]) # Early stopping utility

stats_file = config["output"]["stats_file"]
with open(stats_file, "w") as f:
    f.write("epoch,train_loss,val_loss\n")  # Write header
model_architecture_file = config["output"]["model_architecture_file"]
with open(model_architecture_file, "w") as f:
    f.write(str(model))  # Save the model architecture

# Create a folder to save reconstructed outputs
output_folder = "./data/reconstructed_outputs"
os.makedirs(output_folder, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Train a PyISV autoencoder with flexible config and CLI overrides.")
    parser.add_argument('--config', type=str, default="config_autoencoder.yaml", help="Path to YAML config file.")
    parser.add_argument('--device', type=str, help="Override device (e.g. 'cuda' or 'cpu').")
    parser.add_argument('--batch_size', type=int, help="Override batch size.")
    parser.add_argument('--epochs', type=int, help="Override number of epochs.")
    parser.add_argument('--lr', type=float, help="Override learning rate.")
    parser.add_argument('--log_interval', type=int, help="Override log interval.")
    parser.add_argument('--checkpoint_path', type=str, help="Override checkpoint path.")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # CLI overrides
    if args.device:
        config["device"] = args.device
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs
        config["training"]["max_epochs"] = args.epochs  # for compatibility
    if args.lr:
        config["training"]["lr"] = args.lr
        config["training"]["learning_rate"] = args.lr  # for compatibility
    if args.log_interval:
        config["training"]["log_interval"] = args.log_interval
    if args.checkpoint_path:
        config["output"]["model_name"] = args.checkpoint_path

    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    # --- Data Preparation --- #

    train_loader, valid_loader, input_data = prepare_autoencoder_data(config)

    # --- Model and Utility Initialization --- #
    model, loss_function, early_stopping, save_best_model = initialize_autoencoder_components(config, input_data)

    # Merge YAML config with script-specific/dynamic options (do not mutate original)
    base_cfg = config["training"]
    # Pass scheduler config instead of scheduler object; let train_model create it after optimizer is initialized
    scheduler_cfg = None
    if base_cfg.get("scheduled_lr", True):
        scheduler_cfg = {
            "type": "MultiStepLR",
            "milestones": [100, 250],
            "gamma": 0.5
        }

    train_cfg = {
        **base_cfg,
        "device": device,
        "criterion": loss_function,
        "early_stopping": early_stopping,
        "checkpoint_path": config["output"]["model_name"],
        "scheduler_cfg": scheduler_cfg,
        "best_model_callback": save_best_model
    }
    model.train_model(
        train_loader,
        valid_loader,
        config=train_cfg
    )

    # Apply JIT tracing after training
    example_input = torch.randn(1, 1, input_data.shape[-1]).to(device)  # Adjust shape as needed
    traced_model = model.apply_jit(example_input)
    traced_model_path = config["output"].get("traced_model_name", "./models/traced_autoencoder_model.pt")
    traced_model.save(traced_model_path)
    logging.info(f"Traced model saved to {traced_model_path}")

if __name__ == "__main__":
    main()