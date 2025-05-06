import torch
from torch.utils.data import DataLoader
from PyISV.neural_network import NeuralNetwork
from PyISV.train_utils import Dataset, CrossEntropyLoss
import yaml
import logging
import os

# Load the classification-specific configuration file
with open("config_classification.yaml", "r") as file:
    config = yaml.safe_load(file)

# Setup logging
log_file = config["output"]["log_file"]
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# General settings
device = torch.device(config["device"])
torch.manual_seed(config["seed"])

# Model settings
model_config = config["model"]
model_type = model_config["type"]
activation_fn = getattr(torch.nn, model_config["activation_fn"], torch.nn.ReLU)
embed_dim = model_config.get("embed_dim", 16)  # Default to 16 if not present
num_classes = model_config["num_classes"]
encoder_channels = model_config["encoder_channels"]
kernel_size = model_config.get("kernel_size", 5)  # Default to 5 if not specified

# Ensure encoder_channels is a list
if isinstance(encoder_channels, int):
    encoder_channels = [encoder_channels]

# Training settings
training_config = config["training"]
batch_size = training_config["batch_size"]
train_fraction = training_config["train_fraction"]
max_epochs = training_config["max_epochs"]
lrate = training_config["learning_rate"]

# Data settings
data_config = config["input"]

# Resolve absolute paths for data files
input_data_path = os.path.abspath(data_config["path"])
labels_data_path = os.path.abspath(data_config["labels_path"])

# Load input data
input_data = torch.load(input_data_path).float()
labels_data = torch.load(labels_data_path).long()

# Initialize dataset
dataset = Dataset(
    input_data,
    labels_data,
    norm_inputs=True,
    norm_targets=False  # Labels should not be normalized
)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    dataset.inputs, dataset.targets, train_size=train_fraction, shuffle=True, random_state=config["seed"]
)

train_dataset = Dataset(X_train, y_train, norm_inputs=False, norm_targets=False)
valid_dataset = Dataset(X_valid, y_valid, norm_inputs=False, norm_targets=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = NeuralNetwork(
    model_type=model_type,
    input_shape=(1, input_data.shape[-1]),
    embed_dim=embed_dim,
    num_classes=num_classes,
    encoder_channels=encoder_channels,
    activation_fn=activation_fn,
    use_pooling=True,
    kernel_size=kernel_size,
)

model_architecture_file = config["output"]["model_architecture_file"]
with open(model_architecture_file, "w") as f:
    f.write(str(model))  # Save the model architecture

# Train the model
model.train_model(
    train_loader=train_loader,
    val_loader=valid_loader,
    epochs=max_epochs,
    lr=lrate,
    device=device,
    criterion=CrossEntropyLoss()
)

# Save the trained model
torch.save(model.state_dict(), config["output"]["model_name"])
logging.info(f"Model saved to {config['output']['model_name']}")