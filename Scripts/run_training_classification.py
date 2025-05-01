import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  
import time
from torch import nn

# Import the model architecture, training helper, and dataset builder
from PyISV.network_1D_classification import Classifier1D
from PyISV.network_2D_classification import OptimizedClassifier2D
from PyISV.network_2D_classification import Classifier2D
from PyISV.train_utils import Dataset, SaveBestModel, ClassificationTrainer


# —————————————————————————————————————————————
# 1) Configuration
# —————————————————————————————————————————————

def get_config():
    """
    Returns a dictionary of configuration parameters used for training.
    This is the central place to tweak paths, hyperparameters, or architecture settings.
    """

    return {
        # Paths to input data
        "output_model_path": f"/scratch/rasera/PyISV/models/classifier_2D_best.pt",  # Path to save the best model
        "dataset_path": "/scratch/rasera/PyISV/RDFs_2D/rdf_images_2D.pt",     # Path to the precomputed RDFs
        "labels_path": "/scratch/rasera/PyISV/RDFs_2D/labels_2D.pt",          # Path to the labels for the RDFs

        # Model parameters
        "num_classes": 7,             # Number of distinct label classes
        "num_final_channels": 8,     # Number of channels in final conv layer
        "embed_dim": 16,              # Size of embedding before classification

        # Optimization hyperparameters
        "lr": 1e-3,           # Learning rate
        "weight_decay": 1e-5,    # L2 regularization
        "batch_size": 128,
        "num_epochs": 50,
        "val_split": 0.2,     # Fraction of training data used for validation

        # Hardware setting
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        
        # Random seed for reproducibility
        "seed": 42
    }

def set_seed(seed):
    """
    Sets random seeds for NumPy and PyTorch to ensure reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensures determinism on GPU
    torch.backends.cudnn.benchmark = False     # Turn off autotuning for consistency

def log_training_info(cfg, flat_dim, model_path, log_file="train_log.txt"):
    """
    Logs all training configuration and settings to a file for reference.
    """
    with open(log_file, "w") as log:
        log.write("### LOGGING TRAINING INFO ###\n\n")

    with open(log_file, "a") as log:
        log.write(f'''[Model Parameters]
        Embedding dimension = {cfg["embed_dim"]}
        Flat dimension = {flat_dim}
        Seed = {cfg["seed"]}

        [Training Parameters]
        Device = {cfg["device"]}
        Number of epochs = {cfg["num_epochs"]}
        Saved model path = {model_path}
        Batch size = {cfg["batch_size"]}
        Learning rate = {cfg["lr"]}
        Weight decay = {cfg["weight_decay"]}
        Validation fraction = {cfg["val_split"]}

        [Input]
        Dataset path = {cfg["dataset_path"]}
        Labels path = {cfg["labels_path"]}
        Number of classes = {cfg["num_classes"]}
        ''')


# —————————————————————————————————————————————
# 2) Model Training Setup
# —————————————————————————————————————————————

def select_random_subset(images, labels, num_samples, seed=42):
    """
    Randomly selects a subset of samples from images and labels.
    """
    torch.manual_seed(seed)  # Ensure reproducibility
    perm = torch.randperm(images.size(0))[:num_samples]
    images_subset = images[perm]
    labels_subset = labels[perm]
    return images_subset, labels_subset

def load_dataset_for_training(dataset_path, labels_path, batch_size, val_split=0.2, device="cpu", fraction=1.0):
    """
    Loads the precomputed RDF images and labels for training.
    Handles both 1D ([N, 1, L]) and 2D ([N, 1, H, W]) input formats.
    Returns:
        - train_loader, val_loader: DataLoader instances
        - input_shape: tuple describing input shape (1, L) or (1, H, W)
    """
    # 1) Load tensors from disk
    images = torch.load(dataset_path)  # Expected shapes: [N,1,200] or [N,1,200,200]
    labels = torch.load(labels_path)   # shape: [N]

    # 2) Verify and get input shape
    if images.ndim == 3 and images.shape[1] == 1:
        # 1D data: [N, 1, L]
        input_shape = images.shape[1:]  # (1, L)
    elif images.ndim == 4 and images.shape[1] == 1:
        # 2D data: [N, 1, H, W]
        input_shape = images.shape[1:]  # (1, H, W)
    else:
        raise ValueError(f"Unexpected input shape {images.shape}. Expected:\n"
                       "- 1D: [N, 1, L]\n"
                       "- 2D: [N, 1, H, W]")

    # 3) Select subset if needed
    if fraction < 1.0:
        num_samples = int(len(images) * fraction)
        images, labels = select_random_subset(images, labels, num_samples)

    # 4) Create dataset
    dataset = Dataset(
        inputs=images,
        targets=labels,
        norm_inputs=False,
        norm_targets=False
    )

    # 5) Split into train/val
    N = len(dataset)
    n_val = int(val_split * N)
    train_ds, val_ds = random_split(dataset, [N - n_val, n_val])

    # 6) Create DataLoaders
    pin = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=pin
    )

    return train_loader, val_loader, input_shape

def build_model_1D(input_shape, num_final_channels, embed_dim, num_classes, device):
    """
    Initializes the 1D model with proper input shape handling.
    """
    model = Classifier1D(
        input_shape=input_shape,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_encoder_final_channels=num_final_channels
    ).to(device)

    return model

def build_model_2D(input_shape, num_final_channels, embed_dim, num_classes, device):
    """
    Initializes the 2D model with proper input shape handling.
    """
    model = OptimizedClassifier2D(
        input_shape=input_shape,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_encoder_final_channels=num_final_channels
    ).to(device)

    return model

# —————————————————————————————————————————————
# 3) Training Loop
# —————————————————————————————————————————————

def optimized_train_model(trainer, model, num_epochs, learning_rate, model_path, log_file="train_log.txt", save_best_model=True):
    scaler = torch.amp.GradScaler(device_type='cuda', enabled=trainer.device.type == 'cuda')
    saver = SaveBestModel(best_model_name=model_path) if save_best_model else None

    # Define warmup epochs (typically 5-10% of total epochs)
    warmup_epochs = min(5, num_epochs // 10)

    for epoch in tqdm(range(1, num_epochs + 1)):
        start_epoch = time.time()
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for inputs, targets in trainer.train_loader:
            inputs = inputs.to(trainer.device, non_blocking=True)
            targets = targets.to(trainer.device, non_blocking=True)

            with torch.amp.autocast(device_type=trainer.device.type, enabled=trainer.device.type == 'cuda'):
                outputs = model(inputs)
                loss = trainer.criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(trainer.optimizer)
            scaler.update()
            trainer.optimizer.zero_grad()

            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == targets).float().mean().item()
        
        # Validation phase
        val_loss, val_acc = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in trainer.val_loader:
                inputs = inputs.to(trainer.device, non_blocking=True)
                targets = targets.to(trainer.device, non_blocking=True)
                
                outputs = model(inputs)
                loss = trainer.criterion(outputs, targets)
                
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == targets).float().mean().item()
        
        # Calculate epoch metrics
        train_loss /= len(trainer.train_loader)
        train_acc /= len(trainer.train_loader)
        val_loss /= len(trainer.val_loader)
        val_acc /= len(trainer.val_loader)
        
        # Learning rate warmup
        if epoch <= warmup_epochs:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = learning_rate * (epoch / warmup_epochs)
        
        # Logging
        epoch_time = time.time() - start_epoch
        log_str = (f"Epoch {epoch:03d}/{num_epochs} | "
                  f"Train L={train_loss:.4f}, A={train_acc:.4f} | "
                  f"Val L={val_loss:.4f}, A={val_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        print(log_str)
        
        with open(log_file, "a") as f:
            f.write(log_str + "\n")
        
        # Save best model using SaveBestModel
        if save_best_model:
            saver(
                current_valid_loss=val_loss,
                current_train_loss=train_loss/len(trainer.train_loader),
                epoch=epoch,
                model=model,
                optimizer=trainer.optimizer
            )
    
    # Final cleanup
    if trainer.device.type == 'cuda':
        torch.cuda.empty_cache()

def train_model(trainer, model, num_epochs, model_path, log_file="train_log.txt", save_best_model=True):
    """
    Performs the training and validation loop for the classifier model.
    This uses the ClassificationTrainer class instead of custom training logic.
    """
    saver = SaveBestModel(best_model_name=model_path) if save_best_model else None

    for epoch in tqdm(range(1, num_epochs + 1)):
        # Train the model on the current batch of data
        start_epoch = time.time()
        train_loss, train_acc = trainer.train_epoch()
        end_epoch = time.time()
        print(f"Train epoch time: {end_epoch - start_epoch:.2f}s")

        # Validate the model on the validation data
        val_loss, val_acc = trainer.validate_epoch()      # Evaluate on validation data

        log_str = (f"Epoch {epoch:02d} | "
                   f"Train L={train_loss:.4f}, A={train_acc:.4f} | "
                   f"Val   L={val_loss:.4f}, A={val_acc:.4f}")
        print(log_str)

        # Log to file
        with open(log_file, "a") as log:
            log.write(log_str + "\n")

        if save_best_model:
            saver(current_valid_loss=val_loss,
                  current_train_loss=train_loss,
                  epoch=epoch,
                  model=model,
                  optimizer=trainer.optimizer)

# —————————————————————————————————————————————
# 4) Main Entry Point
# —————————————————————————————————————————————

def main():
    cfg = get_config()
    set_seed(cfg["seed"])

    # Load data and get input shape
    train_loader, val_loader, input_shape = load_dataset_for_training(
        cfg["dataset_path"],
        cfg["labels_path"],
        cfg["batch_size"],
        val_split=cfg["val_split"],
        device=cfg["device"],
        fraction=1.0
    )

    # Determine model type based on input shape
    if len(input_shape) == 2:  # (1, L) - 1D
        model = build_model_1D(
            input_shape=input_shape,
            num_final_channels=cfg["num_final_channels"],
            embed_dim=cfg["embed_dim"],
            num_classes=cfg["num_classes"],
            device=torch.device(cfg["device"])
        )
    else:  # (1, H, W) - 2D
        model = build_model_2D(
            input_shape=input_shape,
            num_final_channels=cfg["num_final_channels"],
            embed_dim=cfg["embed_dim"],
            num_classes=cfg["num_classes"],
            device=torch.device(cfg["device"])
        )

    # Create trainer utility with optimizer and loaders
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        device=torch.device(cfg["device"])
    )

    # Train the model
    optimized_train_model(trainer, model, num_epochs=cfg["num_epochs"],learning_rate=cfg["lr"], model_path=cfg["output_model_path"], log_file="train_log.txt")

    print(f"Training complete. Model saved at {cfg['output_model_path']}")


if __name__ == "__main__":
    main()
