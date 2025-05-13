# This script evaluates the autoencoder model by plotting the training loss curve,
# reconstructing the input data, and visualizing the latent space using t-SNE.

# ----------------- Load modules -------------------- #

# Import necessary libraries
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

# Import functions from external libraries
from openTSNE import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm 

# Import custom modules
from PyISV.training_utilities import PreloadedDataset, MSELoss
from PyISV.neural_network import NeuralNetwork 

# ------------ Set paths to directories ------------- #

# Set the run ID for the current evaluation.
# Should match the run ID used during training
RUN_ID = "20250512_195715"

# Get the absolute path to the PyISV root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYISV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Build paths relative to the PyISV root:
models_dir = os.path.join(PYISV_ROOT, 'models')
data_dir = os.path.join(PYISV_ROOT, 'data')
outputs_dir = os.path.join(PYISV_ROOT, 'outputs')
norms_dir = os.path.join(PYISV_ROOT, 'norm_vals')
stats_dir = os.path.join(PYISV_ROOT, 'stats')

# --------------- Define functions ---------------- #

def plot_loss_curve(stats_file):
    """Plot the training and validation loss curves."""

    df = pd.read_csv(stats_file)
    fig, ax = plt.subplots(1,1)
    ax.plot(df['epoch'], df['train_loss'], label='train')
    ax.plot(df['epoch'], df['val_loss'],   label='val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); plt.tight_layout()
    fig.savefig(os.path.join(f"{outputs_dir}/evaluation", 'loss_curve.png'))
    return

def export_onnx(model, input_shape):
    """Export the model to ONNX format."""

    # Ensure input_shape is a tuple of ints (not numpy floats)
    if isinstance(input_shape, np.ndarray):
        input_shape = tuple(int(x) for x in input_shape)
    else:
        input_shape = tuple(int(x) for x in input_shape)

    # Place dummy_input on the same device as the model
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape, device=device)
    onnx_path = os.path.join(models_dir, f'best_autoencoder_model.onnx')

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
    print(f'Exported ONNX model to: {onnx_path}')
    return

def evaluate_reconstructions(model, input_shape, loader, device, loss_fn):
    """Evaluate the model's reconstructions and plot the latent space."""
    
    model.eval()
    export_onnx(model, input_shape)  # Export the model to ONNX format

    all_errors, embeddings, outputs = [], [], []

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device)
            recon, latent = model(x)
            errs = loss_fn(recon, x).detach().cpu().numpy()
            all_errors.append(np.atleast_1d(errs))

            # Flatten latent except batch dimension
            embeddings.append(latent.detach().cpu().numpy().reshape(latent.shape[0], -1))
            outputs.append(recon.detach().cpu().numpy())

    errors = np.concatenate(all_errors)
    embeds = np.concatenate(embeddings, axis=0)
    outputs = np.concatenate(outputs, axis=0)

    # Histogram of errors
    fig, axes = plt.subplots(1,2)
    axes[0].hist(errors, bins=50, alpha=0.7)
    axes[0].set_xlabel('Reconstruction RMSE'); axes[0].set_ylabel('Count')

    # Run t-SNE (openTSNE library)
    tsne = TSNE(n_jobs=4, random_state=0)  # n_jobs sets the number of CPU cores
    z2d = tsne.fit(embeds)
    axes[1].scatter(z2d[:,0], z2d[:,1], s=5, alpha=0.6)
    axes[1].set_title('Latent space t-SNE'); plt.tight_layout()

    fig.savefig(os.path.join(f"{outputs_dir}/evaluation", 'latent_tsne.png'))

    # Unormalize the outputs
    outsubval = np.load(f"{norms_dir}/{RUN_ID}_output_autoen_scaler_subval.npy")
    outdivval = np.load(f"{norms_dir}/{RUN_ID}_output_autoen_scaler_divval.npy")
    outputs_denorm = (outputs * outdivval) + outsubval

    # Save the errors, embeddings, and reconstructed outputs (unnormalized)
    np.save(os.path.join(f"{outputs_dir}/evaluation", f'{RUN_ID}_tsne.npy'), z2d)
    np.save(os.path.join(f"{outputs_dir}/evaluation", f'{RUN_ID}_reconstructed_outputs.npy'), outputs_denorm)
    np.save(os.path.join(f"{outputs_dir}/evaluation", f'{RUN_ID}_recon_errors.npy'), errors)
    np.save(os.path.join(f"{outputs_dir}/evaluation", f'{RUN_ID}_latent_embeddings.npy'), embeds)
    return

def load_model(architecture_file, device):
    """Load the model architecture and state dictionary."""

    # -- Load the architecture of the model -- #
    with open(architecture_file) as f:
        lines = f.read()

    # -- Extract model parameters from the architecture -- #
    encoder_channels = [int(m.group(2)) for m in re.finditer(r'Conv1d\((\d+), (\d+),', lines)][:-1]
    decoder_channels = [int(m.group(2)) for m in re.finditer(r'ConvTranspose1d\((\d+), (\d+),', lines)]
    embed_dim = int(re.search(r'Linear\(in_features=\d+, out_features=(\d+),', lines).group(1))
    activation_fn = torch.nn.ReLU if 'ReLU' in lines else None
    use_pooling = ('MaxPool1d' in lines) or ('Upsample' in lines)
    kernel_size = [int(m.group(1)) for m in re.finditer(r'kernel_size=\((\d+),\)', lines)][0]
    input_shape = (int(re.search(r'Conv1d\((\d+),', lines).group(1)),
                int(re.search(r'Upsample\(size=(\d+),', lines).group(1)))
    
    # -- Create the model architecture -- #
    params = {
        "type": "autoencoder",
        "input_shape": input_shape,
        "embed_dim": embed_dim,
        "encoder_channels": encoder_channels,
        "decoder_channels": decoder_channels,
        "activation_fn": activation_fn,
        "kernel_size": kernel_size,
        "use_pooling": use_pooling,
        "device": device,
    }

    # -- Create the model -- #
    model = NeuralNetwork(params)

    # -- Load the saved state dictionary -- #
    model_path = f"{models_dir}/{RUN_ID}_best_autoencoder_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # -- Load the state dictionary into the model -- #
    # If the  model was trained and saved using torch.nn.DataParallel (DP) or 
    # torch.nn.parallel.DistributedDataParallel (DDP), the model is wrapped and 
    # a module. prefix is set to all parameter keys in the state dict. 
    # When the state dict is loaded into a model not wrapped with DDP/DataParallel, 
    # the keys do not match. So we need to remove the module. prefix from the keys.

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.to(device)

    return model, params

def rebuild_dataset(inputs, targets, batch_size):
    """Create a DataLoader for the dataset."""
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        inputs, targets,
        train_size=0.8,
        random_state=42
    )

    # Save the validation inputs for correct comparison with reconstructions (if pure autoencoder, X_val==y_val)
    np.save(os.path.join(outputs_dir, "evaluation", f"{RUN_ID}_val_inputs.npy"), X_val.cpu().numpy())
    np.save(os.path.join(outputs_dir, "evaluation", f"{RUN_ID}_val_targets.npy"), y_val.cpu().numpy())
    
    val_dataset = PreloadedDataset(
        X_val,
        y_val,
        norm_inputs=True,
        norm_targets=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    return val_loader, X_val, y_val

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """Main function to evaluate the autoencoder model."""

    # Load the model
    device = get_device()
    architecture_file = f"{models_dir}/autoencoder_architecture.txt"
    model, _ = load_model(architecture_file, device)

    # Load the input data for evaluation
    path = f"{data_dir}/RDFs/rdf_images.pt"
    target_path = None
    inputs = torch.load(path).float()
    targets = torch.load(target_path).float() if target_path else inputs.clone()
 
    # Save the unnormalized inputs and targets for later comparison
    np.save(os.path.join(outputs_dir, "evaluation", f"{RUN_ID}_all_inputs.npy"), inputs.cpu().numpy())
    np.save(os.path.join(outputs_dir, "evaluation", f"{RUN_ID}_all_targets.npy"), targets.cpu().numpy())

    # Ensure input_shape is a tuple of ints (not floats or numpy types)
    batch_size = int(inputs.shape[0])
    input_shape = tuple(int(x) for x in inputs.shape[1:])

    # Dataset and DataLoader for evaluation
    val_loader, X_val, y_val = rebuild_dataset(inputs, targets, batch_size=batch_size)

    # Loss function
    loss_fn = MSELoss()

    # run the evaluation
    evaluate_reconstructions(model, input_shape, val_loader, device, loss_fn)
    plot_loss_curve(f"{stats_dir}/{RUN_ID}_train_stats.txt")

# -------------- Execute the main function --------------- #

if __name__ == "__main__":
    main()