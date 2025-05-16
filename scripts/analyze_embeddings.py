import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from PyISV import (
    MODELS_DIR as models_dir,
    OUTPUTS_DIR as outputs_dir
)

print(models_dir)

# Set the run ID to analyze
RUN_ID = "20250513_220708"  # Change this to your run ID

def analyze_embeddings():
    """Analyze the latent embeddings from a trained autoencoder."""
    
    # Define the path to the embeddings file
    embeds_path = f"{outputs_dir}/evaluation/{RUN_ID}_embeddings.npy"
    
    # Load the embeddings
    try:
        embeddings = np.load(embeds_path)
        print(f"Loaded embeddings with shape: {embeddings.shape}")
    except FileNotFoundError:
        print(f"Error: Embeddings file not found at {embeds_path}")
        return
    
    # Analyze the embeddings
    print("\nEmbeddings Statistics:")
    print(f"Mean: {embeddings.mean(axis=0)}")
    print(f"Std Dev: {embeddings.std(axis=0)}")
    print(f"Min: {embeddings.min(axis=0)}")
    print(f"Max: {embeddings.max(axis=0)}")
    
    # Check variance across dimensions
    var_per_dim = embeddings.var(axis=0)
    print(f"\nVariance per dimension: {var_per_dim}")
    
    # Identify zero or near-zero dimensions
    near_zero_mask = var_per_dim < 1e-6
    if np.any(near_zero_mask):
        print(f"\nWARNING: {np.sum(near_zero_mask)} dimensions have near-zero variance!")
        print(f"Near-zero dimensions: {np.where(near_zero_mask)[0].tolist()}")
    
    # Plot a histogram of the values for each dimension
    n_dims = embeddings.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3*n_dims))
    
    if n_dims == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.hist(embeddings[:, i], bins=50, alpha=0.7)
        ax.set_title(f"Dimension {i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    
    plt.tight_layout()
    
    # Save the plot
    fig.savefig(f"{outputs_dir}/evaluation/{RUN_ID}_embedding_histograms.png")
    print(f"\nSaved histogram plot to {outputs_dir}/evaluation/{RUN_ID}_embedding_histograms.png")

    # Plot a scatter plot if there are 2 dimensions
    if n_dims == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5, s=5)
        plt.title(f"Latent Space for {RUN_ID}")
        plt.xlabel("Dimension 0")
        plt.ylabel("Dimension 1")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{outputs_dir}/evaluation/{RUN_ID}_latent_scatter.png")
        print(f"Saved scatter plot to {outputs_dir}/evaluation/{RUN_ID}_latent_scatter.png")
    
    return embeddings

if __name__ == "__main__":
    embeddings = analyze_embeddings()
