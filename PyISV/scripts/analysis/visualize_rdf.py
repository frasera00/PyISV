#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to visualize RDF (Radial Distribution Function) outputs from PyISV tests.
This script loads and plots the various RDF formats stored in the test_outputs directory.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_and_inspect_rdf_file(file_path):
    """
    Load a PyTorch RDF file and print its basic structure and statistics.
    
    Parameters:
    -----------
    file_path : str
        Path to the RDF PyTorch file
    
    Returns:
    --------
    torch.Tensor
        The loaded RDF tensor
    """
    print(f"\nInspecting RDF file: {os.path.basename(file_path)}")
    rdf_tensor = torch.load(file_path)
    
    # Print tensor shape and basic statistics
    print(f"Tensor shape: {rdf_tensor.shape}")
    print(f"Tensor type: {rdf_tensor.dtype}")
    print(f"Min value: {rdf_tensor.min().item():.4f}")
    print(f"Max value: {rdf_tensor.max().item():.4f}")
    print(f"Mean value: {rdf_tensor.mean().item():.4f}")
    
    return rdf_tensor

def plot_single_rdf(rdf_tensor, bins, title, save_path=None):
    """
    Plot a single RDF tensor.
    
    Parameters:
    -----------
    rdf_tensor : torch.Tensor
        RDF tensor with shape (n_frames, 1, n_bins) or (n_frames, n_bins)
    bins : torch.Tensor or np.ndarray
        The bin edges for the RDF
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    # Ensure proper tensor shape
    if rdf_tensor.dim() == 3 and rdf_tensor.shape[1] == 1:
        rdf_tensor = rdf_tensor.squeeze(1)  # (n_frames, n_bins)
    
    # Average over frames
    avg_rdf = rdf_tensor.mean(dim=0).numpy()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bins, avg_rdf, linewidth=2)
    ax.set_xlabel('Distance (Å)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close(fig)

def plot_triple_rdf(rdf_tensor, bins, species_list=['Ag', 'Cu'], title="Triple RDF", save_path=None):
    """
    Plot triple RDF tensor.
    
    Parameters:
    -----------
    rdf_tensor : torch.Tensor
        RDF tensor with shape (n_frames, 3, n_bins)
    bins : torch.Tensor or np.ndarray
        The bin edges for the RDF
    species_list : list of str
        List of chemical species
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    # Average over frames
    avg_rdf = rdf_tensor.mean(dim=0).numpy()  # (3, n_bins)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create labels for the three RDF components
    if len(species_list) == 2:
        labels = [f"{species_list[0]}-{species_list[0]}", 
                 f"{species_list[1]}-{species_list[1]}", 
                 f"{species_list[0]}-{species_list[1]}"]
        colors = ['blue', 'red', 'green']
    else:
        labels = [f"Component {i+1}" for i in range(avg_rdf.shape[0])]
        colors = ['blue', 'red', 'green']
    
    # Plot each component
    for i in range(avg_rdf.shape[0]):
        ax.plot(bins, avg_rdf[i], label=labels[i], linewidth=2, color=colors[i])
    
    ax.set_xlabel('Distance (Å)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close(fig)

def plot_multi_channel_rdf(rdf_tensor, bins, species_list=['Ag', 'Cu'], title="Multi-channel RDF", save_path=None):
    """
    Plot multi-channel RDF tensor.
    
    Parameters:
    -----------
    rdf_tensor : torch.Tensor
        RDF tensor with shape (n_frames, n_pairs, n_bins)
    bins : torch.Tensor or np.ndarray
        The bin edges for the RDF
    species_list : list of str
        List of chemical species
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    # Average over frames
    avg_rdf = rdf_tensor.mean(dim=0).numpy()  # (n_pairs, n_bins)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create labels for each channel
    n_species = len(species_list)
    labels = []
    for i in range(n_species):
        for j in range(i, n_species):
            labels.append(f"{species_list[i]}-{species_list[j]}")
    
    # Plot each channel
    for i in range(avg_rdf.shape[0]):
        ax.plot(bins, avg_rdf[i], label=labels[i], linewidth=2)
    
    ax.set_xlabel('Distance (Å)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close(fig)

def plot_concatenated_rdf(rdf_tensor, bins, n_pairs=3, species_list=['Ag', 'Cu'], title="Concatenated RDF", save_path=None):
    """
    Plot concatenated RDF tensor.
    
    Parameters:
    -----------
    rdf_tensor : torch.Tensor
        RDF tensor with shape (n_frames, 1, n_pairs * n_bins)
    bins : torch.Tensor or np.ndarray
        The bin edges for the RDF
    n_pairs : int
        Number of pairs (usually 3 for binary alloy: AA, BB, AB)
    species_list : list of str
        List of chemical species
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    # Average over frames and reshape
    if rdf_tensor.dim() == 3:
        avg_rdf = rdf_tensor.mean(dim=0).squeeze(0).numpy()  # (n_pairs * n_bins,)
    else:
        avg_rdf = rdf_tensor.mean(dim=0).numpy()
    
    n_bins = len(bins)
    
    # Reshape into separate components
    reshaped_rdf = avg_rdf.reshape(n_pairs, n_bins)
    
    # Create labels for each component
    n_species = len(species_list)
    labels = []
    for i in range(n_species):
        for j in range(i, n_species):
            labels.append(f"{species_list[i]}-{species_list[j]}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each component
    for i in range(n_pairs):
        ax.plot(bins, reshaped_rdf[i], label=labels[i], linewidth=2)
    
    ax.set_xlabel('Distance (Å)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close(fig)

def plot_weighted_attention_rdf(rdf_tensor, bins, title="Weighted Attention RDF", save_path=None):
    """
    Plot weighted attention RDF tensor.
    
    Parameters:
    -----------
    rdf_tensor : torch.Tensor
        RDF tensor with shape (n_frames, 1, n_bins)
    bins : torch.Tensor or np.ndarray
        The bin edges for the RDF
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    # Ensure proper tensor shape
    if rdf_tensor.dim() == 3:
        rdf_tensor = rdf_tensor.squeeze(1)  # (n_frames, n_bins)
    
    # Average over frames
    avg_rdf = rdf_tensor.mean(dim=0).numpy()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bins, avg_rdf, linewidth=2, color='purple')
    ax.set_xlabel('Distance (Å)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close(fig)

def plot_comparative_rdfs(rdf_files, output_dir, species_list=['Ag', 'Cu']):
    """
    Create comparative plots of different RDF approaches.
    
    Parameters:
    -----------
    rdf_files : dict
        Dictionary mapping file names to RDF tensors
    output_dir : str
        Directory to save output plots
    species_list : list of str
        List of chemical species
    """
    # Create the bins (assuming all RDFs use same binning)
    min_dist = 2.0
    max_dist = 8.0
    n_bins = 64
    bins = torch.linspace(min_dist, max_dist, n_bins).numpy()
    
    # Create a comparison of all approaches
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Single vs Triple
    ax1 = fig.add_subplot(gs[0, 0])
    if 'rdf_images_single_multi_channel.pt' in rdf_files:
        single_rdf = rdf_files['rdf_images_single_multi_channel.pt']
        if single_rdf.dim() == 3 and single_rdf.shape[1] == 1:
            avg_single = single_rdf.mean(dim=0).squeeze(0).numpy()
        else:
            avg_single = single_rdf.mean(dim=0).numpy()
        ax1.plot(bins, avg_single, label='Single', linewidth=2, color='blue')
    
    if 'rdf_images_triple_multi_channel.pt' in rdf_files:
        triple_rdf = rdf_files['rdf_images_triple_multi_channel.pt']
        avg_triple = triple_rdf.mean(dim=0).numpy()
        
        # Plot each component of triple RDF
        labels = [f"{species_list[0]}-{species_list[0]}", 
                 f"{species_list[1]}-{species_list[1]}", 
                 f"{species_list[0]}-{species_list[1]}"]
        colors = ['red', 'green', 'purple']
        
        for i in range(avg_triple.shape[0]):
            ax1.plot(bins, avg_triple[i], label=labels[i], linewidth=2, color=colors[i])
    
    ax1.set_xlabel('Distance (Å)', fontsize=12)
    ax1.set_ylabel('g(r)', fontsize=12)
    ax1.set_title('Single vs Triple RDF', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Alloy approaches
    ax2 = fig.add_subplot(gs[0, 1])
    methods = {
        'rdf_images_alloy_multi_channel.pt': 'Multi-channel',
        'rdf_images_alloy_weighted_attention.pt': 'Weighted Attention'
    }
    
    # Plot weighted attention first as a reference
    if 'rdf_images_alloy_weighted_attention.pt' in rdf_files:
        weighted_rdf = rdf_files['rdf_images_alloy_weighted_attention.pt']
        if weighted_rdf.dim() == 3:
            weighted_rdf = weighted_rdf.squeeze(1)  # (n_frames, n_bins)
        avg_weighted = weighted_rdf.mean(dim=0).numpy()
        ax2.plot(bins, avg_weighted, label='Weighted Attention', linewidth=3, color='black')
    
    # Plot multi-channel for comparison
    if 'rdf_images_alloy_multi_channel.pt' in rdf_files:
        multi_rdf = rdf_files['rdf_images_alloy_multi_channel.pt']
        avg_multi = multi_rdf.mean(dim=0).numpy()
        
        n_species = len(species_list)
        labels = []
        for i in range(n_species):
            for j in range(i, n_species):
                labels.append(f"{species_list[i]}-{species_list[j]}")
        
        colors = ['blue', 'red', 'green']
        for i in range(avg_multi.shape[0]):
            ax2.plot(bins, avg_multi[i], label=labels[i], linewidth=2, linestyle='--', color=colors[i])
    
    ax2.set_xlabel('Distance (Å)', fontsize=12)
    ax2.set_ylabel('g(r)', fontsize=12)
    ax2.set_title('Comparison of Alloy RDF Approaches', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Multi-channel vs Concatenated
    ax3 = fig.add_subplot(gs[1, 0])
    if 'rdf_images_alloy_multi_channel.pt' in rdf_files and 'rdf_images_alloy_concatenated.pt' in rdf_files:
        multi_rdf = rdf_files['rdf_images_alloy_multi_channel.pt']
        concat_rdf = rdf_files['rdf_images_alloy_concatenated.pt']
        
        avg_multi = multi_rdf.mean(dim=0).numpy()
        
        # For concatenated, reshape back to multi-channel format
        if concat_rdf.dim() == 3:
            avg_concat = concat_rdf.mean(dim=0).squeeze(0).numpy()
        else:
            avg_concat = concat_rdf.mean(dim=0).numpy()
            
        n_pairs = 3  # Ag-Ag, Cu-Cu, Ag-Cu for binary alloy
        reshaped_concat = avg_concat.reshape(n_pairs, n_bins)
        
        # Verify they're the same
        n_species = len(species_list)
        labels = []
        for i in range(n_species):
            for j in range(i, n_species):
                labels.append(f"{species_list[i]}-{species_list[j]}")
        
        colors = ['blue', 'red', 'green']
        for i in range(avg_multi.shape[0]):
            ax3.plot(bins, avg_multi[i], label=f"Multi: {labels[i]}", linewidth=2, color=colors[i])
            ax3.plot(bins, reshaped_concat[i], label=f"Concat: {labels[i]}", 
                    linewidth=1, linestyle='--', color=colors[i])
    
    ax3.set_xlabel('Distance (Å)', fontsize=12)
    ax3.set_ylabel('g(r)', fontsize=12)
    ax3.set_title('Multi-channel vs Concatenated RDF', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Frame-wise variation in a selected RDF
    ax4 = fig.add_subplot(gs[1, 1])
    if 'rdf_images_alloy_weighted_attention.pt' in rdf_files:
        weighted_rdf = rdf_files['rdf_images_alloy_weighted_attention.pt']
        
        if weighted_rdf.dim() == 3:
            weighted_rdf = weighted_rdf.squeeze(1)  # (n_frames, n_bins)
        
        # Calculate mean and standard deviation across frames
        mean_rdf = weighted_rdf.mean(dim=0).numpy()
        std_rdf = weighted_rdf.std(dim=0).numpy()
        
        # Plot mean with std as shaded region
        ax4.plot(bins, mean_rdf, linewidth=2, color='blue', label='Mean RDF')
        ax4.fill_between(bins, mean_rdf-std_rdf, mean_rdf+std_rdf, 
                        color='blue', alpha=0.2, label='±1σ')
        
        # Plot a few individual frames
        n_frames = weighted_rdf.shape[0]
        n_samples = min(5, n_frames)  # Show up to 5 individual frames
        
        # Select evenly spaced frames
        indices = np.linspace(0, n_frames-1, n_samples, dtype=int)
        
        for i, idx in enumerate(indices):
            ax4.plot(bins, weighted_rdf[idx].numpy(), linestyle='--', linewidth=1, 
                    alpha=0.7, label=f'Frame {idx}')
    
    ax4.set_xlabel('Distance (Å)', fontsize=12)
    ax4.set_ylabel('g(r)', fontsize=12)
    ax4.set_title('Frame-wise Variation in Weighted Attention RDF', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, 'comparative_rdf_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparative plot saved to: {save_path}")
    
    plt.close(fig)

def main():
    """Main function to load and visualize RDF files."""
    parser = argparse.ArgumentParser(description='Visualize RDF outputs from PyISV tests.')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing RDF output files (default: tests/test_outputs)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots (default: current dir)')
    parser.add_argument('--species', type=str, default='Ag,Cu',
                       help='Comma-separated list of chemical species (default: Ag,Cu)')
    
    args = parser.parse_args()
    
    # Set default input directory if not provided
    if args.input_dir is None:
        args.input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                     '..', 'tests', 'test_outputs'))
    
    print(f"Input directory: {args.input_dir}")
    print(f"Input directory exists: {os.path.exists(args.input_dir)}")
    
    if os.path.exists(args.input_dir):
        print(f"Files in input directory: {os.listdir(args.input_dir)}")
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.getcwd()
    
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
    
    # Parse species list
    species_list = args.species.split(',')
    print(f"Using species list: {species_list}")
    
    # Define RDF parameters
    min_dist = 2.0
    max_dist = 8.0
    n_bins = 64
    bins = torch.linspace(min_dist, max_dist, n_bins).numpy()
    
    # Dictionary to store loaded RDF files
    rdf_files = {}
    
    # Load and process all RDF files in the input directory
    try:
        file_list = os.listdir(args.input_dir)
        print(f"Found {len(file_list)} files in the input directory")
        
        for file_name in file_list:
            if file_name.startswith('rdf_images_') and file_name.endswith('.pt'):
                file_path = os.path.join(args.input_dir, file_name)
                print(f"Processing file: {file_path}")
                print(f"File exists: {os.path.exists(file_path)}")
                
                try:
                    rdf_tensor = load_and_inspect_rdf_file(file_path)
                    rdf_files[file_name] = rdf_tensor
                    print(f"Successfully loaded {file_name}")
                except Exception as e:
                    print(f"Error loading file {file_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Process and plot each file based on its type
                if 'single' in file_name:
                    plot_single_rdf(
                        rdf_tensor, 
                        bins, 
                        title="Single RDF", 
                        save_path=os.path.join(args.output_dir, f"{os.path.splitext(file_name)[0]}.png")
                    )
                
                elif 'triple' in file_name:
                    plot_triple_rdf(
                        rdf_tensor, 
                        bins, 
                        species_list=species_list,
                        title="Triple RDF", 
                        save_path=os.path.join(args.output_dir, f"{os.path.splitext(file_name)[0]}.png")
                    )
                
                elif 'multi_channel' in file_name and 'alloy' in file_name:
                    plot_multi_channel_rdf(
                        rdf_tensor, 
                        bins, 
                        species_list=species_list,
                        title="Multi-channel Alloy RDF", 
                        save_path=os.path.join(args.output_dir, f"{os.path.splitext(file_name)[0]}.png")
                    )
                
                elif 'concatenated' in file_name:
                    n_pairs = 3  # Default for binary alloy
                    plot_concatenated_rdf(
                        rdf_tensor, 
                        bins, 
                        n_pairs=n_pairs,
                        species_list=species_list,
                        title="Concatenated RDF", 
                        save_path=os.path.join(args.output_dir, f"{os.path.splitext(file_name)[0]}.png")
                    )
                
                elif 'weighted_attention' in file_name:
                    plot_weighted_attention_rdf(
                        rdf_tensor, 
                        bins, 
                        title="Weighted Attention RDF", 
                        save_path=os.path.join(args.output_dir, f"{os.path.splitext(file_name)[0]}.png")
                    )
            
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
    
    # Create comparative plots
    if len(rdf_files) > 1:
        try:
            plot_comparative_rdfs(rdf_files, args.output_dir, species_list=species_list)
        except Exception as e:
            print(f"Error creating comparative plots: {str(e)}")

if __name__ == '__main__':
    main()
