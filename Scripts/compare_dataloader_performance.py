import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
from torch.utils.data import DataLoader
from PyISV.train_utils import Dataset, PreloadedDataset

if __name__ == "__main__":
    # Generate synthetic data for testing
    num_samples = 72000
    num_features = 200
    inputs = torch.randn(num_samples, num_features)
    targets = torch.randn(num_samples, num_features)

    # Test with standard Dataset
    standard_dataset = Dataset(inputs, targets)
    standard_loader = DataLoader(standard_dataset, batch_size=64, shuffle=True, num_workers=0)

    start_time = time.time()
    for batch in standard_loader:
        pass  # Simulate processing
    standard_time = time.time() - start_time

    print(f"Time taken with standard Dataset: {standard_time:.4f} seconds")

    # Test with PreloadedDataset
    preloaded_dataset = PreloadedDataset(inputs, targets)
    preloaded_loader = DataLoader(preloaded_dataset, batch_size=64, shuffle=True, num_workers=4)

    start_time = time.time()
    for batch in preloaded_loader:
        pass  # Simulate processing
    preloaded_time = time.time() - start_time

    print(f"Time taken with PreloadedDataset: {preloaded_time:.4f} seconds")

    # Compare results
    speedup = standard_time / preloaded_time if preloaded_time > 0 else float('inf')
    print(f"Speedup using PreloadedDataset: {speedup:.2f}x")