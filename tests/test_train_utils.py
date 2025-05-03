import os
import torch
import pytest
from PyISV.train_utils import Dataset

@pytest.fixture
def setup_data():
    # Create dummy data for testing
    input_data = torch.rand(100, 1, 50)  # 100 samples, 1 channel, 50 features
    labels = torch.randint(0, 10, (100,))  # Random labels for 10 classes
    return input_data, labels

def test_dataset_normalization(setup_data):
    input_data, labels = setup_data

    # Initialize dataset with normalization
    dataset = Dataset(input_data, labels, norm_inputs=True, norm_targets=False)

    # Check if normalization parameters are saved
    assert hasattr(dataset, 'subval_inputs'), "Normalization parameters (subval_inputs) not found."
    assert hasattr(dataset, 'divval_inputs'), "Normalization parameters (divval_inputs) not found."

    # Verify normalized data
    normalized_data = (input_data - dataset.subval_inputs) / dataset.divval_inputs
    assert torch.allclose(dataset.inputs, normalized_data, atol=1e-5), "Input data normalization failed."

def test_dataset_no_normalization(setup_data):
    input_data, labels = setup_data

    # Initialize dataset without normalization
    dataset = Dataset(input_data, labels, norm_inputs=False, norm_targets=False)

    # Verify data remains unchanged
    assert torch.equal(dataset.inputs, input_data), "Input data should not be normalized."
    assert torch.equal(dataset.targets, labels), "Labels should remain unchanged."