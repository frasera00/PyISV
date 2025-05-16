
import os
import torch
import unittest
from PyISV_project.PyISV.training_utilities import Dataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        # Create dummy data for testing
        self.input_data = torch.rand(100, 1, 50)  # 100 samples, 1 channel, 50 features
        self.labels = torch.randint(0, 10, (100,))  # Random labels for 10 classes

    def test_dataset_normalization(self):
        # Initialize dataset with normalization
        dataset = Dataset(self.input_data, self.labels, norm_inputs=True, norm_targets=False)

        # Check if normalization parameters are saved
        self.assertTrue(hasattr(dataset, 'subval_inputs'), "Normalization parameters (subval_inputs) not found.")
        self.assertTrue(hasattr(dataset, 'divval_inputs'), "Normalization parameters (divval_inputs) not found.")

        # Verify normalized data
        normalized_data = (self.input_data - dataset.subval_inputs) / dataset.divval_inputs
        self.assertTrue(torch.allclose(dataset.inputs, normalized_data, atol=1e-5), "Input data normalization failed.")

    def test_dataset_no_normalization(self):
        # Initialize dataset without normalization
        dataset = Dataset(self.input_data, self.labels, norm_inputs=False, norm_targets=False)

        # Verify data remains unchanged
        self.assertTrue(torch.equal(dataset.inputs, self.input_data), "Input data should not be normalized.")
        self.assertTrue(torch.equal(dataset.targets, self.labels), "Labels should remain unchanged.")

if __name__ == "__main__":
    unittest.main()