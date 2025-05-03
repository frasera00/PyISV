import unittest
from unittest.mock import patch, MagicMock
from scripts.train_classification import model, train_loader, valid_loader, lrate
from PyISV.neural_network import NeuralNetwork
from torch.nn import CrossEntropyLoss

class TestClassificationTrainingScript(unittest.TestCase):
    @patch("torch.load")
    def test_training(self, mock_torch_load):
        """Test the training process of the classification model."""
        # Mock torch.load to avoid FileNotFoundError
        mock_torch_load.side_effect = [
            MagicMock(shape=(72000, 1, 200)),  # Mock input data
            MagicMock(shape=(72000,))  # Mock labels
        ]

        # Ensure the model is initialized correctly
        self.assertIsInstance(model, NeuralNetwork)
        self.assertEqual(model.model_type, 'classifier')

        # Ensure data loaders are not empty
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(valid_loader), 0)

        # Call the training method of the model
        try:
            model.train_model(
                train_loader=train_loader,
                val_loader=valid_loader,
                epochs=1,  # Run for 1 epoch to test functionality
                lr=lrate,
                device=model.device,
                criterion=CrossEntropyLoss()  # Use a real loss function
            )
        except Exception as e:
            self.fail(f"Training failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()