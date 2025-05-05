import unittest
from unittest.mock import patch, MagicMock
from scripts.train_autoencoder import model, train_loader, valid_loader, lrate, loss_function, train_autoencoder
from PyISV.neural_network import NeuralNetwork

class TestModelTrainingScript(unittest.TestCase):
    @patch("torch.load")
    def test_training(self, mock_torch_load):
        """Test the training process of the model."""
        mock_torch_load.return_value = MagicMock(shape=(72000, 1, 200))

        # Ensure the model is initialized correctly
        self.assertIsInstance(model, NeuralNetwork)
        self.assertEqual(model.model_type, 'autoencoder')

        # Ensure data loaders are not empty
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(valid_loader), 0)

        # Call the encapsulated training function
        try:
            train_autoencoder(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                max_epochs=1,  # Run for 1 epoch to test functionality
                min_epochs=1,
                lrate=lrate,
                device=model.device,
                loss_function=loss_function,
                early_stopping=MagicMock(),  # Mock early stopping
                save_best_model=MagicMock()  # Mock save best model
            )
        except Exception as e:
            self.fail(f"Training failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()