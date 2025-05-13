import unittest
from unittest.mock import patch, MagicMock
import importlib
import torch

class TestModelTrainingScript(unittest.TestCase):
    @patch("torch.load")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.cuda.device_count", return_value=1)
    def test_training_cpu(self, mock_device_count, mock_is_available, mock_torch_load):
        """Test the training process of the model on CPU only."""
        mock_torch_load.return_value = torch.randn(72000, 1, 320)

        import scripts.train_autoencoder as ta
        ta.use_ddp = False
        ta.pin_memory = False
        ta.num_workers = 0
        
        # Patch output paths to avoid overwriting real outputs
        ta.RUN_ID = "TEST_RUN_ID"
        ta.model_save_path = f"test_{ta.model_save_path}"
        ta.stats_file = f"test_{ta.stats_file}"
        ta.model_architecture_file = f"test_{ta.model_architecture_file}"
        importlib.reload(ta)

        # Ensure the model is initialized correctly
        self.assertEqual(getattr(ta.model, 'model_type', None), 'autoencoder')

        # Ensure data loaders are not empty
        self.assertGreater(len(ta.train_loader), 0)
        self.assertGreater(len(ta.valid_loader), 0)

        # Call the encapsulated training function for 1 epoch
        try:
            ta.train_autoencoder(
                model=ta.model,
                train_loader=ta.train_loader,
                valid_loader=ta.valid_loader,
                training_params={
                    'max_epochs': 1,
                    'min_epochs': 1,
                    'lrate': ta.lrate,
                    'scheduled_lr': False,
                    'start_epoch': 0,
                    'use_ddp': False,
                },
                utilities={
                    'early_stopping': MagicMock(),
                    'save_best_model': MagicMock()
                },
                device=torch.device('cpu'),
                loss_function=ta.LOSS_FUNCTION
            )
        except Exception as e:
            self.fail(f"Training failed with exception (CPU): {e}")

if __name__ == "__main__":
    unittest.main()