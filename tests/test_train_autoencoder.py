import torch
import torch.nn as nn
import unittest
from unittest.mock import patch, MagicMock, mock_open
import importlib.util
import pathlib

# Patch torch.jit.script to a no-op for test robustness
torch.jit.script = lambda fn=None, *a, **kw: fn if fn is not None else (lambda f: f)

class TestTrainerMinimal(unittest.TestCase):
    @patch("torch.load")
    @patch("numpy.save")
    @patch("scripts.train_autoencoder.open", new_callable=mock_open)
    @patch("PyISV.training_utils.SaveBestModel", autospec=True)
    @patch("PyISV.training_utils.setup_tensorboard_writer", return_value=(MagicMock(), "dummy_dir"))
    def test_trainer_minimal(self, mock_tb_writer, mock_savebest, mock_openfile, mock_npsave, mock_torchload):
        # Provide dummy data for torch.load
        mock_torchload.return_value = torch.randn(8, 1, 4)

        # Minimal config
        config = {
            "GENERAL": {
                "device": "cpu",
                "seed": 0,
                "apply_jit_tracing": False,
                "use_ddp": False,
                "use_lr_finder": False,
                "use_tensorboard": False,
                "num_classes": 1,
                "num_bins": 4,
                "num_channels": 1,
                "num_features": 4,
                "num_features_flattened": 4,
            },
            "MODEL": {
                "type": "autoencoder",
                "input_shape": [1, 4],
                "embed_dim": 2,
                "encoder_layers": {
                    0: [nn.Conv1d(1, 2, 3, padding=1), nn.ReLU()],
                },
                "bottleneck_layers": {
                    0: [nn.Flatten(), nn.Linear(2*4, 2), nn.ReLU()],
                    1: [nn.Linear(2, 2*4), nn.ReLU()],
                },
                "decoder_layers": {
                    0: [nn.ConvTranspose1d(2, 1, 3, padding=1)],
                },
            },
            "TRAINING": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "min_epochs": 1,
                "max_epochs": 1,
                "train_size": 0.5,
                "early_stopping": {"patience": 1, "delta": 0.1},
                "loss_function": torch.nn.MSELoss(),
            },
            "LEARNING": {
                "learning_rate": 0.001,
                "scheduled_lr": False,
                "lr_warmup_epochs": 0,
                "milestones": [],
                "gamma": 0.5,
            },
            "INPUTS": {
                "dataset": "dummy.pt",
                "normalization": "minmax",
            },
            "OUTPUTS": {
                "normalization_params": {"inputs": "dummy.npy", "targets": "dummy.npy"},
                "log_file": "dummy.log",
                "stats_file": "dummy_stats.dat",
                "model_architecture_file": "dummy_arch.dat",
                "model_file": "dummy_model.pt",
            },
        }

        # Dynamically import Trainer from scripts/train_autoencoder.py
        train_autoencoder_path = pathlib.Path(__file__).parent.parent / 'scripts' / 'train_autoencoder.py'
        spec = importlib.util.spec_from_file_location('train_autoencoder', str(train_autoencoder_path))
        train_autoencoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_autoencoder)
        Trainer = train_autoencoder.Trainer

        trainer = Trainer(config)
        trainer.prepare_data()
        trainer.prepare_model()
        trainer.run()

        self.assertTrue(hasattr(trainer, "model"))
        self.assertEqual(str(trainer.device), "cpu")

if __name__ == "__main__":
    unittest.main()