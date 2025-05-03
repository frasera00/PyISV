import unittest
from PyISV.train_utils import ClassificationTrainer

class TestTrainUtils(unittest.TestCase):

    def test_classification_trainer_initialization(self):
        # Example test case for ClassificationTrainer initialization
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        model = torch.nn.Linear(10, 2)
        dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
        train_loader = DataLoader(dataset, batch_size=10, num_workers=4)

        trainer = ClassificationTrainer(model, train_loader)
        self.assertIsNotNone(trainer, "ClassificationTrainer instance should not be None")

if __name__ == "__main__":
    unittest.main()