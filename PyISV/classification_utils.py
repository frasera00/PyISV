import torch
import torch.nn as nn

def build_classification_head(encoder, input_shape, embed_dim, num_classes, activation_fn):
    """Builds a classification head."""
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape)
        dummy_output = encoder(dummy_input)
        flat_dim = dummy_output.view(1, -1).shape[1]

    return nn.Sequential(
        nn.Linear(flat_dim, embed_dim),
        activation_fn(),
        nn.Dropout(0.5),
        nn.Linear(embed_dim, num_classes),
    )


class ClassificationTrainer:
    """
    Trainer for supervised classification using CrossEntropyLoss.
    Implements a clean and efficient training/validation loop.
    """
    def __init__(self, model, train_loader, val_loader=None, lr=1e-3, weight_decay=0, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _run_epoch(self, loader, training=False):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                if training:
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        return total_loss / total, correct / total

    def train_epoch(self):
        return self._run_epoch(self.train_loader, training=True)

    def validate_epoch(self):
        if self.val_loader is None:
            raise ValueError("No validation loader provided.")
        return self._run_epoch(self.val_loader, training=False)
