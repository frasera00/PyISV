import torch
import torch.nn as nn
import torch.jit
from PyISV.classification_utils import build_classification_head
from PyISV.autoencoder_utils import build_encoder, build_decoder, build_bottleneck

class NeuralNetwork(nn.Module):
    """
    NeuralNetwork class for Autoencoder and Classifier.
    Includes methods for building, training, and evaluating the model.
    """
    def __init__(self, model_type, input_shape, embed_dim=2, num_classes=None, 
                 encoder_channels=[128, 64, 32], decoder_channels=[32, 64, 128],
                 activation_fn=nn.ReLU, use_pooling=True):
        super(NeuralNetwork, self).__init__()

        self.device = torch.device('cpu')  # Default to CPU; can be updated during training
        self.model_type = model_type
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.activation_fn = activation_fn
        self.use_pooling = use_pooling

        # Calculate num_encoder_final_channels internally based on encoder_channels
        self.num_encoder_final_channels = encoder_channels[-1]

        # Calculate flat_dim internally based on input shape and encoder_channels
        self.flat_dim = input_shape[1] // (2 ** len(encoder_channels))

        if self.model_type == "autoencoder":
            self.encoder = build_encoder(
                input_shape[0],  # input_channels
                input_shape[1],  # input_length
                encoder_channels,  # encoder_channels
                activation_fn  # activation_fn
            )
            self.bottleneck = build_bottleneck(
                flat_dim=self.flat_dim,  # Use flat_dim attribute
                embed_dim=embed_dim,
                num_encoder_final_channels=encoder_channels[-1]
            )
            self.decoder = build_decoder(
                [encoder_channels[-1]] + decoder_channels,
                activation_fn,
                output_length=input_shape[1]  # Pass the original input length to the decoder
            )
        elif self.model_type == "classifier":
            self.encoder = build_encoder(
                input_shape[0],  # input_channels
                input_shape[1],  # input_length
                encoder_channels,  # encoder_channels
                activation_fn  # activation_fn
            )
            self.classification_head = build_classification_head(self.encoder, input_shape, embed_dim, num_classes, activation_fn)
        else:
            raise ValueError("Invalid model_type. Choose from 'autoencoder' or 'classifier'.")

    def forward(self, x):
        if self.model_type == "autoencoder":
            z = self.encoder(x)
            z = z.view(z.size(0), -1)  # Flatten the tensor
            z = self.bottleneck(z)
            z = z.view(z.size(0), -1, self.flat_dim)  # Reshape to match decoder input
            reconstructed = self.decoder(z)
            return reconstructed, z
        elif self.model_type == "classifier":
            z = self.encoder(x)
            z = z.view(z.size(0), -1)
            logits = self.classification_head(z)
            return logits
        else:
            raise ValueError("Invalid model_type. Choose from 'autoencoder' or 'classifier'.")

    def train_model(self, train_loader, val_loader=None, epochs=10, lr=1e-3, weight_decay=0, device='cpu', criterion=None, use_separate_target=False):
        """
        Train the model using the provided data loaders.

        Parameters:
        - train_loader: DataLoader for training data.
        - val_loader: DataLoader for validation data (optional).
        - epochs: Number of training epochs.
        - lr: Learning rate.
        - weight_decay: Weight decay for the optimizer.
        - device: Device to run the training on ('cpu' or 'cuda').
        - criterion: Loss function to use (default: CrossEntropyLoss for classifier, MSELoss for autoencoders).
        - use_separate_target: If True, use the target dataset (y) for the loss function instead of the input data (x).
        """
        self.device = torch.device(device)
        self.to(self.device)

        if criterion is None:
            criterion = nn.CrossEntropyLoss() if self.model_type == "classifier" else nn.MSELoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            correct = 0
            total = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()

                if self.model_type == "classifier":
                    outputs = self.forward(x)
                    loss = criterion(outputs, y)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
                else:  # Autoencoder
                    reconstructed, _ = self.forward(x)
                    if use_separate_target:
                        loss = criterion(reconstructed, y)
                    else:
                        loss = criterion(reconstructed, x)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.model_type == "classifier":
                accuracy = correct / total
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

            if val_loader:
                self.validate(val_loader, criterion, use_separate_target)

    def validate(self, val_loader, criterion, use_separate_target=False):
        """
        Validate the model using the provided validation data loader.

        Parameters:
        - val_loader: DataLoader for validation data.
        - criterion: Loss function to use.
        - use_separate_target: If True, use the target dataset (y) for the loss function instead of the input data (x).
        """
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                if self.model_type == "classifier":
                    outputs = self.forward(x)
                    loss = criterion(outputs, y)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
                else:  # Autoencoder
                    reconstructed, _ = self.forward(x)
                    if use_separate_target:
                        loss = criterion(reconstructed, y)
                    else:
                        loss = criterion(reconstructed, x)

                total_loss += loss.item()

        if self.model_type == "classifier":
            accuracy = correct / total
            print(f"Validation Loss: {total_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        else:
            print(f"Validation Loss: {total_loss:.4f}")

    def evaluate(self, test_loader):
        """
        Evaluate the model on the test dataset.
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.forward(x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def apply_jit(self, example_input):
        """Applies JIT tracing to the model and returns the traced model."""
        traced_model = torch.jit.trace(self, example_input)
        return traced_model