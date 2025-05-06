import torch
import torch.nn as nn
import torch.jit
import numpy as np
from PyISV.model_building import build_encoder, build_decoder, build_bottleneck, build_classification_head

class NeuralNetwork(nn.Module):
    # --- Utility: shape assertion for debugging ---
    def _assert_shape(self, tensor, expected_shape, name="Tensor"):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} shape {tensor.shape} does not match expected {expected_shape}")

    # --- Utility: move model to device and update self.device ---
    def move_to(self, device):
        self.device = torch.device(device)
        self.to(self.device)
        return self

    """
    NeuralNetwork class for Autoencoder and Classifier.
    Includes methods for building, training, and evaluating the model.
    """
    def __init__(self, model_type, input_shape, embed_dim=2, num_classes=None, 
                 encoder_channels=None, decoder_channels=None, kernel_size=5,
                 activation_fn=nn.ReLU, use_pooling=True, device='cpu'):
        super(NeuralNetwork, self).__init__()

        # Validate input parameters
        if model_type == "classifier" and encoder_channels is None:
            raise ValueError("Encoder channels must be specified for classifier.")
        if model_type == "autoencoder" and (encoder_channels is None or decoder_channels is None):
            raise ValueError("Encoder and decoder channels must be specified for autoencoder.")

        self.device = torch.device(device)  # Default to CPU; can be updated during training
        self.model_type = model_type
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.activation_fn = activation_fn
        self.use_pooling = use_pooling
        self.kernel_size = kernel_size

        # Calculate num_encoder_final_channels internally based on encoder_channels
        self.num_encoder_final_channels = encoder_channels[-1]

        # Infer spatial dimension dynamically from input shape
        # Dynamically infer flat_dim using a dummy input tensor
        dummy_input = torch.zeros(1, input_shape[0], input_shape[1])  # Batch size 1, input channels, input length
        dummy_output = build_encoder(input_shape[0], encoder_channels, self.activation_fn, kernel_size=self.kernel_size)(dummy_input)
        self.flat_dim = dummy_output.size(1) * dummy_output.size(2)  # Channels * spatial dimension

        if self.model_type == "autoencoder":
            self.encoder = build_encoder(
                input_shape[0],          # input_channels
                encoder_channels,        # encoder_channels
                self.activation_fn,           # activation_fn
                kernel_size=self.kernel_size,  # Pass kernel size
            )
            # Ensure bottleneck input size matches encoder output
            self.bottleneck = build_bottleneck(
                flat_dim=self.flat_dim,
                embed_dim=self.embed_dim,
                num_encoder_final_channels=self.num_encoder_final_channels
            )
            self.decoder = build_decoder(
                [encoder_channels[-1]] + decoder_channels,
                self.activation_fn,
                output_length=input_shape[1],  # Pass the original input length to the decoder
                kernel_size=self.kernel_size  # Pass kernel size
            )
        elif self.model_type == "classifier":
            self.encoder = build_encoder(
                input_shape[0],
                encoder_channels,
                self.activation_fn,
            )
            # Enhanced classification head
            self.classification_head = nn.Sequential(
                nn.Linear(self.flat_dim, 128),  # Added a fully connected layer
                nn.ReLU(),
                nn.Dropout(0.5),  # Added dropout for regularization
                nn.Linear(128, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, num_classes)
            )
        else:
            raise ValueError("Invalid model_type. Choose from 'autoencoder' or 'classifier'.")

    def forward(self, x):
        # Assert input shape for debugging
        expected_shape = (x.size(0), self.encoder[0].in_channels, x.size(2))
        self._assert_shape(x, expected_shape, name="Input")
        
        if self.model_type == "autoencoder":
            z = self.encoder(x)
            z_flat = z.view(z.size(0), -1)  # Flatten the tensor
            z_bottleneck = self.bottleneck(z_flat)

            # Calculate the correct dimensions for reshaping
            new_flat_dim = z_bottleneck.size(1) // self.num_encoder_final_channels
            z_out = z_bottleneck.view(z.size(0), self.num_encoder_final_channels, new_flat_dim)  # Correctly reshape for decoder

            reconstructed = self.decoder(z_out)
            self._assert_shape(reconstructed, x.shape, name="Reconstructed")
            
            return reconstructed, z_out
        
        elif self.model_type == "classifier":
            z = self.encoder(x)
            z = z.view(z.size(0), -1)
            logits = self.classification_head(z)
            return logits
        else:
            raise ValueError("Invalid model_type. Choose from 'autoencoder' or 'classifier'.")

    def train_model(self, train_loader, val_loader=None, epochs=10,
                    lr=1e-3, weight_decay=0, device='cpu', criterion=None, 
                    use_separate_target=False, amp=False, grad_clip=None, 
                    scheduler=None, early_stopping=None, checkpoint_path=None, 
                    log_interval=5):
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
        - amp: If True, use automatic mixed precision (AMP) for faster training on GPUs.
        - grad_clip: If set, clip gradients to this value.
        - scheduler: Learning rate scheduler (optional). Supports schedulers that require validation loss (e.g., ReduceLROnPlateau) or standard schedulers.
        - early_stopping: Early stopping callback that takes validation loss and returns True if training should stop (optional).
        - checkpoint_path: If set, save model checkpoints to this path after each epoch (optional).
        - log_interval: Print training progress every log_interval epochs.
        """
        
        self.move_to(device)
        if criterion is None:
            criterion = nn.CrossEntropyLoss() if self.model_type == "classifier" else nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler() if amp else None
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            correct = 0
            total = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                if amp:
                    with torch.cuda.amp.autocast():
                        if self.model_type == "classifier":
                            outputs = self.forward(x)
                            loss = criterion(outputs, y)
                            _, predicted = torch.max(outputs, 1)
                            correct += (predicted == y).sum().item()
                            total += y.size(0)
                        else:
                            reconstructed, _ = self.forward(x)
                            loss = criterion(reconstructed, y if use_separate_target else x)
                else:
                    if self.model_type == "classifier":
                        outputs = self.forward(x)
                        loss = criterion(outputs, y)
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == y).sum().item()
                        total += y.size(0)
                    else:
                        reconstructed, _ = self.forward(x)
                        loss = criterion(reconstructed, y if use_separate_target else x)

                if amp:
                    scaler.scale(loss).backward()
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                    optimizer.step()

                total_loss += loss.item()


            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader, criterion, use_separate_target)

            # Scheduler step: handle ReduceLROnPlateau and others
            if scheduler is not None:
                if hasattr(scheduler, 'step') and 'ReduceLROnPlateau' in scheduler.__class__.__name__:
                    # For ReduceLROnPlateau, step with validation loss
                    if val_loss is not None:
                        scheduler.step(val_loss)
                else:
                    scheduler.step()

            if (epoch + 1) % log_interval == 0:
                if self.model_type == "classifier":
                    accuracy = correct / total if total > 0 else 0
                    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss:.4f}")

            # Early stopping
            if early_stopping is not None and val_loss is not None:
                if early_stopping(val_loss):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Checkpointing
            if checkpoint_path is not None and val_loss is not None:
                torch.save(self.state_dict(), checkpoint_path)

    def validate(self, val_loader, criterion, use_separate_target=False, scaler_subval=None, scaler_divval=None, save_outputs=False, encoder_only=False):
        """
        Validate the model using the provided validation data loader.

        Parameters:
        - val_loader: DataLoader for validation data.
        - criterion: Loss function to use.
        - use_separate_target: If True, use the target dataset (y) for the loss function instead of the input data (x).
        - scaler_subval: Subtraction value for input scaling (optional).
        - scaler_divval: Division value for input scaling (optional).
        - save_outputs: If True, save embeddings and reconstructed outputs to files.
        - encoder_only: If True, only evaluate the encoder and save bottleneck embeddings.
        """
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        embeddings = []
        outputs = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Scale inputs if scalers are provided
                if scaler_subval is not None and scaler_divval is not None:
                    x = (x - scaler_subval) / scaler_divval

                if self.model_type == "classifier":
                    outputs_batch = self.forward(x)
                    loss = criterion(outputs_batch, y)
                    _, predicted = torch.max(outputs_batch, 1)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
                else:  # Autoencoder
                    if encoder_only:
                        embeddings_batch = self.encoder(x).view(x.size(0), -1).detach().cpu().numpy()
                        embeddings.append(embeddings_batch)
                        continue

                    reconstructed, bottleneck = self.forward(x)
                    if use_separate_target:
                        loss = criterion(reconstructed, y)
                    else:
                        loss = criterion(reconstructed, x)

                    if save_outputs:
                        outputs.append(reconstructed.detach().cpu().numpy())
                        embeddings.append(bottleneck.detach().cpu().numpy())

                total_loss += loss.item()

        if self.model_type == "classifier":
            accuracy = correct / total if total > 0 else 0
            print(f"Validation Loss: {total_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        else:
            print(f"Validation Loss: {total_loss:.4f}")

        # Save outputs and embeddings if required
        if save_outputs and len(embeddings) > 0:
            np.save("embeddings.npy", np.vstack(embeddings))
        if save_outputs and len(outputs) > 0:
            np.save("reconstructed_outputs.npy", np.vstack(outputs))

        return total_loss

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