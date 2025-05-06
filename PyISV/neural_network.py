# -*- coding: utf-8 -*-

# Import necessary libraries
import torch
import torch.nn as nn
import torch.jit

# Import custom modules
from PyISV.model_building import build_encoder, build_decoder, build_bottleneck, build_classification_head
from PyISV.train_utils import InvalidModelTypeError

# Import training and validation helpers
from PyISV.training_helpers import *
from PyISV.validation_helpers import *

class NeuralNetwork(nn.Module):
    """
    NeuralNetwork class for Autoencoder and Classifier.
    Includes methods for building, training, and evaluating the model.
    """

    # Initialization of the NeuralNetwork class
    def __init__(self, config: dict):
        super(NeuralNetwork, self).__init__()

        # Validate input parameters
        model_type = config["type"]
        encoder_channels = config.get("encoder_channels")
        decoder_channels = config.get("decoder_channels")
        if model_type == "classifier" and encoder_channels is None:
            raise ValueError("Encoder channels must be specified for classifier.")
        if self._autoencoder_channels_missing(model_type, encoder_channels, decoder_channels):
            raise ValueError("Encoder and decoder channels must be specified for autoencoder.")
    
        self.device = torch.device(config.get("device", "cpu"))
        self.model_type = model_type
        self.embed_dim = config.get("embed_dim", 2)
        self.num_classes = config.get("num_classes")
        self.use_pooling = config.get("use_pooling", True)
        self.kernel_size = config.get("kernel_size", 5)
        self.activation_fn = getattr(torch.nn, config["activation_fn"], torch.nn.ReLU)

        # Calculate num_encoder_final_channels internally based on encoder_channels
        self.num_encoder_final_channels = encoder_channels[-1]

        # Infer spatial dimension dynamically from input shape
        input_shape = config["input_shape"]
        dummy_input = torch.zeros(1, input_shape[0], input_shape[1])
        dummy_output = build_encoder(input_shape[0], encoder_channels, self.activation_fn, kernel_size=self.kernel_size)(dummy_input)
        self.flat_dim = dummy_output.size(1) * dummy_output.size(2)

        if self.model_type == "autoencoder":
            self.encoder = build_encoder(
                input_shape[0],
                encoder_channels,
                self.activation_fn,
                kernel_size=self.kernel_size,
            )
            self.bottleneck = build_bottleneck(
                flat_dim=self.flat_dim,
                embed_dim=self.embed_dim,
            )
            self.decoder = build_decoder(
                [encoder_channels[-1]] + decoder_channels,
                self.activation_fn,
                output_length=input_shape[1],
                kernel_size=self.kernel_size
            )
        elif self.model_type == "classifier":
            self.encoder = build_encoder(
                input_shape[0],
                encoder_channels,
                self.activation_fn,
            )
            self.classification_head = build_classification_head(
                embed_dim=self.embed_dim,
                num_classes=self.num_classes,
                activation_fn=self.activation_fn,
                flat_dim=self.flat_dim,
                hidden_dim=64,
                dropout=None
            )
        else:
            raise InvalidModelTypeError()
  
    # --- Forward pass ---
    def forward(self, x):
        # Assert input shape for debugging
        expected_shape = (x.size(0), self.encoder[0].in_channels, x.size(2))
        assert_shape(x, expected_shape, name="Input")
        
        if self.model_type == "autoencoder":
            z = self.encoder(x)
            z_flat = z.view(z.size(0), -1)  # Flatten the tensor
            z_bottleneck = self.bottleneck(z_flat)

            # Calculate the correct dimensions for reshaping
            new_flat_dim = z_bottleneck.size(1) // self.num_encoder_final_channels
            z_out = z_bottleneck.view(z.size(0), self.num_encoder_final_channels, new_flat_dim)  # Correctly reshape for decoder

            reconstructed = self.decoder(z_out)
            assert_shape(reconstructed, x.shape, name="Reconstructed")
            
            return reconstructed, z_out
        
        elif self.model_type == "classifier":
            z = self.encoder(x)
            z = z.view(z.size(0), -1)
            logits = self.classification_head(z)
            return logits
        else:
            raise InvalidModelTypeError()

    # --- Training and Evaluation Methods ---
    def train_model(self, train_loader, val_loader=None, config=None):
        """
        Train the model using the provided data loaders and a configuration dictionary.

        Parameters:
        - train_loader: DataLoader for training data.
        - val_loader: DataLoader for validation data (optional).
        - config: Dictionary containing training configuration.
        """
        # Set defaults
        defaults = {
            'epochs': 10,
            'lr': 1e-3,
            'weight_decay': 0,
            'device': 'cpu',
            'criterion': None,
            'use_separate_target': False,
            'amp': False,
            'grad_clip': None,
            'scheduler': None,
            'early_stopping': None,
            'checkpoint_path': None,
            'log_interval': 5
        }
        if config is None:
            config = defaults

        self._move_to(config['device'])
        self.amp = config['amp']
        self.grad_clip = config['grad_clip']
        self.use_separate_target = config['use_separate_target']
        self.log_interval = config['log_interval']
        self.early_stopping = config['early_stopping']
        self.checkpoint_path = config['checkpoint_path']
        self.min_epochs = config.get('min_epochs', 1)
        self.max_epochs = config.get('max_epochs', config['epochs'])
        self.learning_rate = config['learning_rate']
        self.scheduler = None

        # Initialize criterion and optimizer before scheduler
        self.criterion = config['criterion']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=config['weight_decay'])
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

        # Create scheduler if config provides scheduler_cfg
        scheduler_cfg = config.get('scheduler_cfg', None)
        if scheduler_cfg is not None:
            if scheduler_cfg["type"] == "MultiStepLR":
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=scheduler_cfg.get("milestones", [100, 250]),
                    gamma=scheduler_cfg.get("gamma", 0.5)
                )
            # Add more scheduler types here as needed

        best_model_callback = config.get('best_model_callback', None) if config else None
        best_val_loss = float('inf')
        epoch = 0
        stop = False
        while epoch < self.max_epochs and not stop:
            self.train()
            total_loss = 0
            correct = 0
            total = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                if self.amp:
                    with torch.cuda.amp.autocast():
                        if self.model_type == "classifier":
                            loss, batch_correct, batch_total = train_classifier_batch(self, x, y)
                        elif self.model_type == "autoencoder":
                            loss, batch_correct, batch_total = train_autoencoder_batch(self, x, y)
                        else:
                            raise InvalidModelTypeError()
                    self.scaler.scale(loss).backward()
                    clip_gradients(self)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.model_type == "classifier":
                        loss, batch_correct, batch_total = train_classifier_batch(self, x, y)
                    elif self.model_type == "autoencoder":
                        loss, batch_correct, batch_total = train_autoencoder_batch(self, x, y)
                    else:
                        raise InvalidModelTypeError()
                    loss.backward()
                    clip_gradients(self)
                    self.optimizer.step()
                total_loss += loss.item()
                if self.model_type == "classifier":
                    correct += batch_correct
                    total += batch_total

            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)

            # Save best model if callback is provided and val_loss improves
            if best_model_callback is not None and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_callback(val_loss, total_loss, epoch, self, self.optimizer)

            scheduler_step(self, val_loss)

            if (epoch + 1) % self.log_interval == 0:
                log_epoch(self, epoch, total_loss, correct, total)

            # Only allow early stopping after min_epochs
            if epoch + 1 >= self.min_epochs:
                if early_stop_and_checkpoint(self, val_loss, epoch):
                    stop = True
            epoch += 1

    def validate(self, val_loader, config=None):
        """
        Validate the model using the provided validation data loader.

        Parameters:
        - val_loader: DataLoader for validation data.
        - config: Dictionary containing validation configuration.
        """
        if config is None:
            config = {}
        criterion = config.get('criterion', getattr(self, 'criterion', None))
        use_separate_target = config.get('use_separate_target', getattr(self, 'use_separate_target', False))
        scaler_subval = config.get('scaler_subval', None)
        scaler_divval = config.get('scaler_divval', None)
        save_outputs = config.get('save_outputs', False)
        encoder_only = config.get('encoder_only', False)

        self.eval()
        total_loss, correct, total = 0, 0, 0
        embeddings = []
        outputs = []

        with torch.no_grad():
            if self.model_type == "classifier":
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    x = normalize_input(x, scaler_subval, scaler_divval)
                    loss, batch_correct, batch_total = validate_classifier(self, x, y, criterion)
                    correct += batch_correct
                    total += batch_total
                    total_loss += loss.item()
            elif self.model_type == "autoencoder":
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    x = normalize_input(x, scaler_subval, scaler_divval)
                    options = {
                        "use_separate_target": use_separate_target,
                        "encoder_only": encoder_only,
                        "save_outputs": save_outputs,
                        "embeddings": embeddings,
                        "outputs": outputs,
                    }
                    loss, _, _ = validate_autoencoder(self, x, y, criterion, options)
                    if loss is None:
                        continue
                    total_loss += loss.item()
            else:
                raise InvalidModelTypeError()

        if self.model_type == "classifier":
            accuracy = correct / total if total > 0 else 0
            print(f"Validation Loss: {total_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        else:
            print(f"Validation Loss: {total_loss:.4f}")

        save_validation_outputs(save_outputs, embeddings, outputs)
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

    # --- Utility Methods --- #
    def apply_jit(self, example_input):
        """Applies JIT tracing to the model and returns the traced model."""
        traced_model = torch.jit.trace(self, example_input)
        return traced_model
    
    def _move_to(self, device):
        self.device = torch.device(device)
        self.to(self.device)
        return self
    
    def _autoencoder_channels_missing(self, model_type, encoder_channels, decoder_channels):
        """Return True if model_type is 'autoencoder' and encoder or decoder channels are missing."""
        return model_type == "autoencoder" and (encoder_channels is None or decoder_channels is None)
