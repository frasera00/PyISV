# Neural Network architecture for Autoencoder and Classifier
# This module defines a neural network class that can be used 
# for both autoencoding and classification tasks.   

import torch, torch.nn as nn
from PyISV.utils.training_utils import *
from PyISV.utils.validation_utils import *
from PyISV.model_building import (build_encoder, build_decoder, build_bottleneck)

class NeuralNetwork(nn.Module):
    """NeuralNetwork class for Autoencoder and Classifier.
    Includes methods for building, training, and evaluating the model."""
    def __init__(self, params: dict) -> None:
        super(NeuralNetwork, self).__init__()

        # Accepts config dict as in architecture.py
        self.config = params
        self.model_type = self.config["type"]
        self.embed_dim = self.config["embedding_dim"]
        self.flat_dim = self.config["flattened_dim"]
        self.final_length = self.config["feature_map_length"]
        self.input_shape = self.config["input_shape"]
        self.n_channels = self.flat_dim // self.final_length

        self.encoder = build_encoder(self.config["encoder_layers"])
        self.bottleneck = build_bottleneck(self.config["bottleneck_layers"])
        self.decoder = build_decoder(self.config["decoder_layers"])
        self.embed_linear = self.bottleneck[0]
        self.decode_linear = self.bottleneck[1]
        
        # Store the last embedding for later access
        self.last_embedding = None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_flat = z.view(z.size(0), -1)  # Flatten the tensor
        embedding = self.embed_linear(z_flat)
        
        # Store embedding for later access
        self.last_embedding = embedding.detach().clone()

        # Pass through the bottleneck
        z_expanded = self.decode_linear(embedding)

        # Calculate the correct dimensions for reshaping
        z_out = z_expanded.view(z.size(0), self.n_channels, self.final_length)
        output = self.decoder(z_out)
        recon = self._reshape_output(x=x, recon=output)
        return recon
    
    def _reshape_output(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        input_length = x.shape[-1]
        output_length = recon.shape[-1]
        if output_length > input_length:
            recon = recon[..., :input_length]
        elif output_length < input_length:
            pad_amount = input_length - output_length
            recon = torch.nn.functional.pad(
                recon, 
                (0, pad_amount)
            )
        return recon

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input data without reconstruction"""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            z = self.encoder(x)
            z_flat = z.view(z.size(0), -1)
            embedding = self.embed_linear(z_flat)
        return embedding
    
    def get_last_embedding(self) -> torch.Tensor | None:
        """Return the last computed embedding"""
        return self.last_embedding