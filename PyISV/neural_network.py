# Neural Network architecture for Autoencoder and Classifier
# This module defines a neural network class that can be used 
# for both autoencoding and classification tasks.   

import torch
import torch.nn as nn
import torch.jit
import copy
from PyISV.model_building import (
    build_encoder,
    build_decoder,
    build_bottleneck,
)
from PyISV.training_utils import *
from PyISV.validation_utils import *

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
        
        self.embed_linear, self.decode_linear = build_bottleneck(self.config["bottleneck_layers"])
        self.decoder = build_decoder(self.config["decoder_layers"])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_flat = z.view(z.size(0), -1)  # Flatten the tensor
        embedding = self.embed_linear(z_flat)
        z_expanded = self.decode_linear(embedding)

        # Calculate the correct dimensions for reshaping
        z_out = z_expanded.view(z.size(0), self.n_channels, self.final_length)
        reconstructed = self.decoder(z_out)

        # Output shape handling: crop or pad to match input length
        input_length = x.shape[-1]
        output_length = reconstructed.shape[-1]
        if output_length > input_length:
            reconstructed = reconstructed[..., :input_length]
        elif output_length < input_length:
            pad_amount = input_length - output_length
            reconstructed = torch.nn.functional.pad(reconstructed, (0, pad_amount))

        return reconstructed, embedding