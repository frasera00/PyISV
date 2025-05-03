import torch
import torch.nn as nn

def build_encoder(input_channels, input_length, encoder_channels, activation_fn):
    """Builds the encoder for 1D data."""
    layers = []
    in_channels = input_channels
    for out_channels in encoder_channels:
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2))  # padding=same
        layers.append(nn.MaxPool1d(kernel_size=2))
        layers.append(activation_fn())
        layers.append(nn.BatchNorm1d(out_channels))
        in_channels = out_channels
    return nn.Sequential(*layers)

def build_bottleneck(flat_dim, embed_dim, num_encoder_final_channels):
    """Builds the bottleneck layer for 1D data."""
    layers = [
        nn.Flatten(),
        nn.Linear(flat_dim * num_encoder_final_channels, embed_dim),
        nn.Linear(embed_dim, flat_dim * num_encoder_final_channels),
        nn.ReLU()
    ]
    return nn.Sequential(*layers)

def build_decoder(decoder_channels, activation_fn, output_length):
    """Builds the decoder for 1D data and ensures it reconstructs the original input size."""
    layers = []
    in_channels = decoder_channels[0]
    for out_channels in decoder_channels[1:]:
        layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=True))
        layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=5, padding=2))
        layers.append(activation_fn())
        layers.append(nn.BatchNorm1d(out_channels))
        in_channels = out_channels
    layers.append(nn.Conv1d(in_channels, 1, kernel_size=5, padding=2))  # Final layer to match input channels

    # Ensure the output length matches the original input length
    layers.append(nn.Upsample(size=output_length, mode='linear', align_corners=True))

    return nn.Sequential(*layers)