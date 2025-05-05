import torch.nn as nn

def build_encoder(input_channels, encoder_channels, activation_fn, kernel_size=5):
    """Builds the encoder for 1D data with `same` padding."""
    layers = []
    in_channels = input_channels
    for out_channels in encoder_channels:
        padding = kernel_size // 2  # Calculate padding for `same` behavior
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
        layers.append(nn.MaxPool1d(kernel_size=2))
        layers.append(activation_fn())
        layers.append(nn.BatchNorm1d(out_channels))
        in_channels = out_channels
    return nn.Sequential(*layers)

def build_bottleneck(flat_dim, embed_dim):
    """Builds the bottleneck layer for 1D data."""
    # Explicitly calculate the expected input size for the bottleneck layer
    expected_input_size = flat_dim

    layers = [
        nn.Flatten(),
        nn.Linear(expected_input_size, embed_dim),  # Use the correct input size
        nn.Linear(embed_dim, expected_input_size),
        nn.ReLU()
    ]
    return nn.Sequential(*layers)

def build_decoder(decoder_channels, activation_fn, output_length, kernel_size=5):
    """Builds the decoder for 1D data with `same` padding and ensures it reconstructs the original input size."""
    
    padding = kernel_size // 2  # Calculate padding for `same` behavior
    layers = []
    in_channels = decoder_channels[0]
    for out_channels in decoder_channels[1:]:
        layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=True))
        layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
        layers.append(activation_fn())
        layers.append(nn.BatchNorm1d(out_channels))
        in_channels = out_channels
    layers.append(nn.Conv1d(in_channels, 1, kernel_size=kernel_size, padding=padding))  # Final layer to match input channels

    # Ensure the output length matches the original input length
    layers.append(nn.Upsample(size=output_length, mode='linear', align_corners=True))

    return nn.Sequential(*layers)