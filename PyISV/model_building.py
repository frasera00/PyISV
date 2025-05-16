# Model building algorithms for PyISV

import torch.nn as nn

def build_encoder(encoder_layers_dict: dict) -> nn.Sequential:
    """Builds an encoder from an ordered dict of lists of nn.Module objects."""
    layers = []
    for key in sorted(encoder_layers_dict.keys()):
        layers.extend(encoder_layers_dict[key])
    return nn.Sequential(*layers)

def build_bottleneck(bottleneck_layers_dict: dict) -> tuple[nn.Sequential, nn.Sequential]:
    """Builds a bottleneck from an ordered dict of lists of nn.Module objects."""
    embed_layers = bottleneck_layers_dict[0]  # encoder to embedding
    decode_layers = bottleneck_layers_dict[1] # embedding to expanded
    return nn.Sequential(*embed_layers), nn.Sequential(*decode_layers)

def build_decoder(decoder_layers_dict: dict) -> nn.Sequential:
    """Builds a decoder from an ordered dict of lists of nn.Module objects."""
    layers = []
    for key in sorted(decoder_layers_dict.keys()):
        layers.extend(decoder_layers_dict[key])
    return nn.Sequential(*layers)