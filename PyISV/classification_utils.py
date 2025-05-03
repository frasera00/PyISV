import torch
import torch.nn as nn

def build_classification_head(encoder, input_shape, embed_dim, num_classes, activation_fn, dropout=None):
    """Builds a classification head."""
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape)
        dummy_output = encoder(dummy_input)
        flat_dim = dummy_output.view(1, -1).shape[1]

    return nn.Sequential(
        nn.Linear(flat_dim, embed_dim),
        activation_fn(),
        nn.Dropout(dropout) if dropout else nn.Identity(),
        nn.Linear(embed_dim, num_classes),
    )
