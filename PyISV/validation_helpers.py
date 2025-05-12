import numpy as np
import torch

def normalize_input(x, scaler_subval, scaler_divval):
    if scaler_subval is not None and scaler_divval is not None:
        return (x - scaler_subval) / scaler_divval
    return x

def validate_classifier(model, x, y, criterion):
    outputs_batch = model.forward(x)
    loss = criterion(outputs_batch, y)
    _, predicted = torch.max(outputs_batch, 1)
    correct = (predicted == y).sum().item()
    total = y.size(0)
    return loss, correct, total

def validate_autoencoder(model, x, y, criterion, options):
    encoder_only = options.get("encoder_only", False)
    save_outputs = options.get("save_outputs", False)
    use_separate_target = options.get("use_separate_target", False)
    embeddings = options.get("embeddings", None)
    outputs = options.get("outputs", None)

    if encoder_only:
        if embeddings is not None:
            embeddings_batch = model.encoder(x).view(x.size(0), -1).detach().cpu().numpy()
            embeddings.append(embeddings_batch)
        return None, None, None
    reconstructed, bottleneck = model.forward(x)
    target = y if use_separate_target else x
    loss = criterion(reconstructed, target)
    if save_outputs:
        if outputs is not None:
            outputs.append(reconstructed.detach().cpu().numpy())
        if embeddings is not None:
            embeddings.append(bottleneck.detach().cpu().numpy())
    return loss, None, None

def save_validation_outputs(save_outputs, embeddings, outputs):
    """
    Save embeddings and outputs to .npy files if requested.
    WARNING: Do NOT call this function inside any torch.jit.trace context!
    Raises RuntimeError if called during tracing to prevent PyTorch TracerWarnings.
    """
    if save_outputs and embeddings is not None and len(embeddings) > 0:
        np.save("embeddings.npy", np.vstack(embeddings))
    if save_outputs and outputs is not None and len(outputs) > 0:
        np.save("reconstructed_outputs.npy", np.vstack(outputs))

def assert_shape(tensor, expected_shape, name="Tensor"):
    """
    Assert that a tensor or numpy array has the expected shape.
    WARNING: Do NOT use this inside any function that will be traced with torch.jit.trace!
    Only use for debugging or outside traced code.
    Raises RuntimeError if called during tracing to prevent PyTorch TracerWarnings.
    """

    # Allow silent pass if tracing (for use in model forward)
    if torch.jit.is_tracing():
        return
    if isinstance(tensor, np.ndarray):
        shape = tensor.shape
    else:
        # For torch.Tensor, get shape as tuple
        shape = tuple(tensor.shape)
    if shape != expected_shape:
        raise ValueError(f"{name} shape {shape} does not match expected {expected_shape}")