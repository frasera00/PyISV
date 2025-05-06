import numpy as np

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
    embeddings = options.get("embeddings", [])
    outputs = options.get("outputs", [])

    if encoder_only:
        embeddings_batch = model.encoder(x).view(x.size(0), -1).detach().cpu().numpy()
        embeddings.append(embeddings_batch)
        return None, None, None
    reconstructed, bottleneck = model.forward(x)
    target = y if use_separate_target else x
    loss = criterion(reconstructed, target)
    if save_outputs:
        outputs.append(reconstructed.detach().cpu().numpy())
        embeddings.append(bottleneck.detach().cpu().numpy())
    return loss, None, None

def save_validation_outputs(save_outputs, embeddings, outputs):
    if save_outputs and len(embeddings) > 0:
        np.save("embeddings.npy", np.vstack(embeddings))
    if save_outputs and len(outputs) > 0:
        np.save("reconstructed_outputs.npy", np.vstack(outputs))

def assert_shape(tensor, expected_shape, name="Tensor"):
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} shape {tensor.shape} does not match expected {expected_shape}")
