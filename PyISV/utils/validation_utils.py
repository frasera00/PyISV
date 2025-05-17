# Validation utilities for PyISV
# This module contains functions to validate models, including autoencoders and classifiers.
# It also includes functions to normalize inputs, validate epochs, and save validation outputs.

import numpy as np
import torch
from typing import Optional

class Validator:
    """Validation class for PyISV models."""
    def __init__(self, config: dict) -> None:
        self.config = config

    def validate_epoch(self,
            model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            loss_function: torch.nn.Module,
            device: torch.device,
            emb_file: Optional[str] = None,
            out_file: Optional[str] = None,
        ) -> float:
        """Run validation for a single epoch."""
        model.eval()
        val_loss = 0
        num_batches = 0
        embeddings = []
        outputs = []
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    recon, embed = model(x)
                    loss = loss_function(recon, x)
                val_loss += loss.item()
                num_batches += 1
                embeddings.append(embed.detach().cpu())
                outputs.append(recon.detach().cpu())

        # At the end, concatenate to get a single tensor for each
        embeddings = torch.cat(embeddings, dim=0)
        outputs = torch.cat(outputs, dim=0)
        
        # Move all tensors to CPU before converting to numpy
        self.save_validation_outputs(
            emb_file=emb_file,
            out_file=out_file,
            embeddings=embeddings,
            outputs=outputs
        )
        return val_loss / num_batches if num_batches > 0 else 0

    def save_validation_outputs(self,
            emb_file: Optional[str] = None,
            out_file: Optional[str] = None,
            embeddings: Optional[torch.Tensor] = None,
            outputs: Optional[torch.Tensor] = None
        ) -> None:
        """Save embeddings and outputs to .pt files."""
        if emb_file and embeddings is not None and len(embeddings) > 0:
            torch.save(embeddings, emb_file)
        if out_file and outputs is not None and len(outputs) > 0:
            torch.save(outputs, out_file)

    def assert_shape(self,
            tensor: torch.Tensor, expected_shape: tuple, name: str = "Tensor"
        ) -> None:
        """Assert that a tensor or numpy array has the expected shape."""
        # Allow silent pass if tracing (for use in model forward)
        if torch.jit.is_tracing(): # type: ignore[return-value]
            return
        if isinstance(tensor, np.ndarray):
            shape = tensor.shape
        else:
            # For torch.Tensor, get shape as tuple
            shape = tuple(tensor.shape)
        if shape != expected_shape:
            raise ValueError(f"{name} shape {shape} does not match expected {expected_shape}")