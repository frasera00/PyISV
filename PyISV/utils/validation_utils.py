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
        ) -> float:
        """Run validation for a single epoch."""
        model.eval()
        val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for inp, targ in data_loader:
                inp = inp.to(device)
                targ = targ.to(device)
                
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    output = model(inp) 
                    loss = loss_function(output, targ)
                val_loss += loss.item()
                num_batches += 1

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