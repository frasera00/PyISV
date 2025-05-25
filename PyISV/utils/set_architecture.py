# set_architecture.py
# Utility to load a JSON config and produce a Python config dict with nn.Module objects for PyISV

import json
import torch.nn as nn
import inspect
from pathlib import Path
from typing import Optional

class InputsReader():
    """Class to read and validate inputs from a JSON config file."""
    def __init__(self, param_dict: dict) -> None:
        self.params = self._dict_to_torch(input_dict=param_dict)

    def __call__(self) -> dict:
        """Return the params dictionary when the class instance is called."""
        return self.params

    def _get_nn_class(self, name: str) -> type:
        """Get any nn.Module class by name, with validation."""
        # Try to get the class from nn module
        if not hasattr(nn, name):
            raise ValueError(f"Unknown layer type: {name}. Not found in torch.nn")
        
        # Verify it's a class and a subclass of nn.Module
        cls = getattr(nn, name)
        if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
            raise ValueError(f"{name} is not a proper nn.Module class")
        return cls


    def _build_layer(self, layer_dict: dict) -> nn.Module:
        """Instantiate a torch.nn layer from a dict spec, fully generic."""
        layer_type = layer_dict["type"]
        layer_cls = self._get_nn_class(layer_type)

        # Try to instantiate with kwargs, fallback to no-arg if fails
        kwargs = {k: v for k, v in layer_dict.items() if k != "type"}
        if "padding" in kwargs and kwargs["padding"] == "same":
            kwargs["padding"] = "same"  # Keep as string, torch.nn supports "same" padding
        
        try:
            return layer_cls(**kwargs)
        except TypeError:
            return layer_cls()

    def _build_layers(self, layer_list: list[dict]) -> nn.ModuleList:
        """Build a list of nn.Module layers from a list of dicts."""
        return nn.ModuleList([self._build_layer(layer) for layer in layer_list])

    def _build_block(self, block: list[dict]) -> nn.Sequential:
        """Build a block (list of layers) from a list of dicts."""
        return nn.Sequential(*self._build_layers(block))
    
    def _dict_to_torch(self, input_dict: dict) -> dict: 
        """Convert a config dict into the params structure with instantiated nn.Module objects.
        Assumes encoder_layers, bottleneck_layers, decoder_layers are all dicts of blocks (after normalization)."""
        # Extract parameters
        in_bins = input_dict["GENERAL"]["input_length"]
        in_channels = input_dict["GENERAL"]["input_channels"]
        embedding_dim = input_dict["MODEL"]["embedding_dim"]
        feature_map_length = input_dict["MODEL"]["feature_map_length"]

        # Calculate num_pooling_layers from the encoder structure
        encoder_layers = input_dict["MODEL"]["encoder_layers"]
        num_pooling_layers = len(encoder_layers)

        # Build nn.Module objects for each layer (assume dict format)
        encoder_modules = {i: self._build_block(block) for i, block in input_dict["MODEL"]["encoder_layers"].items()}
        bottleneck_modules = {i: self._build_block(block) for i, block in input_dict["MODEL"]["bottleneck_layers"].items()}
        decoder_modules = {i: self._build_block(block) for i, block in input_dict["MODEL"]["decoder_layers"].items()}

        # Get loss function
        loss_function = self._get_nn_class(input_dict["TRAINING"]["loss_function"])()

        # Construct params dictionary like in architecture.py
        params = {
            "GENERAL": input_dict["GENERAL"],
            "MODEL": {
                "type": input_dict["MODEL"]["type"],
                "input_shape": input_dict["MODEL"]["input_shape"],
                "embedding_dim": embedding_dim,
                "flattened_dim": input_dict["MODEL"]["flattened_dim"],
                "feature_map_length": feature_map_length,
                "encoder_layers": encoder_modules,
                "bottleneck_layers": bottleneck_modules,
                "decoder_layers": decoder_modules
            },
            "TRAINING": input_dict["TRAINING"],
            "INPUTS": input_dict["INPUTS"]
        }

        # Replace loss function string with actual object
        params["TRAINING"]["loss_function"] = loss_function
        return params

# Helper function to make it easier to load architecture params
def import_config(json_file: str | Path | None = None, param_dict: dict | None = None) -> dict:
    """Load architecture parameters from a JSON config file or dict, always normalizing layer configs to dicts."""
    if param_dict is not None:
        param_dict = convert_layer_lists_to_dicts(param_dict)
        return InputsReader(param_dict=param_dict)()
    elif json_file is not None:
        with open(json_file, "r") as f:
            params = json.load(f)
        param_dict = convert_layer_lists_to_dicts(params)
        return InputsReader(param_dict=param_dict)()
    else:
        raise ValueError("Either param_dict or json_file must be provided.")
    
def convert_layer_lists_to_dicts(params: dict) -> dict:
    """Convert layer configuration lists to dictionaries for model_building.py compatibility."""
    for key in ["encoder_layers", "bottleneck_layers", "decoder_layers"]:
        layers = params["MODEL"].get(key)
        if isinstance(layers, list):
            params["MODEL"][key] = {i: block for i, block in enumerate(layers)}
    return params


