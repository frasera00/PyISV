# set_architecture.py
# Utility to load a JSON config and produce a Python config dict with nn.Module objects for PyISV

import json
import torch.nn as nn
import inspect
from pathlib import Path

class InputsReader():
    """Class to read and validate inputs from a JSON config file."""
    def __init__(self, json_path: str | Path) -> None:
        self.json_path = json_path
        self.params = self._json_to_arch_params(json_path)  

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

    def _json_to_arch_params(self, json_path: str | Path) -> dict: 
        """ Convert a JSON config file into the params structure with instantiated nn.Module objects."""
        with open(json_path, "r") as f:
            config = json.load(f)
        
        # Extract parameters
        in_bins = config["GENERAL"]["input_length"]
        in_channels = config["GENERAL"]["input_channels"]
        embedding_dim = config["MODEL"]["embedding_dim"]
        feature_map_length = config["MODEL"]["feature_map_length"]
        
        # Calculate num_pooling_layers from the encoder structure
        num_pooling_layers = len(config["MODEL"]["encoder_layers"])
        
        # Get channels from the last encoder layer
        last_enc_layer = config["MODEL"]["encoder_layers"][-1]
        for layer in last_enc_layer:
            if layer["type"] == "Conv1d":
                last_in_channels = layer["out_channels"]
                break
        
        # Create Python-based architecture like the original
        # Build nn.Module objects for each layer
        encoder_layers = {}
        for i, block in enumerate(config["MODEL"]["encoder_layers"]):
            encoder_layers[i] = self._build_block(block)
        
        bottleneck_layers = {}
        for i, block in enumerate(config["MODEL"]["bottleneck_layers"]):
            bottleneck_layers[i] = self._build_block(block)

        decoder_layers = {}
        for i, block in enumerate(config["MODEL"]["decoder_layers"]):
            decoder_layers[i] = self._build_block(block)

        # Get loss function
        loss_function = self._get_nn_class(config["TRAINING"]["loss_function"])()
        
        # Construct params dictionary like in architecture.py
        params = {
            "GENERAL": config["GENERAL"],
            "MODEL": {
                "type": config["MODEL"]["type"],
                "input_shape": config["MODEL"]["input_shape"],
                "embedding_dim": embedding_dim,
                "flattened_dim": config["MODEL"]["flattened_dim"],
                "feature_map_length": feature_map_length,
                "encoder_layers": encoder_layers,
                "bottleneck_layers": bottleneck_layers,
                "decoder_layers": decoder_layers
            },
            "TRAINING": config["TRAINING"].copy(),
            "LEARNING": config["LEARNING"],
            "INPUTS": config["INPUTS"]
        }
        
        # Replace loss function string with actual object
        params["TRAINING"]["loss_function"] = loss_function
        return params

# Helper function to make it easier to load architecture params
def import_config(json_path: str | Path) -> dict:
    """Load architecture parameters from a JSON config file."""
    reader = InputsReader(json_path)
    return reader()


