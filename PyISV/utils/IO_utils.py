
import re, os, logging, json, inspect
import numpy as np

from pathlib import Path
from typing import Optional, Union
from PyISV.utils import training_utils

import torch
import torch.nn as nn

class InputsReader():
    """Class to read and validate inputs from a JSON config file."""
    def __init__(self, param_dict: dict) -> None:
        self.params = self._dict_to_torch(input_dict=param_dict)

    def __call__(self) -> dict:
        """Return the params dictionary when the class instance is called."""
        return self.params

    def _get_nn_class(self, name: str) -> type:
        """Get any nn.Module class by name, with validation from torch.nn or train_utils."""
        cls = None
        
        # First try torch.nn
        if hasattr(nn, name):
            cls = getattr(nn, name)
        # Then try training_utils for custom layers
        elif hasattr(training_utils, name):
            cls = getattr(training_utils, name)
        else:
            raise ValueError(f"Unknown layer type: {name}. Not found in torch.nn or train_utils")
        
        # Verify it's a class and a subclass of nn.Module
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
        loss_fn_name = input_dict["TRAINING"]["loss_function"]
        loss_params = input_dict["TRAINING"]["loss_params"]
        loss_function = self._get_nn_class(loss_fn_name)(**loss_params)

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
def import_config(json_file: Optional[Union[str, Path]],
                  param_dict: Optional[dict]) -> dict:
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

def setup_logging(log_file: Optional[str] = None, 
                  log_level: int = logging.INFO, 
                  log_format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
    """Set up logging configuration. If log_file is provided, logs will be written to that file.
    Removes all existing handlers to ensure this config takes effect."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename="train.log" if log_file is None else log_file,
        level=log_level,
        format=log_format
    )

def is_main_process() -> bool:
    # For torch.distributed, rank 0 is the main process
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # For SLURM or other multi-process setups, check environment variables
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"]) == 0
    # Fallback: single process
    return True

def load_tensor(file_path: str) -> torch.Tensor:
    if file_path.endswith('.npy'):
        arr = np.load(file_path)
        tensor = torch.from_numpy(arr).float()
    else:
        tensor = torch.load(file_path).float()
    # Only unsqueeze if 2D
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    return tensor

def find_project_root(current: Optional[Path] = None, markers: Optional[list[str]] = None) -> Path:
    """Find project root by checking for multiple possible markers."""
    if current is None:
        current = Path(__file__).parent
        
    if markers is None:
        markers = [
            "pyproject.toml",   # Modern Python projects
            ".git",            # Git repository  
            "setup.py",         # Traditional Python projects
            "setup.cfg",        # Your current marker
            "requirements.txt", # Common in Python projects
            ".gitignore"       # Almost always at project root
        ]
    
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    raise RuntimeError(f"Project root not found. Searched for markers: {markers}")

class RegexFilter(logging.Filter):
    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = re.compile(pattern)
    def filter(self, record: logging.LogRecord) -> bool:
        return not self.pattern.search(record.getMessage())
