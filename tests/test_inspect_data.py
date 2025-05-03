import pytest
import torch
from scripts.inspect_data import inspect_data

def test_inspect_data(monkeypatch):
    """Test the inspect_data function."""
    # Mock torch.load to return dummy data
    def mock_torch_load(path):
        if "labels.pt" in path:
            return torch.zeros(10, 1)  # Mocked labels
        elif "rdf_images.pt" in path:
            return torch.zeros(10, 200)  # Mocked RDF images
        else:
            raise FileNotFoundError(f"File not found: {path}")

    monkeypatch.setattr(torch, "load", mock_torch_load)

    # Run the inspect_data function
    try:
        inspect_data()
    except Exception as e:
        pytest.fail(f"inspect_data raised an exception: {e}")