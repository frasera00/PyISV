import torch
import unittest
from unittest.mock import patch
from scripts.analysis.inspect_data import inspect_data

class TestInspectData(unittest.TestCase):
    def mock_torch_load(self, path):
        if "labels.pt" in path:
            return torch.zeros(10, 1)  # Mocked labels
        elif "rdf_images.pt" in path:
            return torch.zeros(10, 200)  # Mocked RDF images
        else:
            raise FileNotFoundError(f"File not found: {path}")

    @patch("torch.load")
    def test_inspect_data(self, mock_load):
        """Test the inspect_data function."""
        mock_load.side_effect = self.mock_torch_load
        try:
            inspect_data()
        except Exception as e:
            self.fail(f"inspect_data raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()