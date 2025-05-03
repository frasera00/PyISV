import unittest
from PyISV.features_calc_utils import single_kde_calc

class TestFeaturesCalcUtils(unittest.TestCase):

    def test_single_kde_calc(self):
        # Example test case for single_kde_calc
        import torch
        bins = torch.linspace(0, 10, 100)
        bandwidth = torch.tensor([0.1])
        # Mock Atoms object or replace with a real one
        class MockAtoms:
            def get_all_distances(self, mic):
                return [[0.5, 1.0], [1.0, 0.5]]
        
        atoms = MockAtoms()
        result = single_kde_calc(atoms, bins, bandwidth)
        self.assertIsNotNone(result, "RDF computation should not return None")
        self.assertIsInstance(result, torch.Tensor, "RDF computation should return a torch.Tensor")

if __name__ == "__main__":
    unittest.main()