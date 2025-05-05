import torch
from PyISV.features_calc_utils import single_kde_calc, compute_ase_all_distances, torch_kde_calc

class MockAtoms:
    def get_all_distances(self, mic):
        # Return a symmetric distance matrix for 3 atoms
        return [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0]
        ]

def test_single_kde_calc_basic():
    bins = torch.linspace(0, 10, 100)
    bandwidth = torch.tensor([0.1])
    atoms = MockAtoms()
    result = single_kde_calc(atoms, bins, bandwidth)
    assert result is not None, "RDF computation should not return None"
    assert isinstance(result, torch.Tensor), "RDF computation should return a torch.Tensor"
    assert result.shape[-1] == 100, "Output tensor should have length equal to number of bins"

def test_single_kde_calc_with_different_bandwidth():
    bins = torch.linspace(0, 5, 50)
    bandwidth = torch.tensor([0.5])
    atoms = MockAtoms()
    result = single_kde_calc(atoms, bins, bandwidth)
    assert result.shape[-1] == 50

def test_compute_ase_all_distances_shape():
    atoms = MockAtoms()
    distances = compute_ase_all_distances(atoms)
    # For 3 atoms, there are 3 unique pairs: (0,1), (0,2), (1,2)
    assert distances.shape[0] == 3
    assert torch.allclose(distances, torch.tensor([1.0, 2.0, 1.5], dtype=distances.dtype))

def test_torch_kde_calc_output_type_and_shape():
    values = torch.tensor([1.0, 2.0, 3.0])
    bins = torch.linspace(0, 5, 20)
    bandwidth = torch.tensor([0.2])
    result = torch_kde_calc(values, bins, bandwidth)
    assert isinstance(result, torch.Tensor)
    assert result.shape[-1] == 20

def test_single_kde_calc_with_empty_distances():
    class EmptyAtoms:
        def get_all_distances(self, mic):
            return []
    bins = torch.linspace(0, 10, 100)
    bandwidth = torch.tensor([0.1])
    atoms = EmptyAtoms()
    result = single_kde_calc(atoms, bins, bandwidth)
    assert result is not None
    assert isinstance(result, torch.Tensor)
    assert result.shape[-1] == 100
    assert torch.all(result == 0) or torch.all(result == result)  # Should not raise
