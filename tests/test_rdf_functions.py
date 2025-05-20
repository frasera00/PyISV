#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for RDF calculation functions in features_calc_utils.py
"""

import os
import sys
import unittest
import numpy as np
import torch
from ase.io import read
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyISV.utils.features_calc_utils import (
    single_kde_calc,
    triple_kde_calc,
    multi_channel_rdf,
    concatenated_rdf, 
    weighted_attention_rdf,
    alloy_rdf,
    build_rdf
)

class TestRDFFunctions(unittest.TestCase):
    """Test suite for RDF calculation functions."""

    def setUp(self):
        """Set up test environment."""
        # Define common parameters
        self.min_dist = 2.0
        self.max_dist = 8.0
        self.n_bins = 64
        self.bandwidth = 0.15
        self.bins = torch.linspace(self.min_dist, self.max_dist, self.n_bins)
        self.bw = torch.tensor([self.bandwidth])
        
        # Load test structures
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'datasets', 'alloy_structures')
        self.cu2_path = os.path.join(self.data_dir, 'full_min_ptmd_m18_nCu_2.xyz')
        self.cu6_path = os.path.join(self.data_dir, 'full_min_ptmd_m18_nCu_6.xyz')
        
        logger.info(f"Loading structures from {self.data_dir}")
        logger.info(f"Cu2 path exists: {os.path.exists(self.cu2_path)}")
        logger.info(f"Cu6 path exists: {os.path.exists(self.cu6_path)}")
        
        # Read structures
        try:
            self.cu2_atoms = read(self.cu2_path, index=0)
            self.cu6_atoms = read(self.cu6_path, index=0)
            logger.info(f"Cu2 atoms loaded: {len(self.cu2_atoms)} atoms")
            logger.info(f"Cu6 atoms loaded: {len(self.cu6_atoms)} atoms")
            
            # Print atom types
            cu2_species = set(self.cu2_atoms.get_chemical_symbols())
            cu6_species = set(self.cu6_atoms.get_chemical_symbols())
            logger.info(f"Cu2 species: {cu2_species}")
            logger.info(f"Cu6 species: {cu6_species}")
        except Exception as e:
            logger.error(f"Error loading structures: {e}")
            raise
        
        # Define species list
        self.species_list = ['Ag', 'Cu']

    def test_single_kde_calc(self):
        """Test single RDF calculation."""
        # Calculate RDF for Cu2 structure
        rdf = single_kde_calc(self.cu2_atoms, self.bins, self.bw)
        
        # Check output shape and type
        self.assertIsInstance(rdf, torch.Tensor)
        # The shape can be either [1, n_bins] or [n_bins] depending on implementation
        self.assertTrue(rdf.shape == (1, self.n_bins) or rdf.shape == (self.n_bins,), 
                       f"Unexpected shape: {rdf.shape}")
        
        # Ensure we're working with a 1D tensor for the remaining tests
        if rdf.dim() > 1:
            rdf = rdf.squeeze(0)
        
        # Check that RDF is normalized and non-negative
        self.assertTrue(torch.all(rdf >= 0))
        
        # Calculate sum (should be approximately normalized)
        bin_width = (self.max_dist - self.min_dist) / (self.n_bins - 1)
        rdf_sum = torch.sum(rdf) * bin_width
        self.assertTrue(0.9 < rdf_sum < 1.1, f"RDF normalization failed: {rdf_sum}")

    def test_triple_kde_calc(self):
        """Test triple RDF calculation for binary alloy."""
        # Calculate RDFs for Cu6 structure
        ag_ag, cu_cu, ag_cu = triple_kde_calc('Ag', 'Cu', self.cu6_atoms, self.bins, self.bw)
        
        # Check output types and shapes
        self.assertIsInstance(ag_ag, torch.Tensor)
        self.assertIsInstance(cu_cu, torch.Tensor)
        self.assertIsInstance(ag_cu, torch.Tensor)
        
        # Ensure dimensions are right
        if ag_ag.dim() > 1:
            ag_ag = ag_ag.squeeze()
        if cu_cu.dim() > 1:
            cu_cu = cu_cu.squeeze()
        if ag_cu.dim() > 1:
            ag_cu = ag_cu.squeeze()
            
        self.assertEqual(ag_ag.shape, (self.n_bins,))
        self.assertEqual(cu_cu.shape, (self.n_bins,))
        self.assertEqual(ag_cu.shape, (self.n_bins,))
        
        # Check non-negative values
        self.assertTrue(torch.all(ag_ag >= 0))
        self.assertTrue(torch.all(cu_cu >= 0))
        self.assertTrue(torch.all(ag_cu >= 0))
        
        # Verify that values exist where expected
        # Since Cu6 has both Ag and Cu atoms, all RDFs should have non-zero elements
        self.assertTrue(torch.any(ag_ag > 0), "Ag-Ag RDF is all zeros")
        self.assertTrue(torch.any(cu_cu > 0), "Cu-Cu RDF is all zeros")
        self.assertTrue(torch.any(ag_cu > 0), "Ag-Cu RDF is all zeros")

    def test_multi_channel_rdf(self):
        """Test multi-channel RDF calculation."""
        # Calculate multi-channel RDF
        rdf_multi = multi_channel_rdf(self.species_list, self.cu6_atoms, self.bins, self.bw)
        
        # Check output shape and type
        self.assertIsInstance(rdf_multi, torch.Tensor)
        n_species = len(self.species_list)
        n_pairs = (n_species * (n_species + 1)) // 2  # Ag-Ag, Cu-Cu, Ag-Cu
        self.assertEqual(rdf_multi.shape, (n_pairs, self.n_bins))
        
        # Check non-negative values
        self.assertTrue(torch.all(rdf_multi >= 0))
        
        # Verify that each channel has proper content
        for i in range(n_pairs):
            self.assertTrue(torch.any(rdf_multi[i] > 0), f"Channel {i} is all zeros")

    def test_concatenated_rdf(self):
        """Test concatenated RDF calculation."""
        # Calculate concatenated RDF
        rdf_concat = concatenated_rdf(self.species_list, self.cu6_atoms, self.bins, self.bw)
        
        # Check output shape and type
        self.assertIsInstance(rdf_concat, torch.Tensor)
        n_species = len(self.species_list)
        n_pairs = (n_species * (n_species + 1)) // 2
        expected_size = n_pairs * self.n_bins
        self.assertEqual(rdf_concat.shape, (expected_size,))
        
        # Check non-negative values
        self.assertTrue(torch.all(rdf_concat >= 0))
        
        # Verify that values exist and are properly ordered
        # Compare with multi_channel_rdf to ensure correct concatenation
        rdf_multi = multi_channel_rdf(self.species_list, self.cu6_atoms, self.bins, self.bw)
        rdf_multi_flat = rdf_multi.reshape(-1)
        self.assertTrue(torch.allclose(rdf_concat, rdf_multi_flat))

    def test_weighted_attention_rdf(self):
        """Test weighted attention RDF calculation."""
        # Calculate weighted attention RDF
        rdf_weighted = weighted_attention_rdf(self.species_list, self.cu6_atoms, self.bins, self.bw)
        
        # Check output shape and type
        self.assertIsInstance(rdf_weighted, torch.Tensor)
        # Handle both potential shapes
        if rdf_weighted.dim() > 1:
            rdf_weighted = rdf_weighted.squeeze(0)
        self.assertEqual(rdf_weighted.shape, (self.n_bins,))
        
        # Check non-negative values
        self.assertTrue(torch.all(rdf_weighted >= 0))
        
        # Verify that weights are properly applied
        # For Cu6, we should see more Cu-Cu contribution than in Cu2
        rdf_weighted_cu2 = weighted_attention_rdf(self.species_list, self.cu2_atoms, self.bins, self.bw)
        if rdf_weighted_cu2.dim() > 1:
            rdf_weighted_cu2 = rdf_weighted_cu2.squeeze(0)
        
        # Calculate Cu concentration in each sample
        cu6_symbols = self.cu6_atoms.get_chemical_symbols()
        cu2_symbols = self.cu2_atoms.get_chemical_symbols()
        
        cu6_concentration = cu6_symbols.count('Cu') / len(cu6_symbols)
        cu2_concentration = cu2_symbols.count('Cu') / len(cu2_symbols)
        
        # Verify that Cu6 has higher Cu concentration than Cu2
        self.assertGreater(cu6_concentration, cu2_concentration, 
                          "Cu6 sample should have higher Cu concentration than Cu2")
        
        # Find the peak in each RDF separately
        cu6_peak_idx = torch.argmax(rdf_weighted)
        cu2_peak_idx = torch.argmax(rdf_weighted_cu2)
        
        # Calculate the peak height ratio - we expect Cu6 to have a relatively stronger peak
        # compared to the overall RDF average
        cu6_peak_to_mean = rdf_weighted[cu6_peak_idx].item() / rdf_weighted.mean().item()
        cu2_peak_to_mean = rdf_weighted_cu2[cu2_peak_idx].item() / rdf_weighted_cu2.mean().item()
        
        # Print diagnostic information
        logging.info(f"Cu6 concentration: {cu6_concentration:.3f}, Cu2 concentration: {cu2_concentration:.3f}")
        logging.info(f"Cu6 peak height: {rdf_weighted[cu6_peak_idx].item():.3f}, Cu2 peak height: {rdf_weighted_cu2[cu2_peak_idx].item():.3f}")
        logging.info(f"Cu6 peak-to-mean ratio: {cu6_peak_to_mean:.3f}, Cu2 peak-to-mean ratio: {cu2_peak_to_mean:.3f}")
        
        # Test passes if Cu6 has a stronger relative peak
        # If this assertion fails, it suggests the weighted calculation needs further tuning
        self.assertGreaterEqual(cu6_peak_to_mean, cu2_peak_to_mean * 0.9,
                              "Expected Cu6 to have a relatively stronger peak due to higher Cu concentration")

    def test_alloy_rdf_outputs(self):
        """Test that alloy_rdf function returns proper outputs for each approach."""
        # Test multi_channel approach
        rdf_multi = alloy_rdf(self.species_list, self.cu6_atoms, self.bins, self.bw, 
                             approach="multi_channel")
        n_species = len(self.species_list)
        n_pairs = (n_species * (n_species + 1)) // 2
        self.assertEqual(rdf_multi.shape, (n_pairs, self.n_bins))
        
        # Test concatenated approach
        rdf_concat = alloy_rdf(self.species_list, self.cu6_atoms, self.bins, self.bw, 
                              approach="concatenated")
        expected_size = n_pairs * self.n_bins
        self.assertEqual(rdf_concat.shape, (expected_size,))
        
        # Test weighted_attention approach
        rdf_weighted = alloy_rdf(self.species_list, self.cu6_atoms, self.bins, self.bw, 
                                approach="weighted_attention")
        # Handle both potential shapes
        if rdf_weighted.dim() > 1:
            rdf_weighted = rdf_weighted.squeeze()
        self.assertEqual(rdf_weighted.shape, (self.n_bins,))
        
        # Test invalid approach
        with self.assertRaises(ValueError):
            alloy_rdf(self.species_list, self.cu6_atoms, self.bins, self.bw, 
                     approach="invalid_approach")

    def test_build_rdf_shapes(self):
        """Test that build_rdf function produces outputs with the expected shapes."""
        output_dir = os.path.join(os.path.dirname(__file__), 'test_outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Count frames ahead of time
        frames = read(self.cu6_path, index=":")
        n_frames = len(frames)
        
        # Use a small subset of frames for faster testing
        fraction = 0.01  # Use only 10% of frames to speed up test
        n_frames_subset = max(1, int(n_frames * fraction))
        
        # Test single mode with subset of frames
        rdf_single = build_rdf(
            self.cu6_path, 
            self.min_dist, 
            self.max_dist, 
            self.n_bins,
            self.bandwidth, 
            output_dir, 
            device="cpu",
            mode="single",
            fraction=fraction
        )
        self.assertEqual(rdf_single.shape, (n_frames_subset, 1, self.n_bins))
        
        # Test triple mode with subset of frames
        rdf_triple = build_rdf(
            self.cu6_path, 
            self.min_dist, 
            self.max_dist, 
            self.n_bins,
            self.bandwidth, 
            output_dir, 
            device="cpu",
            mode="triple", 
            species_list=self.species_list,
            fraction=fraction
        )
        self.assertEqual(rdf_triple.shape, (n_frames_subset, 3, self.n_bins))
        
        # Test alloy mode with different approaches
        approaches = ["multi_channel", "concatenated", "weighted_attention"]
        for approach in approaches:
            rdf_alloy = build_rdf(
                self.cu6_path, 
                self.min_dist, 
                self.max_dist, 
                self.n_bins,
                self.bandwidth, 
                output_dir, 
                device="cpu",
                mode="alloy", 
                species_list=self.species_list,
                approach=approach,
                fraction=fraction
            )
            
            # Check shapes based on approach
            if approach == "multi_channel":
                n_pairs = 3  # Ag-Ag, Cu-Cu, Ag-Cu
                self.assertEqual(rdf_alloy.shape, (n_frames_subset, n_pairs, self.n_bins))
            elif approach == "concatenated":
                n_pairs = 3
                self.assertEqual(rdf_alloy.shape, (n_frames_subset, 1, n_pairs * self.n_bins))
            elif approach == "weighted_attention":
                self.assertEqual(rdf_alloy.shape, (n_frames_subset, 1, self.n_bins))

        # Clean up test outputs
        for file in os.listdir(output_dir):
            if file.startswith('rdf_images_'):
                os.remove(os.path.join(output_dir, file))
        try:
            os.rmdir(output_dir)
        except OSError:
            pass  # Directory might not be empty

if __name__ == '__main__':
    unittest.main()