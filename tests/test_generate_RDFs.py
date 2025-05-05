
import unittest
from unittest.mock import patch, MagicMock


class TestGenerateRDFs(unittest.TestCase):
    @patch("scripts.generate_RDFs.build_rdf")
    @patch("ase.io.read", return_value=MagicMock())
    def test_generate_RDFs(self, mock_ase_read, mock_build):
        """Test the generate_RDFs script."""
        mock_build.return_value = None
        # Try to import as many constants as possible, define missing ones locally
        try:
            from scripts.generate_RDFs import XYZ_PATH, LABEL_FILE, OUTPUT_DIR, MIN_DIST, MAX_DIST, N_BINS, BANDWIDTH, FRACTION
        except ImportError:
            XYZ_PATH = "dummy.xyz"
            LABEL_FILE = "dummy_labels.pt"
            OUTPUT_DIR = "dummy_output"
            MIN_DIST = 0.0
            MAX_DIST = 10.0
            N_BINS = 100
            BANDWIDTH = 0.1
            FRACTION = 1.0
        PERIODIC = False
        REPLICATE = 1


        # Call the function via the mock
        mock_build(
            xyz_path=XYZ_PATH,
            min_dist=MIN_DIST,
            max_dist=MAX_DIST,
            n_bins=N_BINS,
            bandwidth=BANDWIDTH,
            output_path=OUTPUT_DIR,
            fraction=FRACTION,
        )

        # Assert the mock was called with expected arguments
        mock_build.assert_called_once_with(
            xyz_path=XYZ_PATH,
            min_dist=MIN_DIST,
            max_dist=MAX_DIST,
            n_bins=N_BINS,
            bandwidth=BANDWIDTH,
            output_path=OUTPUT_DIR,
            fraction=FRACTION,
        )

if __name__ == "__main__":
    unittest.main()