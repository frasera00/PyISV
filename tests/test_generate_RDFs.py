import pytest
from unittest.mock import patch, MagicMock
from scripts.generate_RDFs import build_rdf

def test_generate_RDFs():
    """Test the generate_RDFs script."""
    with patch("scripts.generate_RDFs.build_rdf") as mock_build, \
         patch("ase.io.read", return_value=MagicMock()):  # Mock file reading
        # Mock the function to ensure it is called with expected arguments
        mock_build.return_value = None

        # Call the script's main function
        try:
            from scripts.generate_RDFs import XYZ_PATH, LABEL_FILE, OUTPUT_DIR, MIN_DIST, MAX_DIST, N_BINS, BANDWIDTH, FRACTION, PERIODIC, REPLICATE
            build_rdf(
                xyz_path=XYZ_PATH,
                label_file=LABEL_FILE,
                min_dist=MIN_DIST,
                max_dist=MAX_DIST,
                n_bins=N_BINS,
                bandwidth=BANDWIDTH,
                output_path=OUTPUT_DIR,
                fraction=FRACTION,
                periodic=PERIODIC,
                replicate=REPLICATE,
            )
        except Exception as e:
            pytest.fail(f"generate_RDFs script raised an exception: {e}")

        # Assert the mock was called with expected arguments
        mock_build.assert_called_once_with(
            xyz_path=XYZ_PATH,
            label_file=LABEL_FILE,
            min_dist=MIN_DIST,
            max_dist=MAX_DIST,
            n_bins=N_BINS,
            bandwidth=BANDWIDTH,
            output_path=OUTPUT_DIR,
            fraction=FRACTION,
            periodic=PERIODIC,
            replicate=REPLICATE,
        )