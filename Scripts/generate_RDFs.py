#!/usr/bin/env python3
import os
import logging  # Import the logging module

from PyISV.features_calc_utils import compute_rdfs_and_save


def setup_logging(output_path):
    """
    Setup the logging configuration.
    """
    log_file = os.path.join(output_path, "rdf_computation.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return log_file


if __name__ == "__main__":
    # — user config —
    XYZ_PATH    = "./structures/full_min_ptmd_m18_nCu_0.xyz"
    LABEL_FILE  = "./Ag38_labels/combined_isv_labels_2D_nonMin_to_min_k15_nCu_0.txt"
    OUTPUT_DIR  = "./RDFs"

    MIN_DIST    = 1.0       # Minimum distance for RDF calculation (in Angstroms)
    MAX_DIST    = 8.0       # Maximum distance for RDF calculation (in Angstroms)
    N_BINS      = 200       # Number of bins for the histogram
    BANDWIDTH   = 0.2       # Bandwidth for KDE (in Angstroms)

    FRACTION    = 1.0       # Fraction of frames to use for RDF calculation (1.0 = all frames)
    PERIODIC    = False     # Set to True for periodic boundary conditions
    REPLICATE   = (2, 2, 2) # Optional, only used if periodic=True
    
    # Set up logging
    log_file = setup_logging(OUTPUT_DIR)

    # Run the RDF computation
    compute_rdfs_and_save(
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

    print(f"Log saved to: {log_file}")
