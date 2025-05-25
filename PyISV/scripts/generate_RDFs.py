# This file generates radial distribution functions (RDFs) from a given XYZ file.

import torch

from PyISV.utils.define_root import PROJECT_ROOT as root_dir
from PyISV.utils.features_calc_utils import calc_rdfs

# Define global constants for testing purposes
params = {
    "xyz_file": f"{root_dir}/datasets/alloy_structures/full_nonMin_ptmd_m18_nCu_2.xyz",
    "output_file": f"{root_dir}/datasets/RDFs/nonMin_nCu_2.pt",
    "min_dist": 1.0,
    "max_dist": 11.0,
    "n_bins": 340,
    "bandwidth": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "species": ["Cu", "Ag"],
    "mode": "multi"
}

if __name__ == "__main__":
    print("Starting RDF computation...")
    try:
        # Run the RDF computation
        rdf_alloy, skipped = calc_rdfs(params)

    except Exception as e:
        print(f"Error during RDF computation: {e}")
    finally:
        if len(skipped) > 0:
            print(f"RDF computation completed with {len(skipped)} skipped structures.")
        print("RDF computation finished.")

