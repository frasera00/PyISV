import torch
import numpy as np

def inspect_data() -> None:
    rdf_files = "data/RDFs/RDFs.npy"

    try:
        if rdf_files.endswith('.npy'):
            rdf_images = np.load(rdf_files)
            print(f"RDF Images shape: {rdf_images.shape}")
        else:
            rdf_images = torch.load(rdf_files)
            print(f"RDF Images shape: {rdf_images.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    inspect_data()