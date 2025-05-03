import torch

def inspect_data():
    labels_path = "data/RDFs/labels.pt"
    rdf_images_path = "data/RDFs/rdf_images.pt"

    try:
        labels = torch.load(labels_path)
        rdf_images = torch.load(rdf_images_path)

        print(f"Labels shape: {labels.shape}")
        print(f"RDF Images shape: {rdf_images.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    inspect_data()