import torch
import os
import numpy as np

LABEL_FILE = "./data/Ag38_labels/combined_isv_labels_2D_nonMin_to_min_k15_nCu_0.txt"
OUTPUT_DIR = "./data/RDFs"

# Generate labels torch file 
label_file = LABEL_FILE
output_path=OUTPUT_DIR

labels = torch.LongTensor(np.loadtxt(label_file, dtype=int))
label_path = os.path.join(output_path, "labels.pt")
torch.save(labels, label_path)

