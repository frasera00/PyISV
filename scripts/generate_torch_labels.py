import torch
import os
import numpy as np
from PyISV import DATA_DIR as data_dir, OUTPUTS_DIR as outputs_dir

LABEL_FILE = f"{data_dir}/Ag38_labels/combined_isv_labels_2D_nonMin_to_min_k15_nCu_0.dat"
OUTPUT_DIR = f"{outputs_dir}/RDFs"

labels = torch.LongTensor(np.loadtxt(LABEL_FILE, dtype=int))
label_path = f"{outputs_dir}/labels.pt"
torch.save(labels, label_path)

