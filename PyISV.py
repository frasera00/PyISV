# main.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from ase import io
import matplotlib.pyplot as plt

from PyISV.network_2D_classification import Classifier2D
from PyISV.train_utils_classification import ClassificationTrainer
from PyISV.train_utils_classification import MultiFrameRDFImageDataset


# —————————————————————————————————————————————
# 1) Configuration (paths, hyperparameters, etc.)
# —————————————————————————————————————————————

XYZ_PATH   = "/Users/frasera/Ricerca/PyISV/structures/full_min_ptmd_m18_nCu_0.xyz"
LABEL_FILE = "Ag38_labels/isv_labels_2D_nonMin_to_min_k15_nCu_0.txt"

N_BINS     = 32
R_MAX      = 8.0
BANDWIDTH  = 0.15

NUM_CLASSES        = 6
NUM_FINAL_CHANNELS = 8
EMBED_DIM          = 64
LR                 = 1e-3
WEIGHT_DECAY       = 0
BATCH_SIZE         = 32
NUM_EPOCHS         = 5
VAL_SPLIT          = 0.3
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# —————————————————————————————————————————————
# 2) Prepare bins & bandwidth tensors
# —————————————————————————————————————————————
bins_tensor      = torch.linspace(0.0, R_MAX, steps=N_BINS)
bandwidth_tensor = torch.full((N_BINS,), BANDWIDTH)

# —————————————————————————————————————————————
# 3) Read labels
# —————————————————————————————————————————————
labels = torch.LongTensor(np.loadtxt(LABEL_FILE, dtype=int))
assert labels.ndim == 1, "Labels must be a 1D array"

# —————————————————————————————————————————————
# 4) Multi‐frame dataset for single XYZ file
# —————————————————————————————————————————————

# instantiate
full_dataset = MultiFrameRDFImageDataset(XYZ_PATH, labels, bins_tensor, bandwidth_tensor)

# —————————————————————————————————————————————
# 5) Split & loaders
# —————————————————————————————————————————————
indices = list(range(0, len(full_dataset), 10))
work_dataset = Subset(full_dataset, indices)

val_size   = int(VAL_SPLIT * len(work_dataset))
train_size = len(work_dataset) - val_size
train_ds, val_ds = random_split(work_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

# —————————————————————————————————————————————
# 6) Model & trainer
# —————————————————————————————————————————————
spatial_size = N_BINS // 4
flat_dim     = NUM_FINAL_CHANNELS * spatial_size * spatial_size

model = Classifier2D(
    embed_dim                  = EMBED_DIM,
    flat_dim                   = flat_dim,
    num_classes                = NUM_CLASSES,
    num_encoder_final_channels = NUM_FINAL_CHANNELS
)

trainer = ClassificationTrainer(
    model        = model,
    train_loader = train_loader,
    val_loader   = val_loader,
    lr           = LR,
    weight_decay = WEIGHT_DECAY,
    device       = DEVICE
)


from tqdm import tqdm

# —————————————————————————————————————————————
# 7) Training loop
# —————————————————————————————————————————————
for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
    train_loss, train_acc = trainer.train_epoch()
    val_loss,   val_acc   = trainer.validate_epoch()
    print(f"Epoch {epoch:02d} | "
          f"Train L={train_loss:.4f}, A={train_acc:.4f} | "
          f"Val   L={val_loss:.4f}, A={val_acc:.4f}")
