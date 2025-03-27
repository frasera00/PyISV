import sys
import torch
import numpy as np
import time
from PyISV.features_calc_utils import single_kde_calc
from ase.io import read
import subprocess
from tqdm import tqdm
#### Example script to calculate RDF via KDE using PyISV ###

# the function torc_rdf_calc needs as inputs:
# - the positions of the atoms as a torch tensor of shape (N_atoms,3)
# - the radial distances at which the RDF is calculated
# - the bandwidth value of the gaussian kernels


# load your conf in a list, here ase Atoms objects are needed
configurations = read('file.xyz')

# distances range and number of bins
min_dist = 0
max_dist = 4.5
num_bins = 200
bins = torch.linspace(min_dist, max_dist, num_bins)
# define bandwidth of the kernels, must be a torch.Tensor
bw = torch.Tensor([0.05])
# periodic calc flag
periodic_calculation = False
# array to store outputs
rdfs=[]
# loop over configurations and calculate RDF

mic=False             
t0 = time.time()
for conf in tqdm(configurations):
    if periodic_calculation:
        conf.set_pbc(True)
        conf=conf.repeat(2)
        mic = True
    conf_rdf = single_kde_calc(conf, bins, bw, mic=mic)
    rdfs.append(conf_rdf)
elapsed_time = time.time()-t0
print(f'Done, Elapsed time: {elapsed_time:.3f} s')
# save the results
rdfs = np.vstack(rdfs)

np.save('rdfs.npy', rdfs)
