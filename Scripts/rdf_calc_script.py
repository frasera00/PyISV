import sys
import torch
import numpy as np
import time
from PyISV.kde_rdf import torch_rdf_calc


#### Example script to calculate RDF via KDE using PyISV ###

# the function torc_rdf_calc needs as inputs:
# - the positions of the atoms as a torch tensor of shape (N_atoms,3)
# - the radial distances at which the RDF is calculated
# - the bandwidth value of the gaussian kernels

# generate 1000 random samples for a system of 147 atoms
configurations = [] # generate random samples for a system of 147 atoms
for _ in range(1000):
    pos = np.random.rand(147,3)
    configurations.append(pos)
# distances range and number of bins
min_dist = 0
max_dist = 2
num_bins = 300
bins = torch.linspace(min_dist, max_dist, num_bins)
# define bandwidth of the kernels, must be a torch.Tensor
bw = torch.Tensor([0.2])
# array to store outputs
rdfs=[]
# loop over configurations and calculate RDF
t0 = time.time()
for conf in configurations:
    torch_conf = torch.Tensor(conf) # cast conf to torch.Tensor
    conf_rdf = torch_rdf_calc(torch_conf, bins, bw) # compute kde
    rdfs.append(conf_rdf.numpy())
elapsed_time = time.time()-t0
print(f'Done, Elapsed time: {elapsed_time:.3f} s')
# save the results
np.save('rdfs.npy', np.vstack(rdfs))