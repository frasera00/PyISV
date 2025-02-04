import sys
import torch
import numpy as np
import time
from PyISV.features_calc_utils import triple_kde_calc
from ase.io import read
import subprocess
from tqdm import tqdm
#### Example script to calculate RDF via KDE using PyISV ###

# the function torc_rdf_calc needs as inputs:
# - the positions of the atoms as a torch tensor of shape (N_atoms,3)
# - the radial distances at which the RDF is calculated
# - the bandwidth value of the gaussian kernels


# load your conf in a list, here ase Atoms objects are needed
configurations = []


# distances range and number of bins
min_dist = 0
max_dist = 4.5
num_bins = 200
bins = torch.linspace(min_dist, max_dist, num_bins)
# define bandwidth of the kernels, must be a torch.Tensor
bw = torch.Tensor([0.05])
# species
species1 = 'Ti'
species2 = 'O'
# periodic calc flag
periodic_calculation = False
# array to store outputs
rdfs_s1_s1=[]
rdfs_s2_s2=[]
rdfs_s1_s2=[]
# loop over configurations and calculate RDF

mic=False             
t0 = time.time()
for conf in tqdm(configurations):
    if periodic_calculation:
        conf.set_pbc(True)
        conf=conf.repeat(2)
        mic = True
    s1_s1_rdf, s2_s2_rdf, s1_s2_rdf = triple_kde_calc(species1, species2, conf, bins, bw, mic=mic)
    rdfs_s1_s1.append(s1_s1_rdf)
    rdfs_s2_s2.append(s2_s2_rdf)
    rdfs_s1_s2.append(s1_s2_rdf)
elapsed_time = time.time()-t0
print(f'Done, Elapsed time: {elapsed_time:.3f} s')
# save the results
rdfs_s1_s1 = np.vstack(rdfs_s1_s1)
rdfs_s2_s2 = np.vstack(rdfs_s2_s2)
rdfs_s1_s2 = np.vstack(rdfs_s1_s2)

collector  = np.zeros((len(configurations),3,num_bins))
collector [:,0,:] = rdfs_s1_s1
collector [:,1,:] = rdfs_s2_s2
collector [:,2,:] = rdfs_s1_s2

np.save('{}{}_triple_rdfs.npy'.format(species1,species2), collector)
