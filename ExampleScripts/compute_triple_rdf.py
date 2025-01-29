import sys
import torch
import numpy as np
import time
from PyISV.kde_rdf import torch_kde_calc
from ase.io import read
import subprocess
from tqdm import tqdm
#### Example script to calculate RDF via KDE using PyISV ###

# the function torc_rdf_calc needs as inputs:
# - the positions of the atoms as a torch tensor of shape (N_atoms,3)
# - the radial distances at which the RDF is calculated
# - the bandwidth value of the gaussian kernels


#### READ XSF ####
def run_ls_grep_command(pattern):
    # Run the 'ls' command and pipe its output to 'grep'
    ls_process = subprocess.Popen(['ls'], stdout=subprocess.PIPE)
    grep_process = subprocess.Popen(['grep', pattern], stdin=ls_process.stdout, stdout=subprocess.PIPE, text=True)

    # Close the stdout of ls_process to allow ls_process to receive a SIGPIPE if grep_process exits
    ls_process.stdout.close()

    # Capture the output of the 'grep' command
    output, _ = grep_process.communicate()

    # Split the output into an array of strings
    output_array = output.splitlines()

    return output_array

# Example usage
pattern = 'xsf'
output = run_ls_grep_command(pattern)

configurations = []
for i in range(len(output)):
    configurations.append(read(output[i]))


# distances range and number of bins
min_dist = 0
max_dist = 4.5
num_bins = 200
bins = torch.linspace(min_dist, max_dist, num_bins)
# define bandwidth of the kernels, must be a torch.Tensor
bw = torch.Tensor([0.05])
# array to store outputs
rdfs_ti_ti=[]
rdfs_o_o=[]
rdfs_ti_o=[]
# loop over configurations and calculate RDF
t0 = time.time()
for conf in tqdm(configurations):
    conf.set_pbc(True)
    repeat = conf.repeat(2)
    ti_indices = [atom.index for atom in repeat if atom.symbol == 'Ti'] 
    o_indices = [atom.index for atom in repeat if atom.symbol == 'O']
    ti_ti_dist = []
    o_o_dist = []
    ti_o_dist = []
    if len(ti_indices) > 1:    
        for i in range(len(ti_indices)-1):
            ti_ti_dist.append(repeat.get_distances(ti_indices[i],ti_indices[i+1:],mic=False)) 
        ti_ti_dist = np.hstack(ti_ti_dist)
        ti_ti_rdf = torch_kde_calc(torch.Tensor(ti_ti_dist), bins, bw) # compute kde
    else:
        ti_ti_rdf = torch.zeros(num_bins)
    if len(o_indices) > 1:    
        for i in range(len(o_indices)-1):
            o_o_dist.append(repeat.get_distances(o_indices[i],o_indices[i+1:],mic=False)) 
        o_o_dist = np.hstack(o_o_dist)
        o_o_rdf = torch_kde_calc(torch.Tensor(o_o_dist), bins, bw) # compute kde
    else:
        o_o_rdf = torch.zeros(num_bins)
    if len(ti_indices) > 1 and len(o_indices) > 1:    
        for i in range(len(ti_indices)-1):
            ti_o_dist.append(repeat.get_distances(ti_indices[i],o_indices,mic=False))
        ti_o_dist = np.hstack(ti_o_dist)
        ti_o_rdf = torch_kde_calc(torch.Tensor(ti_o_dist), bins, bw) # compute kde
    else:
        ti_o_rdf = torch.zeros(num_bins)
    rdfs_ti_ti.append(ti_ti_rdf.numpy())
    rdfs_o_o.append(o_o_rdf.numpy())
    rdfs_ti_o.append(ti_o_rdf.numpy())
elapsed_time = time.time()-t0
print(f'Done, Elapsed time: {elapsed_time:.3f} s')
# save the results
rdfs_ti_ti = np.vstack(rdfs_ti_ti)
rdfs_o_o = np.vstack(rdfs_o_o)
rdfs_ti_o = np.vstack(rdfs_ti_o)

collector  = np.zeros((len(configurations),3,num_bins))
collector [:,0,:] = rdfs_ti_ti
collector [:,1,:] = rdfs_o_o
collector [:,2,:] = rdfs_ti_o

np.save('TiO_triple_rdfs.npy', collector)
