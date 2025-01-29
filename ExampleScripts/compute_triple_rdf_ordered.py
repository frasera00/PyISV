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


def compute_ase_idx_distances(conf, indices1, indices2=[], mic=False):
    dist_list = []
    if indices2 == []:
        for i in range(len(indices1)-1):
            dist_list.append(conf.get_distances(indices1[i],indices1[i+1:],mic=mic))
    else:    
        for i in range(len(indices1)):
            dist_list.append(conf.get_distances(indices1[i],indices2,mic=mic))     
    return np.hstack(dist_list)    



def compute_double_element_rdf(specie1, specie2, conf, bins, bw, mic=False):
    there_is_s1 = False
    there_is_s2 = False
   
    indices_specie1 = [atom.index for atom in conf if atom.symbol == specie1] 
    indices_specie2 = [atom.index for atom in conf if atom.symbol == specie2]
    
    if len(indices_specie1) > 1:
        there_is_s1 = True
    if len(indices_specie2) > 1:
        there_is_s2 = True 

    s1_s1_rdf = np.zeros(num_bins)
    s2_s2_rdf = np.zeros(num_bins)
    s1_s2_rdf = np.zeros(num_bins)

    if there_is_s1:
        s1_s1_dist = compute_ase_idx_distances(conf, indices_specie1, indices2=[], mic=mic)
        s1_s1_rdf = torch_kde_calc(torch.Tensor(s1_s1_dist), bins, bw).numpy()

    if there_is_s2:
        s2_s2_dist = compute_ase_idx_distances(conf, indices_specie2, indices2=[], mic=mic)
        s2_s2_rdf = torch_kde_calc(torch.Tensor(s2_s2_dist), bins, bw).numpy()

    if there_is_s1 and there_is_s2:    
        s1_s2_dist = compute_ase_idx_distances(conf, indices_specie1, indices2=indices_specie2, mic=mic)
        s1_s2_rdf = torch_kde_calc(torch.Tensor(s1_s2_dist), bins, bw).numpy()        
    
    return s1_s1_rdf, s2_s2_rdf, s1_s2_rdf

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
# species
species1 = 'Ti'
species2 = 'O'
# periodic calc flag
periodic_calculation = True
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
    s1_s1_rdf, s2_s2_rdf, s1_s2_rdf = compute_double_element_rdf(species1, species2, conf, bins, bw, mic=mic)
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
