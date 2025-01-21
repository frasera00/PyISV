import sys
import torch
import numpy as np
import time
from PyISV.kde_rdf import periodic_torch_rdf_calc


#### Example script to calculate RDF via KDE using PyISV ###

# the function torc_rdf_calc needs as inputs:
# - the positions of the atoms as a torch tensor of shape (N_atoms,3)
# - the radial distances at which the RDF is calculated
# - the bandwidth value of the gaussian kernels


periodic_system = True
if periodic_system:
    box = np.array([[27.2,0,0],
           [0,27.2,0],
           [0,0,27.2]])
    z_periodicity = True
if periodic_system:
    plane_replicas = 2
    z_replicas = 2
# generate 1000 random samples for a system of 147 atoms
configurations = [] # generate random samples for a system of 147 atoms
for _ in range(1):
    pos = np.random.rand(30,3)*5
    configurations.append(pos)
configurations = []
configurations.append(np.genfromtxt('argon_np.xyz'))
#configurations = []
#configurations.append(np.array([[1,0,0],
#                                [1,1,0],
#                                [0,1,0],
#                                [0,0,0],
#                                [1,0,1],
#                                [1,1,1],
#                                [0,1,1],
#                                [0,0,1]]))   

if periodic_system:
    main_atoms = len(configurations[0])

if periodic_system:

    periodic_configurations = []   
    for conf in configurations:
        replicates = []
        for xx in range(1,plane_replicas):
            new_conf = conf + box[0]*(xx)
            replicates.append(new_conf[new_conf[:,0] > box[0][0]*(xx)])
            new_conf = conf - box[0]*(xx)
            replicates.append(new_conf[new_conf[:,0] < (0 - box[0][0]*(xx-1))])
        replicates = np.vstack(replicates)
        conf = np.vstack((conf,replicates))
        replicates = []
        for yy in range(1,plane_replicas):
            new_conf = conf + box[1]*(yy)
            replicates.append(new_conf[new_conf[:,1] > np.sum(box[1]*(yy))])
            new_conf = conf - box[1]*(yy)
            replicates.append(new_conf[new_conf[:,1] < (0 - np.sum(box[1]*(yy-1)))])
        replicates = np.vstack(replicates)
        conf = np.vstack((conf,replicates))
        if z_periodicity:
            replicates = []
            for zz in range(1,z_replicas):
                new_conf = conf + box[2]*(zz)
                replicates.append(new_conf[new_conf[:,2] > np.sum(box[2]*(zz))])
                new_conf = conf - box[2]*(zz)
                replicates.append(new_conf[new_conf[:,2] < (0 - np.sum(box[2]*(zz-1)))])
            replicates = np.vstack(replicates)
            conf = np.vstack((conf,replicates))  
        periodic_configurations.append(conf)
    del configurations
    configurations = periodic_configurations
    del periodic_configurations


indices = []
print(len(configurations[0]))
for i in range(main_atoms):
    for j in range(len(configurations[0])):
        if j>i:
           indices.append(np.array([i,j]))
indices = torch.tensor(np.vstack(indices).T, dtype=torch.int32)

# distances range and number of bins
min_dist = 0
max_dist = 13.5
num_bins = 300
bins = torch.linspace(min_dist, max_dist, num_bins)
# define bandwidth of the kernels, must be a torch.Tensor
bw = torch.Tensor([0.1])
# array to store outputs
rdfs=[]
# loop over configurations and calculate RDF
t0 = time.time()
for conf in configurations:
    torch_conf = torch.Tensor(conf) # cast conf to torch.Tensor
    conf_rdf = periodic_torch_rdf_calc(torch_conf, indices, bins, bw) # compute kde
    rdfs.append(conf_rdf.numpy())
elapsed_time = time.time()-t0
print(f'Done, Elapsed time: {elapsed_time:.3f} s')
# save the results
np.save('rdfs.npy', np.vstack(rdfs))
