import torch
from PyISV.network import Autoencoder
from PyISV.train_utils import Dataset,RMSELoss,MSELoss,SaveBestModel
from torchsummary import summary
import numpy as np
import time
from tqdm import tqdm


##########################
####### PARAMETERS #######
##########################

# set to True if random data from training script have been saved and specify path
saved_random_data = True
if saved_random_data: # specify their paths
    random_inputs_path = './random_inputs.npy'

# model file path
model_path = './best_model.pth' # complete path of the saved model, model file included

# scaler parameters path
path_input_scaler_subval = "./input_scaler_subval.dat"
path_input_scaler_divval = "./input_scaler_divval.dat"
path_output_scaler_subval = "./output_scaler_subval.dat"
path_output_scaler_divval = "./output_scaler_divval.dat"

# model parameters
embed_dim = 2 # bottleneck size
flat_dim = 1 # geometric parameter of the network, 21 is needed for input vector of 340 numbers
device = 'cpu' # set the device for running the model

# evaluation parameter
# if True only encoder will run, otherwise also reconstruction will be saved
run_only_encoder = True 

#########################
####### LOAD DATA #######
#########################

# load your data in a shape of (N_data, N_features)
# if data are RDFs the shape will corresponf to (N_data, N_bins)

# if random data from training script have been saved they will be loaded for evaluation
if saved_random_data:
    input_data = np.load(random_inputs_path)
# otherwise other random data are generated, of size 340
else:
    input_data = []
    for _ in range(5000):
        input_data.append(np.random.rand(340))
    input_data = np.vstack(input_data) 

################################
########## LOAD MODEL ##########
################################

# path where model and scaler parameters are stored

model_file=model_path

# load scalers parameters 
input_scaler_subval=np.genfromtxt(path_input_scaler_subval)
input_scaler_divval=np.genfromtxt(path_input_scaler_divval)

output_scaler_subval=np.genfromtxt(path_output_scaler_subval)
output_scaler_divval=np.genfromtxt(path_output_scaler_divval)

# initialize model class and load the saved checkpoint

num_channels = 1
if len(input_data.shape)>2:
    num_channels = input_data.shape[1]
model_kwargs={
        'embed_dim': embed_dim,
        'flat_dim': flat_dim,
        'input_channels': num_channels
         }   

checkpoint = torch.load(model_path,map_location=torch.device(device))
print('Best model at epoch: ', checkpoint['epoch'])

model = Autoencoder(**model_kwargs)
model.load_state_dict(checkpoint['model_state_dict'])
# set the device (cpu or cuda)
model.to(device)
model.eval()    

# uncomment next line to print summary of the model
#_=summary(model,(1,340)) 

################################
######## EVALUATE MODEL ########
################################

# scaling inputs before feeding them to the model
scaled_input=(input_data-input_scaler_subval)/input_scaler_divval
scaled_input=torch.Tensor(scaled_input)
if len(scaled_input.shape)==2:
    scaled_input = scaled_input.unsqueeze(1)

# in the evaluation to look only at the bottleneck values (the cvs)
# otherwise, to look also to the reconstructions the full network needs to be forwarded

embed=[]
if (run_only_encoder!=True):
    outputs=[]
t0 = time.time()    
for sample in tqdm(scaled_input):
    # run only encoder and save only the bottleneck values
    if (run_only_encoder==True):
        embed.append(model.encode(sample.unsqueeze(0).to(device)).detach().cpu().numpy())
    else:
    # run all the network and save outputs and bottleneck values
        output,embed_val=model(sample.unsqueeze(0).to(device))
        outputs.append(output.squeeze(0).detach().cpu().numpy())
        embed.append(embed_val.detach().cpu().numpy())
embed=np.vstack(embed)  
if (run_only_encoder!=True):
    outputs=np.stack(outputs)
    # outputs need to be scaled back to be compared with the original data
    scaledback_outputs=outputs*output_scaler_divval + output_scaler_subval
print("Elapsed time for evaluation: {0:.2f} s".format(time.time()-t0))
np.save("embed.npy",embed)
if (run_only_encoder!=True):
    np.save("rescaled_outputs.npy",scaledback_outputs)

