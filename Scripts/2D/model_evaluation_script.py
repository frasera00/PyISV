import torch
from PyISV.network_ret import Autoencoder2D
from PyISV.train_utils import Dataset,RMSELoss,MSELoss,SaveBestModel
from torchsummary import summary
import numpy as np
import time
from tqdm import tqdm


##########################
####### PARAMETERS #######
##########################


# model file path
model_path = './best_model.pth' # complete path of the saved model, model file included
input_data_path = 'path_to_data'
input_dimensionality = 2 # 1 if the input is an array (1D convolutions), 2 if using matrices (2D convolutions)
pure_autoencoder = True # if True the model aims to reproduce the inputs in the output layer
padding = True
if padding:
   padding_final_size = 32


# scaler parameters path
path_input_scaler_subval = "./input_scaler_subval.npy"
path_input_scaler_divval = "./input_scaler_divval.npy"
if pure_autoencoder == False:
    path_output_scaler_subval = "./output_scaler_subval.npy"
    path_output_scaler_divval = "./output_scaler_divval.npy"
# model parameters
embed_dim = 2 # bottleneck size
flat_dim = 2*2 # geometric parameter of the network, 21 is needed for input vector of 340 numbers
device = 'cpu' # set the device for running the model

# evaluation parameter
# if True only encoder will run, otherwise also reconstruction will be saved
run_only_encoder = False

#########################
####### LOAD DATA #######
#########################

# load your data in a shape of (N_data, N_features)
# if data are RDFs the shape will corresponf to (N_data, N_bins)

input_data = np.load(input_data_path)
input_size = [*input_data.shape]
if padding:
    final_size = padding_final_size
    padded = []
    pad_size = final_size - input_size[1]
    for sample in input_data:
        padded.append(np.pad(sample, ((pad_size, 0), (pad_size, 0)), mode='constant', constant_values=0))
    padded = np.stack(padded)  
    del input_data
    input_data = np.copy(padded) 
    input_size = [*input_data.shape]
    del padded


################################
########## LOAD MODEL ##########
################################

# path where model and scaler parameters are stored

model_file=model_path

# load scalers parameters 
input_scaler_subval=np.load(path_input_scaler_subval)
input_scaler_divval=np.load(path_input_scaler_divval)
if pure_autoencoder == False:
    output_scaler_subval=np.load(path_output_scaler_subval)
    output_scaler_divval=np.load(path_output_scaler_divval)
else:
    output_scaler_subval=input_scaler_subval
    output_scaler_divval=input_scaler_divval

# initialize model class and load the saved checkpoint

num_channels = 1
if len(input_data.shape) > input_dimensionality+1:
    num_channels = input_data.shape[1]
else:
    input_data = np.expand_dims(input_data,1)
model_kwargs={
        'embed_dim': embed_dim,
        'flat_dim': flat_dim,
         }   

print(input_data.shape, input_scaler_divval.shape)

checkpoint = torch.load(model_path,map_location=torch.device(device))
print('Best model at epoch: ', checkpoint['epoch'])

model = Autoencoder2D(**model_kwargs)
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
del input_data
scaled_input=torch.Tensor(scaled_input)

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
print(scaledback_outputs.shape)
if (run_only_encoder!=True):
    np.save("rescaled_outputs.npy",scaledback_outputs)

