import torch
from torch.utils.data import DataLoader
from PyISV.network import Autoencoder
from PyISV.train_utils import Dataset,RMSELoss,MSELoss,SaveBestModel
from torchsummary import summary
import numpy as np
from sklearn.model_selection import train_test_split
import time

##########################
####### PARAMETERS #######
##########################

# set to True testing the script with random data and you wish to save them for evaluation
save_random_data=True

# data parameters
train_fraction = 0.8 # percentage of the data to use for training set
batch_size = 64 # batch size

# model parameters
embed_dim = 2 # bottleneck size
flat_dim = 1 # geometric parameter of the network, 21 is needed for input vector of 340 numbers
seed = 7352143264209594346 # manual seed for model initialization

# training parameters
device = 'cpu' # set the device for training
num_epochs = 150 # number of epochs to perform in the training loop 
saved_model_name = 'best_model' # name of the .pth file where best model is saved during training

#########################
####### LOAD DATA #######
#########################

# load your data in a shape of (N_data, N_features)
# if data are RDFs the shape will corresponf to (N_data, N_bins)

# here some random data are generated, of size 340
input_data = []
target_data = []
for _ in range(5000):
    input_data.append(np.random.rand(340))
    target_data.append(np.random.rand(340))
input_data = np.vstack(input_data)
target_data = np.vstack(target_data)   

if save_random_data:
    np.save('random_inputs.npy',input_data)
    np.save('random_targets.npy',target_data)



###############################
####### SETTING UP DATA #######
###############################

input_size = input_data.shape[-1]
num_channels = 1
if len(input_data.shape)>2:
    num_channels = input_data.shape[1]
# switching on the normalization for both inputs and targets 
norm_inputs = True
norm_targets = True
norm_mode = "minmax" # two types of normalizations implemented 'minmax' and 'gaussian'
# initializing dataset class
input_data = torch.tensor(input_data,dtype=torch.double) 
target_data = torch.tensor(target_data,dtype=torch.double) 
if len(input_data.shape)==2:
    input_data = input_data.unsqueeze(1)
if len(target_data.shape)==2:
    target_data = target_data.unsqueeze(1)    

dataset = Dataset(
                input_data,
                target_data,
                norm_inputs=norm_inputs, 
                norm_targets=norm_targets,
                norm_mode=norm_mode
                )

# save the normalization parameters for both inputs and targets, in order to be able to scale again new data to feed to the network or scale back the reconstructed outputs
np.savetxt("input_scaler_subval.dat",dataset.subval_inputs.numpy())
np.savetxt("input_scaler_divval.dat",dataset.divval_inputs.numpy())
np.savetxt("output_scaler_subval.dat",dataset.subval_targets.numpy())
np.savetxt("output_scaler_divval.dat",dataset.divval_targets.numpy())

# percentage of data to use as training set
train_fraction = train_fraction

# splitting inputs (X) and targets (Y) into training and validation sets
X_train,X_valid,Y_train,Y_valid = train_test_split(dataset.inputs,dataset.targets,train_size=train_fraction, shuffle=True, random_state=1234)

print('''SPLITTING DATA INTO TRAINING AND VALIDATION SETS FOR TRAINING
###################################
{0:.2f}% used as training set
{1:.2f}% used as validation set
- Training data   = {2:d}
- Validation data = {3:d}
###################################'''.format(train_fraction*100,(1 - train_fraction)*100,len(X_train),len(X_valid)))

# creating batches for training, here Dataset class is used again but with no normalization
batch_size = batch_size

train_dataset=Dataset(X_train, Y_train, norm_inputs=False, norm_targets=False)
train_loader=DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

valid_dataset=Dataset(X_valid, Y_valid, norm_inputs=False, norm_targets=False)
valid_loader=DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

print('''SHUFFLING TRAINING AND VALIDATION DATASETS AND CREATING BATCHES")
###################################
FINAL SUMMARY
Data fraction used as training set = {0:.2f}
Training data      = {1:d}
Validation data    = {2:d}
Batch size         = {3:d}
Training batches   = {4:d}
Validation batches = {5:d}
###################################'''.format(train_fraction,len(train_loader)*batch_size,
len(valid_loader)*batch_size,batch_size,len(train_loader),len(valid_loader))) 

####################################
####### MODEL INITIALIZATION #######
####################################

model_kwargs = {
    'embed_dim': embed_dim,
    'flat_dim': flat_dim,
    'input_channels': num_channels
}
torch.manual_seed(seed) # set seed if reproducibility required
model = Autoencoder(**model_kwargs)

_ = summary(model,(num_channels,input_size))

####################################
##### TRAINING HYPERPARAMETERS #####
####################################

# set init epoch to 0 
init_epoch = 0

# set the device
device=device
model.to(device)

# define starting learning rate
print("[Optimizer Parameters]")
lrate = 5e-3
print("Learning rate \t=",lrate)

# optimizer
scheduled_lr = True
optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

# setting scheduler for the lr. Here multistep and cosineannealing schedulers are proposed   
if (scheduled_lr == True): 
    print ("Scheduler On")
    #lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25, eta_min=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75,125], gamma = 0.5)  
else:
    print ("Scheduler Off")
    
# set loss function
print("")
print("[Loss]")
Loss_type = "MSE"

### RMSE ####
if (Loss_type=="RMSE"):
    loss_function=RMSELoss()
    print("Loss function = "+Loss_type)

### MSE ####
if (Loss_type=="MSE"):
    loss_function=MSELoss()
    print("Loss function = "+Loss_type)

# define arrays to store infos
hist_train = []
hist_valid = []
learn_rate = []

####################################
######### TRAINING LOOP ############
####################################

# set number of epoch (iterations) to perform during the training loop and how frequently print stats
num_epochs=num_epochs
print_loss=5 # will print every 5 epochs

# initializing class to save best model during training
save_best_model = SaveBestModel(best_model_name = saved_model_name)

# initialize log file
with open("train_stats.txt", "w") as f:
    log = "#Epoch  Time(s)  Train_loss  Valid_Loss  LR"
    f.write(log+'\n')

# training loop
for epoch in range(num_epochs):
    t0 = time.time()
    
    # set up the model for training and weights updating
    model.train()

    train_loss = 0.0
    counter = 0

    for x,y in train_loader:
        counter = counter + 1
        x = x.float().to(device)
        y = y.float().to(device)
        # model evaluation
        output,hidden = model(x)
        # loss
        loss=loss_function(output,y)
        train_loss = train_loss + loss.item()
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = train_loss/counter
    hist_train.append(train_loss)
    
    # set up the model for validation
    model.eval()

    valid_loss = 0.0
    counter = 0
    with torch.no_grad():
        for x_valid,y_valid in valid_loader:
            counter = counter + 1
            x_valid = x_valid.float().to(device)
            y_valid = y_valid.float().to(device)
            output_valid,hidden_valid = model(x_valid)
            vloss=loss_function(output_valid,y_valid)
            valid_loss = valid_loss + vloss.item()

    valid_loss = valid_loss/counter
    hist_valid.append(valid_loss)

    # call SaveBestModel to compare the iteration result with the previous saved model
    save_best_model(valid_loss, init_epoch+epoch, model, optimizer)

    # update lr following the scheduler
    if (scheduled_lr == True):
        current_lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()
    else:
        current_lr = optimizer.param_groups[0]['lr']
    learn_rate.append(current_lr)
    elapsed_time = time.time()-t0

    # append data to plot
    if ((epoch+1)%print_loss == 0):
        print("Epoch: {0:d}, Time(s) = {1:.2f}, Train_loss = {2:.9f}, Valid_Loss = {3:.9f}, LR = {4:1.2e}".format(
            init_epoch+epoch+1, elapsed_time, train_loss, valid_loss, current_lr))
    # print training stats to file
    log="{0:d} {1:.2f} {2:.9f} {3:9f} {4:1.2e}".format(init_epoch+epoch+1, elapsed_time, train_loss, valid_loss, current_lr)
    #log=np.array([init_epoch+epoch+1, elapsed_time, train_loss, valid_loss, current_lr]).reshape(1,-1)
    with open("train_stats.txt", "a") as f:
        f.write(log+'\n')