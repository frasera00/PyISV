import torch
from torch.utils.data import DataLoader
from PyISV.network import Autoencoder
from PyISV.train_utils import Dataset,RMSELoss,MSELoss,SaveBestModel,EarlyStopping
from torchsummary import summary
import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt



train_fraction = 0.8 # percentage of the data to use for training set
batch_size = 64 # batch size

# model parameters
embed_dim = 2 # bottleneck size
flat_dim = 1 # geometric parameter of the network, 21 is needed for input vector of 340 numbers
seed = 7352143264209594346 # manual seed for model initialization

# training parameters
device = 'cpu' # set the device for training
num_epochs = 250 # number of epochs to perform in the training loop 
saved_model_name = 'best_model' # name of the .pth file where best model is saved during training
stopper_patience = 10 # early stopping patience
stopper_delta = 0.0005 # early stopping delta
input_path = 'path_to_data' # path to input data, include file name
padding = True # turn on or off the padding to match the established network input size

with open("train_log.txt", "w") as log:
    log.write("### LOGGING TRAINING INFO ###\n\n")
with open("train_log.txt", "a") as log:
    log.write(f'''[Model Parameters]
Embedding dimension = {embed_dim}
Flat dimension = {flat_dim}
Seed = {seed}

[Training Parameters]
Device = {device}
Maximum number of epochs = {num_epochs}
Saved model name = {saved_model_name}
Early stopping patience = {stopper_patience}
Early stopping delta = {stopper_delta}
Input path = {input_path}
Padded input = {padding}
Fraction of data for the training set = {train_fraction}
Batch size = {batch_size}

''')



input_data = np.load(input_path)
target_data = np.zeros((len(input_data),3))
input_size = [input_data.shape[1],input_data.shape[2]]
with open("train_log.txt", "a") as log:
    log.write(f"LOADING DATA\n")
    log.write(f"Input size: {input_size}\n")
num_channels = 1

if padding:
    with open("train_log.txt", "a") as log:
        log.write("Padding on!\n")
    padded = []
    target_shape = 32
    pad_size = target_shape - input_size[0]
    for sample in input_data:
        padded.append(np.pad(sample, ((pad_size, 0), (pad_size, 0)), mode='constant', constant_values=0))
    padded = np.stack(padded)  
    del input_data
    input_data = np.copy(padded) 
    input_size = [input_data.shape[1],input_data.shape[2]]
    with open("train_log.txt", "a") as log:
        log.write(f"Padded input size: {input_size}\n")
    del padded

norm_inputs = True
norm_targets = True
norm_mode = "minmax" # two types of normalizations implemented 'minmax' and 'gaussian'
# initializing dataset class
input_size = input_data.shape[-1]
num_channels = 1
if len(input_data.shape)>2:
    num_channels = input_data.shape[1]

dataset = Dataset(
                torch.tensor(input_data,dtype=torch.double), 
                torch.tensor(target_data,dtype=torch.double), 
                norm_inputs=norm_inputs, 
                norm_targets=norm_targets,
                norm_mode=norm_mode
                )
# save the normalization parameters for both inputs and targets, in order to be able to scale again new data to feed to the network or scale back the reconstructed outputs
np.savetxt("input_scaler_subval.dat",dataset.subval_inputs.numpy())
np.savetxt("input_scaler_divval.dat",dataset.divval_inputs.numpy())
#np.savetxt("output_scaler_subval.dat",dataset.subval_targets.numpy())
#np.savetxt("output_scaler_divval.dat",dataset.divval_targets.numpy())

del input_data
del target_data

# percentage of data to use as training set
train_fraction = train_fraction

# splitting inputs (X) and targets (Y) into training and validation sets
X_train,X_valid,Y_train,Y_valid = train_test_split(dataset.inputs,dataset.targets,train_size=train_fraction, shuffle=True, random_state=1234)

with open("train_log.txt", "a") as log:
    log.write('''
SPLITTING DATA INTO TRAINING AND VALIDATION SETS FOR TRAINING
###################################
{0:.2f}% used as training set
{1:.2f}% used as validation set
- Training data   = {2:d}
- Validation data = {3:d}
###################################

'''.format(train_fraction*100,(1 - train_fraction)*100,len(X_train),len(X_valid)))

# creating batches for training, here Dataset class is used again but with no normalization
batch_size = batch_size

train_dataset=Dataset(X_train, Y_train, norm_inputs=False, norm_targets=False)
train_loader=DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

valid_dataset=Dataset(X_valid, Y_valid, norm_inputs=False, norm_targets=False)
valid_loader=DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
del dataset
with open("train_log.txt", "a") as log:
    log.write('''SHUFFLING TRAINING AND VALIDATION DATASETS AND CREATING BATCHES")
###################################
FINAL SUMMARY
Data fraction used as training set = {0:.2f}
Training data      = {1:d}
Validation data    = {2:d}
Batch size         = {3:d}
Training batches   = {4:d}
Validation batches = {5:d}
###################################
'''.format(train_fraction,len(train_loader)*batch_size,
len(valid_loader)*batch_size,batch_size,len(train_loader),len(valid_loader))) 


num_channels = 1
model_kwargs = {
    'embed_dim': embed_dim,
    'flat_dim': flat_dim
}
torch.manual_seed(seed) # set seed if reproducibility required
model = Autoencoder2D(**model_kwargs)
model.to(device)
_ = summary(model,(num_channels,input_size[0],input_size[1]))



init_epoch = 0

# set the device

# define starting learning rate
with open("train_log.txt", "a") as log:
    log.write("\n[Optimizer Parameters]\n")
lrate = 5e-3
with open("train_log.txt", "a") as log:
    log.write(f"Learning rate = {lrate}\n")

# optimizer
scheduled_lr = True
optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

# setting scheduler for the lr. Here multistep and cosineannealing schedulers are proposed   
if (scheduled_lr == True): 
    with open("train_log.txt", "a") as log:
        log.write(f"Scheduler On\n")
    #lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25, eta_min=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,250], gamma = 0.5)  
else:
    with open("train_log.txt", "a") as log:
        log.write(f"Scheduler Off\n")
    
# set loss function
Loss_type = "MSE"
with open("train_log.txt", "a") as log:
    log.write(f'''[Loss]
Loss function = {Loss_type}
''')

### RMSE ####
if (Loss_type=="RMSE"):
    loss_function=RMSELoss()

### MSE ####
if (Loss_type=="MSE"):
    loss_function=MSELoss()

# define arrays to store infos
hist_train = []
hist_valid = []
learn_rate = []


num_epochs=num_epochs

with open("train_log.txt", "a") as log:
    log.write("\nPrinting training stat in train_stats.txt\n")
# initializing class to save best model during training
save_best_model = SaveBestModel(best_model_name = saved_model_name)
early_stopping = EarlyStopping(patience=stopper_patience, min_delta=stopper_delta)
# initialize log file
with open("train_stats.txt", "w") as f:
    train_log = "#Epoch  Time(s)  Train_loss  Valid_Loss  LR\n"
    f.write(train_log)

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
        output,hidden = model(x.unsqueeze(1))
        # loss
        loss=loss_function(output,x.unsqueeze(1))
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
            output_valid,hidden_valid = model(x_valid.unsqueeze(1))
            vloss=loss_function(output_valid,x_valid.unsqueeze(1))
            valid_loss = valid_loss + vloss.item()

    valid_loss = valid_loss/counter
    hist_valid.append(valid_loss)

    # call SaveBestModel to compare the iteration result with the previous saved model
    save_best_model(valid_loss, init_epoch+epoch, model, optimizer)
    if early_stopping(valid_loss):
        print(f"Early stopping at epoch {epoch}")
        with open("train_stats.txt", "a") as f:
            f.write(f"Early stopping at epoch {epoch}\n")
        with open("train_log.txt", "a") as log:
            log.write(f"###################\n")
            log.write(f"\n")
            log.write(f"Training completed!\n")
            log.write(f"Early stopping at epoch {epoch}\n")
            log.write(f"Last saved model at epoch: {save_best_model.best_epoch}\n")
            log.write(f"Best validation loss: {save_best_model.best_valid_loss}\n")
        break

    # update lr following the scheduler
    if (scheduled_lr == True):
        current_lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()
    else:
        current_lr = optimizer.param_groups[0]['lr']
    learn_rate.append(current_lr)
    elapsed_time = time.time()-t0

    # append data to plot
    # print training stats to file
    train_log="{0:d} {1:.2f} {2:.9f} {3:9f} {4:1.2e}".format(init_epoch+epoch, elapsed_time, train_loss, valid_loss, current_lr)
    #log=np.array([init_epoch+epoch+1, elapsed_time, train_loss, valid_loss, current_lr]).reshape(1,-1)
    with open("train_stats.txt", "a") as f:
        f.write(train_log+'\n')

if (epoch == num_epochs-1):
   with open("train_log.txt", "a") as log:
       log.write(f"###################\n")
       log.write(f"\n")
       log.write(f"Training completed!\n")
       log.write(f"!!! No early stopping !!!\n")
       log.write(f"Last saved model at epoch: {save_best_model.best_epoch}\n")
       log.write(f"Best validation loss: {save_best_model.best_valid_loss}\n")
