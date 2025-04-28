import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import ase.io as io
from PyISV.features_calc_utils import single_kde_calc     

# --- at the bottom of train_utils.py ---

# === Classification Helpers ===

class ClassificationLoss(CrossEntropyLoss):
    """
    Alias for CrossEntropyLoss specifically for classification tasks.
    """
    def __init__(self):
        super(ClassificationLoss, self).__init__()


class ClassificationTrainer:
    """
    Trainer for supervised classification using CrossEntropyLoss.
    """
    def __init__(self, model, train_loader, val_loader=None,
                 lr=1e-3, weight_decay=0, device='cpu'):
        self.model      = model.to(device)
        self.device     = device
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion  = ClassificationLoss()
        self.optimizer  = torch.optim.Adam(
                              model.parameters(),
                              lr=lr,
                              weight_decay=weight_decay
                          )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds         = torch.argmax(logits, dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

        return running_loss / total, correct / total

    def validate_epoch(self):
        if self.val_loader is None:
            raise ValueError("No validation loader provided.")
        self.model.eval()
        running_loss = 0.0
        correct      = 0
        total        = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                loss   = self.criterion(logits, labels)

                running_loss += loss.item() * inputs.size(0)
                preds         = torch.argmax(logits, dim=1)
                correct      += (preds == labels).sum().item()
                total        += labels.size(0)

        return running_loss / total, correct / total


################################################
################# Loss Functions ###############
################################################

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(x, y)
        return loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.BCELoss()
        loss = criterion(x, y)
        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(x, y)
        return loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss,self).__init__()

    def forward(self,x,y):
        criterion =  nn.MSELoss()
        loss = criterion(x, y)
        return loss

class HuberLoss(nn.Module):
    def __init__(self):
        super(HuberLoss,self).__init__()

    def forward(self,x,y,delta):
        a=torch.abs(x-y)
        a=torch.flatten(a)
        batchsize=a.shape[0]
        lossval=torch.zeros(batchsize)
        for i in range(batchsize):
            if (a[i] <= delta):
                lossval[i] = 0.5*(a[i]**2.0)
            else:
                lossval[i] = delta*(a[i]-0.5*delta)
        loss=torch.sum(lossval)/batchsize
        return loss

################################################
################# Dataset class  ###############
################################################

class Dataset(Dataset):

    def __init__(self, inputs, targets, norm_inputs=False, norm_targets=False, norm_mode="minmax", norm_threshold_inputs=1e-6, norm_threshold_targets=1e-6):
        
        self.num_inputs = inputs.shape[0]    
        self.num_inputs_features = inputs.shape[-1]
        self.num_targets = targets.shape[0]    
        self.num_targets_features = targets.shape[-1]
        
        self.inputs = inputs
        self.targets = targets
        
        self.norm_threshold_inputs = norm_threshold_inputs
        self.norm_threshold_targets = norm_threshold_targets

        self.norm_mode= norm_mode # 'minmax' or 'gaussian'
        if ((self.norm_mode != "gaussian") and (self.norm_mode != "minmax")):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
            print('norm_mode must be equal to "minmax" or "gaussian"\n')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
        
        if (norm_inputs==True):  
            self.inputs = self.set_norm_inputs(inputs)
        if (norm_targets==True):
            self.targets = self.set_norm_targets(targets)
            
    def __len__(self):
        return self.num_inputs
        
    def set_norm_inputs(self, x): 
        if (self.norm_mode == "minmax"):
            self.maxval_inputs = torch.max(x,axis=0).values
            self.subval_inputs = torch.min(x,axis=0).values
            self.divval_inputs = (self.maxval_inputs - self.subval_inputs)
            self.divval_inputs[self.divval_inputs < self.norm_threshold_inputs] = 1
        if (self.norm_mode == "gaussian"):
            self.subval_inputs = torch.mean(x,axis=0)
            self.divval_inputs = torch.std(x,axis=0)
            self.divval_inputs[self.divval_inputs < self.norm_threshold_inputs] = 1            

        x = self.rescale(x, self.subval_inputs, self.divval_inputs)
        return x

    def set_norm_targets(self, x):
        if (self.norm_mode == "minmax"):
            self.maxval_targets = torch.max(x,axis=0).values
            self.subval_targets = torch.min(x,axis=0).values
            self.divval_targets = (self.maxval_targets - self.subval_targets)
            self.divval_targets[self.divval_targets < self.norm_threshold_targets] = 1
        if (self.norm_mode == "gaussian"):
            self.subval_targets = torch.mean(x,axis=0)
            self.divval_targets = torch.std(x,axis=0)
            self.divval_targets[self.divval_targets < self.norm_threshold_targets] = 1            

        x = self.rescale(x, self.subval_targets, self.divval_targets)
        return x    

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
        
    def rescale(self, x, minval, rangeval):
        dataset_size = x.shape[0]
        sample_size = x.shape[1:]
        subval = torch.Tensor(minval).unsqueeze(0).expand(dataset_size, *sample_size)
        divval = torch.Tensor(rangeval).unsqueeze(0).expand(dataset_size, *sample_size)
        return torch.Tensor(x).sub(subval).div(divval)  

################################################
#########  Save best model class ###############
################################################

class SaveBestModel():
    # see: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def __init__(self, best_valid_loss=float('inf'), best_train_loss=float('inf'), best_model_name="best_model"): 
        self.best_valid_loss = best_valid_loss
        self.best_train_loss = best_train_loss
        self.best_model_name = best_model_name
        self.best_epoch = None
    
    def __call__(self, current_valid_loss, current_train_loss, epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_train_loss = current_train_loss
            self.best_epoch = epoch
            print(f"  Saving best model at epoch: {epoch}, Best validation loss: {self.best_valid_loss}")
            torch.save({
                'epoch': self.best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': self.best_valid_loss,
                'best_train_loss': self.best_train_loss
                }, self.best_model_name+'.pth')

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait before stopping when loss stops improving.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0  # Reset counter if improvement
        else:
            self.counter += 1  # Increment counter if no improvement

        return self.counter >= self.patience

class MultiFrameRDFImageDataset(Dataset):
    def __init__(self, xyz_path, labels, bins, bandwidth):
        self.xyz_path  = xyz_path
        self.labels    = labels
        self.bins      = bins
        self.bandwidth = bandwidth
        self.n_bins    = bins.numel()

        # count frames
        try:
            self.num_frames = io.get_frame_count(xyz_path)
        except AttributeError:
            # fallback
            frames = io.read(xyz_path, index=":")
            self.num_frames = len(frames)

        assert self.num_frames == labels.size(0), (
            f"{self.num_frames} frames vs {labels.size(0)} labels"
        )

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        atoms  = io.read(self.xyz_path, index=idx)
        rdf_1d = single_kde_calc(atoms, self.bins, self.bandwidth)
        rdf    = torch.from_numpy(rdf_1d).float()

        if rdf.dim() == 2:              # if shape (1, N)
            rdf = rdf.squeeze(0)         # remove first dimension
        elif rdf.dim() == 1:
            pass                        # already OK
        else:
            raise ValueError(f"Unexpected RDF shape: {rdf.shape}")
        
        img = rdf.unsqueeze(0).repeat(self.n_bins, 1).unsqueeze(0)
        return img, self.labels[idx]



