from pathlib import Path
import glob
import os
import shutil
import argparse
import random
import matplotlib.pyplot as plt

import xarray as xr
from datetime import datetime, timedelta
import numpy as np
import time
import random
import pandas as pd
import re
import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


data_path = Path("../")
output_path = Path("../output/")


# conda activate mlGpu2

class FC_model(nn.Module):
    def __init__(self, in_shape=None, out_shape=None):
        super(FC_model, self).__init__()
        
        feat_in = in_shape[1]
        feat_out = out_shape[1]
        
        self.fc1 = nn.Linear(feat_in, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, feat_out)
        
    def forward(self, x):
    
        print(f"Input shape: {x.shape}") 
        x = F.relu(self.fc1(x))
        print(f"FC1 shape: {x.shape}")
        x = F.relu(self.fc2(x))
        print(f"FC2 shape: {x.shape}")
        x = self.fc3(x)
        print(f"FC3 shape: {x.shape}")
        
        return(x)


# iteratively train the model (args, Net(), cpu/gpu, how to load trining data, optimizer, number of epochs/iterations) - first epoch index is 1
def train(args, model, device, train_loader, optimizer, epoch, loss_fxn, loss_dict):
    # set the model in training mode: Docs>torch.nn>Module. Training mode affects some modules differently than eval() (used for validation/testing), i.e. Dropout, where no channels/nodes would be zeroed out during testing.
    model.train()
    # iterate over minibatches
    for batch_idx, (data, target) in enumerate(train_loader):
        # send I/O data to cpu/gpu
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # clear gradients for all tensors
        output = model(data)  # calculate yhat
        # compute loss between (yhat,y).
        loss = loss_fxn(output, target)
        loss.backward()  # perform back prop 
        optimizer.step()  # update parameters
        # print training progress to console if there is no remainder in the division of batch_idx/args.log_interval
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
                
    loss_dict['train_loss'].append(float(loss))


# validate the trained model on the val set 
def val(model, device, val_loader, loss_fxn, loss_dict):
    model.eval()  # enter evaluation mode (no Dropout, batchnorm, others?)
    # initialize vars
    val_loss = 0  
    # the following syntax will run the model without tracking computations on each tensor - saves memory
    with torch.no_grad():
        # val one each sample/label pair in val set
        for data, target in val_loader:
            # send to cpu/gpu device
            data, target = data.to(device), target.to(device)
            val_feat = data.shape[1]
            output = model(data)#, in_channels=val_feat, out_channels=val_feat)  # compute yhat
            val_loss += float(F.mse_loss(output, target, reduction='sum').item())  # sum up batch loss

    # compute mean of val_loss that was iteratively summed in val loop over samples
    val_loss /= len(val_loader.dataset)
    loss_dict['val_loss'].append(float(val_loss))

    # display results on val set to console
    print('\nValidation set: Average loss: {:.6f}\n'.format(val_loss))

# save a checkpoint at the last epoch with minimum val loss    
def save_checkpoint(state, is_best, filename=str(output_path / 'GEOS-CF_checkpoint.pth.tar')):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, str(output_path / 'GEOS-CF_best.pth.tar'))


def main(train_x, train_y, val_x, val_y, args, kwargs):
    
 
    # prep iterable datasets for train/val
    print("####PREPARE TENSOR DATASETS")
    data1 = torch.utils.data.TensorDataset(train_x,train_y)
    data2 = torch.utils.data.TensorDataset(val_x,val_y)
    
    # load train/val data given keyword arguments
    train_loader = torch.utils.data.DataLoader(data1,**kwargs)
    kwargs.update({'batch_size': args.val_batch_size})
    val_loader = torch.utils.data.DataLoader(data2, **kwargs)
    
    # define model
    model = FC_model(in_shape=train_x.shape, out_shape=(0,1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
                     
    # optionally resume from a checkpoint
    if args.resume:  
        if os.path.isfile(args.checkpoint):
            print("=> loading best checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            args.start_epoch = checkpoint['epoch']
            min_val_loss = checkpoint['min_val_loss']
            GEOSFP2ERA5.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("=> starting training from scratch ... ")

    # define a dict to store loss during training
    loss_dict = {'epoch':[],'train_loss':[],'val_loss':[]}
    
    # start training timer
    start_time = time.time()
    
    print("\n****** TRAIN/VAL ****** \n")
    # for each epoch, train on training data, val trained model on val data
    for epoch in range(args.start_epoch, args.epochs + 1):
        loss_dict['epoch'].append(epoch)
        train(args, model, device, train_loader, optimizer, epoch, criterion, loss_dict)
        val(model, device, val_loader, criterion, loss_dict)
        
        # timer for training
        end_time = time.time() - start_time
        print('TRAINING TIME: ',end_time/60,' MINUTES')
        
        tmp = {
        'epoch': epoch,
        'train_mse': loss_dict['train_loss'][-1],
        'val_mse:': loss_dict['val_loss'][-1],
        'runtime': end_time/60
        }
        
        # write loss to file during training to monitor
        tmp_df = pd.DataFrame(tmp,index=[0])
        if not os.path.exists('monitor_loss.csv'):
            tmp_df.to_csv('monitor_loss.csv', mode='w+', index=False, header=True)
        else:
            tmp_df.to_csv('monitor_loss.csv', mode='a', index=False, header=False)
        
        # save checkpoint if the val loss is the min over time
        val_loss = loss_dict['val_loss']
        min_val_loss = np.nanmin(val_loss)
        is_best = min_val_loss <= val_loss[-1]
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': Gmodel.state_dict(),
            'min_val_loss': min_val_loss,
            'optimizer' : optimizer.state_dict()
        }, is_best)
        
        # if val loss has not improved within the last N epochs by delta, then exit the training loop
        # do this by comparing the mean of the last N epochs to the loss of the current epoch
        stop_intvl = 5
        delta = 0.0001
        if len(val_loss) <= stop_intvl:
            continue
        lastN = val_loss[-stop_intvl-1:-1]
        diff = np.abs(np.mean(lastN) - val_loss[-1])
        if diff <= delta:
            print('EARLY STOPPING - validation loss has not improved by less than '+ str(delta) +' in '+ str(stop_intvl) +' epochs')
            break
        
    # write loss dict to csv
    loss_df = pd.DataFrame(loss_dict)
    if not os.path.exists(str(output_path / 'loss.csv')):
        loss_df.to_csv('loss.csv', mode='w+', index=False, header=True)
    else:
        loss_df.to_csv('loss.csv', mode='a', index=False, header=False)
                
    
    


def get_data(fls=[], test=False):

    df = pd.concat([pd.read_parquet(f) for f in fls], ignore_index=True)
    
    feat_list = ["Jval", "Met", "ConcBeforeChem", "AeroArea"]
    pattern = "|".join(feat_list)
    features_df = df.filter(regex=pattern)
    targets_df = df["ConcAfterChem_CO"]
    
    print(features_df.head())
    print(targets_df.head())
    
    if test:
        features = torch.tensor(features.values, dtype=torch.float32)
        targets = torch.tensor(targets.values, dtype=torch.float32)
        return features, targets
    
    else:
        # 1. Split data
        X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
            features_df, targets_df, test_size=0.2, random_state=42
        )

        # 2. Compute normalization stats on training data only
        X_min = X_train_df.min()
        X_max = X_train_df.max()

        # 3. Apply Min-Max scaling
        X_train_scaled = (X_train_df - X_min) / (X_max - X_min)
        X_val_scaled = (X_val_df - X_min) / (X_max - X_min)  # Use train stats!

        # 4. Convert to PyTorch tensors
        X_train = torch.tensor(X_train_scaled.values, dtype=torch.float32)
        X_val   = torch.tensor(X_val_scaled.values, dtype=torch.float32)

        y_train = torch.tensor(y_train_df.values, dtype=torch.float32)
        y_val   = torch.tensor(y_val_df.values, dtype=torch.float32)
        
        return X_train, y_train, X_val, y_val









train_model = True
test_model = True

if train_model:
    # Training settings
    # argparse is a python module...I don't fully understand it, but this is how the hyperparameters are specified
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch-size', type=int, default=3200, metavar='N',
                        help='input batch size for training')  # if the batch size is larger than 1, we get an OOM at the first conv
    parser.add_argument('--val-batch-size', type=int, default=3200, metavar='N',
                        help='input batch size for testing/validation')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate ')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--checkpoint', default=str(output_path / 'GEOS-CF_checkpoint.pth.tar'), type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume', default=False, 
                    help='resume training from last checkpoint')
    args = parser.parse_args()  # input arguments
   
    use_cuda = not args.no_cuda and torch.cuda.is_available()  # use GPU or not

    torch.manual_seed(args.seed)  # set the random seed

    # device = torch.device("cuda" if use_cuda else "cpu")  # define gpu/cpu device
    device = torch.device("cuda")  # define gpu/cpu device

    # *args are non-keyword arguments, *kwargs are keyword arguments
    kwargs = {'batch_size': args.batch_size}
    # GPU settings
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )



    
    # read files specified for training
    with open("training_filenames.txt", "r") as f:
        train_fls = [line.strip() for line in f]
    #print(train_fls)
        
    train_X, train_Y, val_X, val_Y = get_data(fls=train_fls, test=False)
    
    print(train_X.shape, val_X.shape)
    
    # prepare dataloader, train, and validate
    main(train_X, train_Y, val_X, val_Y, args, kwargs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
