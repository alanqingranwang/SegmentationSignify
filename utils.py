"""
Utility functions for CSMPM
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import numpy as np
import os
import pickle
import glob
import myutils
import math

def normalize(arr):
    """Normalizes a batch of images into range [0, 1]"""
    if type(arr) is np.ndarray:
        if len(arr.shape) > 2:
            res = np.zeros(arr.shape)
            for i in range(len(arr)):
                res[i] = (arr[i] - np.min(arr[i])) / (np.max(arr[i]) - np.min(arr[i]))
            return res
        else:
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    else:
        if len(arr.shape) > 2:
            res = torch.zeros_like(arr)
            for i in range(len(arr)):
                res[i] = (arr[i] - torch.min(arr[i])) / (torch.max(arr[i]) - torch.min(arr[i]))
            return res
        else:
            return (arr - torch.min(arr)) / (torch.max(arr) - torch.min(arr))

######### Saving/Loading checkpoints ############
def load_checkpoint(model, path, optimizer=None, suppress=False):
    if not suppress:
        print('Loading checkpoint from', path)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer
    else:
        return model

def save_checkpoint(epoch, model_state, optimizer_state, loss, val_loss, model_folder, log_interval, scheduler=None):
    if epoch % log_interval == 0:
        state = {
            'epoch': epoch,
            'state_dict': model_state,
            'optimizer' : optimizer_state,
            'loss': loss,
            'val_loss': val_loss
        }
        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict(),

        filename = os.path.join(model_folder, 'model.{epoch:04d}.h5')
        torch.save(state, filename.format(epoch=epoch))
        print('Saved checkpoint to', filename.format(epoch=epoch))

def save_loss(epoch, loss, val_loss, model_path, name=None):
    if name is None:
        pkl_path = os.path.join(model_path, 'losses.pkl')
    else:
        pkl_path = os.path.join(model_path, '%s.pkl' % name)
    if os.path.exists(pkl_path):
        f = open(pkl_path, 'rb') 
        losses = pickle.load(f)
        train_loss_list = losses['loss']
        val_loss_list = losses['val_loss']

        if epoch-1 < len(train_loss_list):
            train_loss_list[epoch-1] = loss
            val_loss_list[epoch-1] = val_loss
        else:
            train_loss_list.append(loss)
            val_loss_list.append(val_loss)

    else:
        train_loss_list = []
        val_loss_list = []
        train_loss_list.append(loss)
        val_loss_list.append(val_loss)

    loss_dict = {'loss' : train_loss_list, 'val_loss' : val_loss_list}
    f = open(pkl_path,"wb")
    pickle.dump(loss_dict,f)
    f.close()
    print('Saved loss to', pkl_path) 

def add_bool_arg(parser, name, default=True):
    """Add boolean argument to argparse parser"""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})
