"""
Training loop for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import model, data, metrics, utils
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sys
import os
import argparse
import time
from pprint import pprint
import json

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Image Segmentation for Signify')
        # data 
        self.add_argument('--data-root', type=str, default='./data/', help='directory to dataset root')
        self.add_argument('--transform', type=str, default='four_crop', choices=['four_crop', 'center_crop'])

        self.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
        self.add_argument('--models_dir', default='./out/', type=str, help='directory to save models')
	
        self.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.add_argument('--epochs', type=int, default=100, help='Total training epochs')
        self.add_argument('--load_checkpoint', type=int, default=0, help='Load checkpoint at specificed epoch')
        self.add_argument('--log_interval', type=int, default=1, help='Frequency of logs')
        self.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
        self.add_argument('--unet_hidden', type=int, default=64)

    def parse(self):
        args = self.parse_args()
        date = '{}'.format(time.strftime('%b_%d'))
        args.run_dir = os.path.join(args.models_dir, args.filename_prefix, date, \
            f'nh{args.unet_hidden}')
        args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')

        model_folder = args.ckpt_dir
        if not os.path.isdir(model_folder):   
            os.makedirs(model_folder)
        args.filename = model_folder

        print('Arguments:')
        pprint(vars(args))

        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)

        return args

def trainer(args):
    """Training loop. 

    Handles model, optimizer, loss, and sampler generation.
    Handles data loading. Handles i/o and checkpoint loading.
        

    Parameters
    ----------
    conf : dict
        Miscellaneous parameters
    """

    ###############  Dataset ########################
    loader = data.load_denoising(args.data_root, train=True, 
        batch_size=args.batch_size, transform=None)

    val_loader = data.load_denoising(args.data_root, train=False, 
        batch_size=args.batch_size, transform=None)
    ##################################################

    ##### Model, Optimizer, Loss ############
    num_classes = 30
    network = model.Unet(args.unet_hidden, num_classes).to(args.device)

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    ##################################################

    ############## Training loop #####################
    for epoch in range(args.load_checkpoint+1, args.epochs+1):
        print('\nEpoch %d/%d' % (epoch, args.epochs))
    
        # Train
        network, optimizer, train_epoch_loss, train_epoch_dice = train(network, loader, criterion, optimizer, args.device)
        # Validate
        network, val_epoch_loss, val_epoch_dice = validate(network, val_loader, criterion, args.device)

        # Save checkpoints
        utils.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), \
                train_epoch_loss, val_epoch_loss, args.filename, args.log_interval)
        utils.save_loss(epoch, train_epoch_loss, val_epoch_loss, args.run_dir)
        utils.save_loss(epoch, train_epoch_dice, val_epoch_dice, args.run_dir, 'dice')
        print("Epoch {}: test loss: {:.6f}, test dice: {:.6f}".format(epoch, val_epoch_loss, val_epoch_dice))

def train(network, dataloader, criterion, optimizer, device):
    """Train for one epoch

        Parameters
        ----------
        network : UNet
            Main network and hypernetwork
        dataloader : torch.utils.data.DataLoader
            Training set dataloader
        optimizer : torch.optim.Adam
            Adam optimizer

        Returns
        ----------
        network : UNet
            Main network and hypernetwork
        optimizer : torch.optim.Adam
            Adam optimizer
        epoch_loss : float
            Loss for this epoch
    """
    network.train()

    epoch_loss = 0
    epoch_samples = 0
    epoch_dice = 0

    for batch_idx, (img, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img, label = img.to(device), label.to(device).long()
        num_classes = len(torch.unique(label[0]))
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            pred = network(img)
            loss = criterion(pred.squeeze(), label.squeeze())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.cpu().numpy()
            dices = 0
            for i in range(len(label)):
                _,_,_,dice = metrics.eval_metrics(label[i], pred[i], 30, device)
                print(dice)
                dices += dice
            epoch_dice += dices.data.cpu().numpy()

        epoch_samples += len(pred)
    epoch_loss /= epoch_samples
    epoch_dice /= epoch_samples
    return network, optimizer, epoch_loss, epoch_dice

def validate(network, dataloader, criterion, device):
    """Validate for one epoch

        Parameters
        ----------
        network : regagcsmri.UNet
            Main network and hypernetwork
        dataloader : torch.utils.data.DataLoader
            Training set dataloader

    """
    network.eval()

    epoch_loss = 0
    epoch_samples = 0
    epoch_dice = 0

    for batch_idx, (img, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img, label = img.to(device), label.to(device).long()
        with torch.set_grad_enabled(False):
            pred = network(img)
            loss = criterion(pred, label.squeeze())

            epoch_loss += loss.data.cpu().numpy()
            dices = 0
            for i in range(len(label)):
                num_classes = len(torch.unique(label[i]))
                _,_,_,dice = metrics.eval_metrics(label[i], pred[i], 30, device)
                dices += dice
            epoch_dice += dices.data.cpu().numpy()

        epoch_samples += len(pred)
    epoch_loss /= epoch_samples
    epoch_dice /= epoch_samples
    return network, epoch_loss, epoch_dice


if __name__ == "__main__":
    ############### Argument Parsing #################
    args = Parser().parse()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        sys.exit('No GPU detected')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    ##################################################

    trainer(args)
