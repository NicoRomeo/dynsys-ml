from scipy.io import loadmat
import numpy as np
import scipy.integrate as scint
import os
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributions as D
from tqdm import tqdm
from dynsys_dataset import DynsysDataset
from infoNCE import I_estimator
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Training')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info("CUDA is available and is being used.")
else:
    logger.info("CUDA is unavailable. Reverting to CPU.")

def rotate_sol_tensor(trajectories, angle):
    # trajectories has size (num trajectories) x (num sample points) x 2
    R = torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]],
                     dtype=trajectories.dtype,
                     device=device
                    ).T
    return torch.tensordot(trajectories, R, dims=([2],[1]))

def rescale_sol_tensor(trajs, scales):
    # trajectories has size (num trajectories) x (num sample points) x 2
    R = torch.tensor([[1. + scales[0], 0.],[0., 1. + scales[1]]],
                     dtype=trajs.dtype,
                    device=device).T
    return torch.tensordot(trajs, R, dims=([2],[1]))

def translate_sol_tensor(trajectories, shift):
    transl = torch.zeros_like(trajectories)
    transl[:,:,0] = trajectories[:,:,0] + shift[0]
    transl[:,:,1] = trajectories[:,:,1] + shift[1]
    return transl


def shuffleIC(traj):
    return traj[torch.randperm(traj.size()[0]), :, :]

def augment_linear(traj):
    scales = torch.randn(2, device=device)
    transl = torch.randn(2, device=device)
    angles = 2*np.pi*torch.rand(1, device=device)
    return shuffleIC(translate_sol_tensor(rescale_sol_tensor(rotate_sol_tensor(traj, angles), scales), transl))

def deform(traj, N_anchors=10, s_anchor=0.1, w_anchor=1.0):
    anchors_strength = s_anchor* torch.randn(N_anchors, 2, device=device)
    anchors_pos = 6*(torch.rand(N_anchors, 2, device=device)-0.5)
    anchors_widths = w_anchor*(1.0+0.05*torch.randn(N_anchors, 2))
    def fun(x,y):
        dx = torch.stack(
            [anchors_strength[i,0] * torch.exp(
                  -(x-anchors_pos[i,0])**2/(2.*anchors_widths[i,0]**2)
                  -(y-anchors_pos[i,1])**2/(2.*anchors_widths[i,1]**2)
                                              ) for i in range(N_anchors)], dim=0)
        dy = torch.stack(
            [anchors_strength[i,1] * torch.exp(
                  -(x-anchors_pos[i,0])**2/(2.*anchors_widths[i,0]**2)
                  -(y-anchors_pos[i,1])**2/(2.*anchors_widths[i,1]**2)
                                              ) for i in range(N_anchors)], dim=0)
        xnew = x + torch.sum(dx, dim=0)
        ynew = y + torch.sum(dy, dim=0)
        return torch.stack([xnew, ynew], dim=-1)
    return fun(traj[:,:,0], traj[:,:,1])

if __name__ == '__main__':
    #####  TRAINING AND NET HYPERPARAMS ##### 
    n_epoch = int(5e4) # training iterations
    batch_size = 300 # samples per batch
    n_layers = 3 # number of DNN layers
    n_channels = 128  # embedding space dimension
    LR = 1e-4 # learning rate
    
    epoch_dec = int(np.log10(n_epoch))
    epoch_mant= round(np.log10(n_epoch)-epoch_dec)
    cur_folder = "./gen2e4_b3e2_e{3}e{4}_lr{0}_d{1}_n{2}_lin".format(int(-np.log10(LR)),
                                                                     n_layers,
                                                                     n_channels,
                                                                     epoch_mant,
                                                                     epoch_dec)
    
    ##### NON-LINEAR AUGMENTATION PARAMETERS ##### 
    N_anchors = 10 # number of deformation nodes
    s_anchors = 0.1 # amplitude of deformation. should stay < 1 for good behavior
    w_anchor = 1.0 # width of deformation bumps
    ratio_deformed = 0. # ratio of samples in a batch going through NL transform

    #####  load data ##### 
    filename = "datagen_generic_pool_20000.hdf5"
    dataset = DynsysDataset(filename, transform=None, stride=4)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    sampled_sys, _ = next(iter(dataloader))

    ##### CREATE MODEL ##### 
    size_sample = np.prod(sampled_sys[0].shape)
    logger.info("Encoder dimensions | {0}x{1} input  >> [{2} layers] >> {3}-dim embedding".format(
        batch_size, size_sample, n_layers, n_channels)
               )

    model = I_estimator(input_shapes=(size_sample , size_sample),
                        n_layers=n_layers,
                        n_channels=n_channels,
                        LR=LR
                       ) #LR=1e-3
    model.to(device)

    ##### LOGGING ##### 
    pbar = tqdm(total=n_epoch, file=open(os.devnull, 'w'))
    LOG_INTERVAL = n_epoch // 100

    ##### BEGIN TRAINING LOOP ##### 
    losses = []
    rng = np.random.default_rng()
    for i in range(n_epoch):
        ## sample batch_size dynamical systems
        sampled_sys_cpu, _ = next(iter(dataloader))
        sampled_sys = sampled_sys_cpu.to(device)
        list_x1s, list_x2s = [], []
        ## data augmentations
        # this loop could be parallelized easily
        for sys in range(sampled_sys.shape[0]):
            x1 = augment_linear(torch.squeeze(sampled_sys[sys,:,:,:]))  
            x2 = augment_linear(torch.squeeze(sampled_sys[sys,:,:,:]))
            ##### NON-LINEAR AUGMENTATIONS
            if torch.rand(1) < ratio_deformed:
                x1 = deform(x1)
                x2 = deform(x2)
            # Standardisation
            x1 = (x1 - torch.mean(x1))/torch.std(x1)
            x2 = (x2 - torch.mean(x2))/torch.std(x2)
            list_x1s.append(x1)
            list_x2s.append(x2)
        x1 = torch.flatten(torch.stack(list_x1s, dim=0),start_dim=1)#.to(device)
        x2 = torch.flatten(torch.stack(list_x2s, dim=0),start_dim=1)#.to(device)
        
        ## feed data and backprop
        model.optimizer.zero_grad()
        loss, _, _, _ = model(x1, x2)
        loss.backward()
        model.optimizer.step()
        # progress update
        if pbar.n % LOG_INTERVAL == 0:
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization(device=device)
                pbar.set_description(f'Loss:\t{-loss.detach().cpu().numpy():0.2f}, gpu:\t{gpu_usage:0.2f}')
            else:
                pbar.set_description(f'Loss:\t{-loss.detach().cpu().numpy():0.2f}')
            logger.info(str(pbar))
        losses.append(loss.detach().cpu().numpy())
        pbar.update(1)
    ##### END OF TRAINING OPERATIONS ##### 
    ## Final progress bar update
    if torch.cuda.is_available():
        gpu_usage = torch.cuda.utilization(device=device)
        pbar.set_description(f'Loss:\t{-loss.detach().cpu().numpy():0.2f}, gpu:\t{gpu_usage:0.2f}')
    else:
        pbar.set_description(f'Loss:\t{-loss.detach().cpu().numpy():0.2f}')
    logger.info(str(pbar))
    ## save model
    losses = np.array(losses)
    if not os.path.isdir(cur_folder):
        os.makedirs(cur_folder)
    torch.save(losses, cur_folder + '/loss.pt')
    torch.save(model, cur_folder + '/model.pt')
    ##### EOF ##### 
