from scipy.io import loadmat
import numpy as np
import scipy.integrate as scint
import os
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
logger = logging.getLogger('tqdm_logger')


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

def rotate_sol_tensor(trajectories, angle):
    # trajectories has size (num trajectories) x (num sample points) x 2
    R = torch.tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=trajectories.dtype).T
    return torch.tensordot(trajectories, R, dims=([2],[1]))

def rescale_sol_tensor(trajs, scales):
    # trajectories has size (num trajectories) x (num sample points) x 2
    R = torch.tensor([[1. + scales[0], 0.],[0., 1. + scales[1]]], dtype=trajs.dtype).T
    return torch.tensordot(trajs, R, dims=([2],[1]))

def translate_sol_tensor(trajectories, shift):
    transl = torch.zeros_like(trajectories)
    transl[:,:,0] = trajectories[:,:,0] + shift[0]
    transl[:,:,1] = trajectories[:,:,1] + shift[1]
    return transl


def shuffleIC(traj):
    return traj[torch.randperm(traj.size()[0]), :, :]

def augment_linear(traj):
    scales = torch.randn(2)
    transl = torch.randn(2)
    angles = 2*np.pi*torch.rand(1)
    return shuffleIC(translate_sol_tensor(rescale_sol_tensor(rotate_sol_tensor(traj, angles), scales), transl))


if __name__ == '__main__':
    n_epoch = int(2e4)
    batch_size = 50
    cur_folder = "./test100"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info("CUDA is available and is being used.")
    else:
        logger.info("CUDA is unavailable. Reverting to CPU.")

    # load data
    filename = "datagen_fhn_pool_100.hdf5"
    dataset = DynsysDataset(filename, transform=None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    sampled_sys, _ = next(iter(dataloader))


    size_sample = np.prod(sampled_sys[0].shape)

    model = I_estimator(input_shapes=(size_sample , size_sample), n_layers=12, n_channels=128, LR=1e-4) #LR=1e-3
    model.to(device) #'cuda:0'


    pbar = tqdm(total=n_epoch, file=open(os.devnull, 'w'))
    LOG_INTERVAL = n_epoch // 20

    losses = []
    rng = np.random.default_rng()
    for i in range(n_epoch):
        # sample batch_size dynamical system
        sampled_sys, _ = next(iter(dataloader))
        list_x1s, list_x2s = [], []
        for sys in range(sampled_sys.shape[0]):
            x1 = augment_linear(torch.squeeze(sampled_sys[sys,:,:,:])).to(device)
            x2 = augment_linear(torch.squeeze(sampled_sys[sys,:,:,:])).to(device)
            list_x1s.append(x1)
            list_x2s.append(x2)
        x1 = torch.stack(list_x1s, dim=0)
        x2 = torch.stack(list_x2s, dim=0)
        
        # feed data and backprop
        model.optimizer.zero_grad()
        loss, _, _, _ = model(x1, x2)
        loss.backward()
        model.optimizer.step()

        if pbar.n % LOG_INTERVAL == 0:
            pbar.set_description(f'Loss:\t{-loss.detach().cpu().numpy():0.2f}')
            logger.info(str(pbar))
        losses.append(loss.detach().cpu().numpy())

    ## save model
    losses = torch.tensor(losses)
    if not os.path.isdir(cur_folder):
        os.makedirs(cur_folder)
    torch.save(losses, cur_folder + '/loss.pt')
    torch.save(model, cur_folder + '/model.pt')