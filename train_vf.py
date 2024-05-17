"""
Trains a neural network on a dynamical system defined by its vector field 
through constrastive learning.

"""

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
from infoNCE import I_estimator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Training')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info("CUDA is available and is being used.")
else:
    logger.info("CUDA is unavailable. Reverting to CPU.")



def create_sample_grid(N0=6, k=40, scale=20):
    Rs = np.hstack([0, np.linspace(0.05, 0.5, N0), np.logspace(np.log10(0.5), np.log10(scale), k-N0)])
    pos = np.zeros((2**(N0+1)*k+1,2))
    pos[0,:] = 0
    label = 1
    for i in range(10*(N0-2)):
        if i <= N0-1:
            num_in_cur_layer = 2*int(2**((i+1)))
            t = np.arange(0, num_in_cur_layer)
            phi_i = np.pi/num_in_cur_layer*i
            pos[label:label+num_in_cur_layer,0]= Rs[i]* np.cos(2*np.pi*t/num_in_cur_layer + phi_i)
            pos[label:label+num_in_cur_layer,1]= Rs[i]* np.sin(2*np.pi*t/num_in_cur_layer + phi_i)
            label += num_in_cur_layer
        if i > N0-1:
            num_in_cur_layer = 2*int(2**(N0 - (i-N0+2)*0.1))
        if num_in_cur_layer >0:
            t = np.arange(0, num_in_cur_layer)

            phi_i = phi_i+(1+0)*np.pi/num_in_cur_layer*i
            pos[label:label+num_in_cur_layer,0]= Rs[i]* np.cos(2*np.pi*t/num_in_cur_layer + phi_i)
            pos[label:label+num_in_cur_layer,1]= Rs[i]* np.sin(2*np.pi*t/num_in_cur_layer + phi_i)
            label += num_in_cur_layer
    pos = pos[:label,:]
    return pos
    
    
def sample_sys():
    # culls systems that flow to infinity far from the origin
    a = 1*np.random.randn(9)
    b = 1*np.random.randn(9)
    # impose a stable cubic form rdot < 0 [ a generalization of (x^2+y^2)*[-A, - B; B, -C] @ (x, y) ]
    a[5] = -np.abs(a[5])
    #a[6] = -B1
    a[7] = -np.abs(a[7])
    #a[8] = -B2
    b[5] = -a[6] # B1
    b[6] = -np.abs(b[6]) #C1
    b[7] = - a[8] # B2
    b[8] = - np.abs(b[8]) #C2
    b = b/ np.abs(a[0])
    a = a/np.abs(a[0])
    # normalize fields X and Y n=such that a[5] = -1, b[8] = -1
    X = np.sqrt(np.abs(1/a[5]))
    Y = np.sqrt(np.abs(1/b[8]))
    # apply field normalization factor
    a = a * np.array([1, Y/X, X, Y, Y*Y/X, X**2, X*Y, Y*Y, Y**3/X])
    b = b * np.array([X/Y, 1, X*X/Y, X, Y, X**3/Y, X*X, X*Y, Y**2])
    return a, b, X, Y

def create_sample(grid_points):
    
    a,b, _,_ = sample_sys()
    
    def func(y):
        # note shape is transposed from usual
        dy = np.zeros_like(y)
        dy[0,:] = a[0]* y[0,:] + a[1] * y[1,:] + a[2]* y[0,:]**2 + a[3]* y[0,:]*y[1,:] + a[4]* y[1,:]**2  \
                + a[5]*y[0,:]**3 + a[6]*y[0,:]**2*y[1,:]+a[7]*y[0,:]*y[1,:]**2 + a[8]* y[1,:]**3
        dy[1,:] = b[0]*y[0,:] + b[1] * y[1,:] + b[2]* y[0,:]**2  + b[3]* y[0,:]*y[1,:] + b[4]* y[1,:]**2 \
                + b[5]*y[0,:]**3 + b[6]*y[0,:]**2*y[1,:]+b[7]*y[0,:]*y[1,:]**2 + b[8]* y[1,:]**3
        return dy
    #def jac(y):
    #    return np.array([[a[0] + 2*a[2]* y[0] + a[3]*y[1]
    #            + 3*a[5]*y[0]**2 + 2*a[6]*y[0] *y[1]+a[7]*y[1]**2,
    #            a[1] * y[1] + a[3]* y[0] + 2*a[4]* y[1]  + a[6]*y[0]**2 +2* a[7]*y[0]*y[1] + 3*a[8]* y[1]**2],
    #    [b[0] + 2*b[2]* y[0]  + b[3]*y[1] + 3*b[5]*y[0]**2 + 2*b[6]*y[0]*y[1]+b[7]*y[1]**2,
    #      b[1]  + b[3]* y[0] + 2*b[4]* y[1]  + b[6]*y[0]**2 + 2*b[7]*y[0]*y[1] + 3*b[8]* y[1]**2
    #    ]])
    
    ### first augmentation
    M = np.eye(2) + np.random.randn(2,2)
    Minv = M**(-1)
    yp1 = np.matmul(Minv, func(np.matmul(M, grid_points.T))).T
    #def augfun(y):
    #    return Minv @ func(M @ y)
    
    ### second augmentation            
    M = np.eye(2) + np.random.randn(2,2)
    Minv = M**(-1)
    yp2 = np.matmul(Minv, func(np.matmul(M, grid_points.T))).T
    return torch.from_numpy(np.float32(yp1)).to(device), torch.from_numpy(np.float32(yp2)).to(device)


if __name__ == '__main__':
    #####  TRAINING AND NET HYPERPARAMS ##### 
    n_epoch = int(5e4) # training iterations
    batch_size = 500 # samples per batch
    n_layers = 5 # number of DNN layers
    n_channels = 128  # embedding space dimension
    LR = 1e-3 # learning rate
    
    epoch_dec = int(np.log10(n_epoch))
    epoch_mant= round(10**(np.log10(n_epoch)-epoch_dec))
    batch_dec = int(np.log10(batch_size))
    batch_mant= round(10**(np.log10(batch_size)-batch_dec))
    cur_folder = "./vf0_b{5}e{6}_e{3}e{4}_lr{0}_d{1}_n{2}".format(int(-np.log10(LR)),
                                                                     n_layers,
                                                                     n_channels,
                                                                     epoch_mant,
                                                                     epoch_dec,
                                                                     batch_mant,
                                                                     batch_dec)
    
    ##### Create sampling grid ####
    points = create_sample_grid(N0=6, k=40, scale=20)
    logger.info("Grid shape: {0}x{1}".format(*points.shape))
    
    ##### CREATE MODEL ##### 
    size_sample = points.shape[0]*points.shape[1]
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
        list_x1s, list_x2s = [], []
        ## data generation+augmentations
        for sys in range(batch_size):
            x1, x2 = create_sample(points)
            # Standardisation
            x1 = (x1 - torch.mean(x1))/torch.std(x1)
            x2 = (x2 - torch.mean(x2))/torch.std(x2)
            list_x1s.append(x1)
            list_x2s.append(x2)
        x1s = torch.flatten(torch.stack(list_x1s, dim=0),start_dim=1).to(device)
        x2s = torch.flatten(torch.stack(list_x2s, dim=0),start_dim=1).to(device)
        
        
        ## feed data and backprop
        model.optimizer.zero_grad()
        loss, _, _, _ = model(x1s, x2s)
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