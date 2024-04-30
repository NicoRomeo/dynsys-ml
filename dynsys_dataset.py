"""Dataset class for dynamical system data

Defines a DynsysDataset class inherited from PyTorch's Dataset class to handle samplign from sets of dynamical system trajectories.

"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

class DynsysDataset(Dataset):
    def __init__(self, data_h5, transform=None, target_transform=None, stride=1):
        self.filename = data_h5 # filename
        with h5py.File(data_h5, 'r') as f:
            self.len = len(f.keys())
        self.file = h5py.File(data_h5, 'r')
        self.transform = transform
        self.target_transform = target_transform
        self.stride = stride

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        dset = self.file[str(idx)]
        params = dset.attrs["params"]
        label = params[-1]
        system = dset[:,::self.stride,:]
        if self.transform:
            system = self.transform(system)
        if self.target_transform:
            label = self.target_transform(label)
        return np.float32(system), label
