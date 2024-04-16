"""Dataset class for dynamical system data

"""

import os
import numpy as np
import h5py as h5
import torch
from torch.utils.data import Dataset

class DynsysDataset(Dataset):
    def __init__(self, data_h5, transform=None, target_transform=None):
        self.filename = data_h5 # filename
        with h5py.File(data_h5, 'r') as f:
            self.len = len(f.keys())
        self.file = h5py.File(data_h5, 'r')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        dset = self.file[str(idx)]
        params = dset.attr["params"]
        label = params[-1]
        system = dset[...]
        if self.transform:
            system = self.transform(system)
        if self.target_transform:
            label = self.target_transform(label)
        return system, label