"""
Some analysis tools to study the learned representation of dynamical systems

"""

import numpy as np
from sklearn import decomposition, preprocessing
from sklearn import manifold
from sklearn.decomposition import PCA
import h5py
import torch


def load_model(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

def get_dist_matrix(filename, model, num_reads):
    with h5py.File(filename, 'r') as f:
        num_systems = len(f.keys())
        num_perclass = num_systems // 4
        dists = np.zeros((4*num_reads, 4*num_reads))
        list_x1s = []
        list_x2s = []
        parameters = []
        for j in range(4):
            for i in range(num_reads):
                idx = str(j*num_perclass + i)
                xi = torch.tensor(np.array(f[idx], dtype=np.float32))
                parameters.append(f[idx].attrs["params"])
                list_x1s.append(xi)
        x1 = torch.flatten(torch.stack(list_x1s, dim=0),start_dim=1)
        
        _, _, _, f  = model(x1, x1)
        return f.detach().numpy(), parameters

def PCA_on_dists(dists):
    dists_scaled = preprocessing.scale(dists, axis=1)
    pca = decomposition.PCA(n_components=7)
    pca.fit(dists_scaled)
    X_pca = pca.transform(dists_scaled)
    return X_pca, pca.explained_variance_ratio_, pca

def get_latent(filename, model, num_reads):
    with h5py.File(filename, 'r') as f:
        num_systems = len(f.keys())
        num_perclass = num_systems // 4
        dists = np.zeros((4*num_reads, 4*num_reads))
        list_x1s = []
        list_x2s = []
        parameters = []
        for j in range(4):
            for i in range(num_reads):
                idx = str(j*num_perclass + i)
                xi = torch.tensor(np.array(f[idx], dtype=np.float32))
                parameters.append(f[idx].attrs["params"])
                list_x1s.append(xi)
        x1 = torch.flatten(torch.stack(list_x1s, dim=0),start_dim=1)
        
        _, enc1, _, f  = model(x1, x1)
        return enc1.detach().numpy(), parameters
    
def PCA_on_latent(latents):
    latents_scaled = preprocessing.scale(latents, axis=1)
    pca = decomposition.PCA(n_components=7)
    pca.fit(latents_scaled)
    X_pca = pca.transform(latents_scaled)
    return X_pca, pca.explained_variance_ratio_, pca