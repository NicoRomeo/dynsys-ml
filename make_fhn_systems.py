"""Dynamical system data generator

Generates trajectories from FitzHugh-Nagumo dynamical systems.
Takes in as argument the filename of output, number of dynamical systems to be generated


"""


from scipy.io import loadmat
import numpy as np
import scipy.integrate as scint
import os
import sys
import h5py
from multiprocessing import Pool, cpu_count
from time import time

def fhn_models(x0, T_samples, a,b, I, tau):
    dx = np.zeros_like(x0)
    def func(t,x):
        dx[0] =  x[0] *( 1.0 - x[0]**2 / 3. ) - x[1] + I
        dx[1] = (x[0] + a - b * x[1]) /tau
        return dx
    sol = scint.solve_ivp(func, (0, T_samples[-1]), x0, t_eval=T_samples, dense_output=False, max_step=1e-2)
    return sol.y.T


def main(a, b, tau, I, label):
    #print('Integrating i={0}'.format(rank))
    results = []
    #a, b, tau, I, _ = parameters
    parameters = a, b, tau, I, label
    trajs_i = []
    for j in range(num_IC):
        x0 = 2*(np.random.rand(2)-.5)
        trajs_i.append(fhn_models(x0, times, a, b, I, tau))
    job_out = parameters, np.array(trajs_i)
    results.append(job_out)
    return results


if __name__ == '__main__':
    
    # get number of samples to generate 
    # create an ensemble of FhN systems
    num_perclass = int(sys.argv[1])
    num_systems = 4* int(sys.argv[1])
    num_IC = int(sys.argv[2])
    
    #n_cpu = cpu_count()
    n_jobs = int(sys.argv[3])
    n_cpu = n_jobs
    
    t0 = time()
    print("+++++ N_CPU = {0} +++++".format(n_cpu))

    times = np.linspace(0, 50, 100)

    # generate systems
    a_list = np.random.rand(num_systems)
    b_list = np.random.rand(num_systems)
    tau_list = 20*np.random.rand(num_systems)
    I_list = 0.3*np.random.randn(num_systems)+0.05

    labels_list = np.zeros(num_systems)
    label_dict = {0:'MS-NE', 1:'LC', 2:'MS-E', 3:'BS'}
    # make a few monostables - non excitable: tau small, b small: label 0
    a_list[0:num_perclass] = 0.2*(1+0.2*np.random.randn(num_perclass))
    b_list[0:num_perclass] = 0.5*(1+0.2*np.random.randn(num_perclass))
    tau_list[0:num_perclass] = 0.125*(1+0.2*np.random.randn(num_perclass))
    I_list[0:num_perclass] = 0.5
    labels_list[0:num_perclass] = 0

    # make a few limit cycles - excitable: tau bigger, b small
    a_list[num_perclass:2*num_perclass] = 0.2*(1+0.2*np.random.randn(num_perclass))
    b_list[num_perclass:2*num_perclass] = 0.5*(1+0.2*np.random.randn(num_perclass))
    tau_list[num_perclass:2*num_perclass] = 1*(1+0.2*np.random.randn(num_perclass))
    I_list[num_perclass:2*num_perclass] = 0.5
    labels_list[num_perclass:2*num_perclass] = 1

    # make a few monostable - excitable: tau bigger, b large, a and I s.t. only 1 fp
    a_list[2*num_perclass:3*num_perclass] = 0.8*(1.0 + 0.2*np.random.randn(num_perclass))
    b_list[2*num_perclass:3*num_perclass] = 1.0*(1.0 + 0.2*np.random.randn(num_perclass))
    tau_list[2*num_perclass:3*num_perclass] = 12.5*(1.0 + 0.2*np.random.randn(num_perclass))
    I_list[2*num_perclass:3*num_perclass] = 0.1
    labels_list[2*num_perclass:3*num_perclass] = 2

    # make a few bistable: tau bigger, b large, a and I s.t. 3 fp
    a_list[3*num_perclass:4*num_perclass] = 0.8*(1.0 + 0.2*np.random.randn(num_perclass))
    b_list[3*num_perclass:4*num_perclass] = 2.0*(1.0 + 0.2*np.random.randn(num_perclass))
    tau_list[3*num_perclass:4*num_perclass] = 12.5*(1.0 + 0.2*np.random.randn(num_perclass))
    I_list[3*num_perclass:4*num_perclass] = 0.5
    labels_list[3*num_perclass:4*num_perclass] = 3

    parameters = [ (a_list[_i], b_list[_i], tau_list[_i], I_list[_i], labels_list[_i]) for _i in range(num_systems)]
    t1 = time()
    print("start-up in {0:.2f} s".format(t1-t0))
    print('###### START INTEGRATION ######')
    #results = [main(p) for i, p in enumerate(parameters)]
    with Pool(processes=n_cpu) as pool:
        results = pool.starmap(main, tuple(parameters))
    results = [_i for temp in results for _i in temp]
    t2 = time()
    print("Integration in {0:.2f} min".format((t2-t1)/60.))
    print('___ saving to h5 ___')
    with h5py.File("datagen_fhn_pool_{0}.hdf5".format(num_perclass), "w") as myh5:
        for _i,res in enumerate(results):
            param = res[0]
            arr = res[1]
            dset = myh5.create_dataset(str(_i), data=arr)
            dset.attrs["params"] = param
    print('###### all done. ######')