"""Dynamical system data generator

Generates trajectories for .
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
from tqdm import tqdm
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Integration')

def hopf(x0, T_samples, r):
    dy = np.zeros_like(x0)
    def func(t, y):
        dy[0] = r * y[0] - y[1] - y[0]**3 - y[0]*y[1]**2 
        dy[1] = y[0] + r * y[1] - y[0]**2*y[1] -  y[1]**3
        return dy
    def jac(t,y):
        return np.array([[r - 3.0 *y[0]**2 - y[1]**2, -1.0 - 2.0 *y[0]*y[1]],
                         [1.0 - 2.0*y[0]*y[1], r - y[0]**2 - 3.0*y[1]**2]])
    # be careful with integration t_eval...
    try:
        sol = scint.solve_ivp(func, (0, T_samples[-1]), x0, t_eval=T_samples, jac=jac, method='BDF', dense_output=False, max_step=1e-3)
        if not sol.success:
            print(sol.success)
        return sol.y.T, sol.success
    except RuntimeWarning:
        np.zeros(0), False
        
def danny(x0, T_samples, k, a):
    dy = np.zeros_like(x0)
    def func(t, y):
        dy[0] = k * y[0] + (1- a)*y[1] - y[0]**3 - y[0]*y[1]**2 
        dy[1] = (1 + a)*y[0] + k * y[1] - y[0]**2*y[1] -  y[1]**3
        return dy
    def jac(t,y):
        return np.array([[k - 3.0 *y[0]**2 - y[1]**2, 1.0-a - 2.0 *y[0]*y[1]],
                         [1.0+a - 2.0*y[0]*y[1], k - y[0]**2 - 3.0*y[1]**2]])
    # be careful with integration t_eval...
    try:
        sol = scint.solve_ivp(func, (0, T_samples[-1]), x0, t_eval=T_samples, jac=jac, method='BDF', dense_output=False, max_step=1e-3)
        if not sol.success:
            print(sol.success)
        return sol.y.T, sol.success
    except RuntimeWarning:
        np.zeros(0), False

def main_hopf(ins):
    #print('Integrating i={0}'.format(rank))
    seed, r = ins
    np.random.seed(seed)
    results = []
    trajs_i = []
    parameters =  r
    for j in range(num_IC):
        x0 = 2*(np.random.rand(2)-.5)
        traj, success = hopf(x0, times, r)
        if traj.shape[0] == len(times):
            traj[:,0] = traj[:,0]
            traj[:,1] = traj[:,1]
            trajs_i.append(traj)
    job_out = parameters, np.array(trajs_i)
    results.append(job_out)
    return results

def main_snic(ins):
    #print('Integrating i={0}'.format(rank))
    seed, k, a = ins
    np.random.seed(seed)
    results = []
    trajs_i = []
    parameters =  np.array([k,a])
    for j in range(num_IC):
        x0 = 2*(np.random.rand(2)-.5)
        traj, success = danny(x0, times, k,a)
        if traj.shape[0] == len(times):
            traj[:,0] = traj[:,0]
            traj[:,1] = traj[:,1]
            trajs_i.append(traj)
    job_out = parameters, np.array(trajs_i)
    results.append(job_out)
    return results

if __name__ == '__main__':
    num_systems = int(sys.argv[1])
    num_IC = int(sys.argv[2])
    n_jobs = int(sys.argv[3]) # number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

    n_cpu = n_jobs

    # ~ 600 systems /hour /core?

    t0 = time()
    print("+++++ N_CPU = {0} +++++".format(n_cpu))

    def make_randseed():
        return int(np.random.rand()*1e16 % 123456789)

    times = np.linspace(0, 10, 400)

    #pbar = tqdm(total=num_systems, file=open(os.devnull, 'w'))
    #LOG_INTERVAL = num_systems // 10
    #logger.info("###### Parsing ######")
    
    k_arr = np.hstack([np.linspace(-2, 2, num_systems), 2*np.ones(num_systems-1), 
                       np.linspace(2, -2, num_systems)[1:], -2*np.ones(num_systems-2)])
    a_arr = np.hstack([np.zeros(num_systems), np.linspace(0,2, num_systems)[1:],
                       2*np.ones(num_systems-1), np.linspace(2, 0, num_systems)[1:-1]])
    hopf_arr = np.linspace(-1, 1, num_systems)
    #logger.info(">> Parsing complete. << ")
    tasks_snic = [(make_randseed(), k, a ) for k,a in zip(k_arr, a_arr)]
    tasks_hopf = [(make_randseed(), r ) for r in hopf_arr]
    logger.info('###### START INTEGRATION HOPF ######')
    #results = [main(p) for i, p in enumerate(parameters)]
    t1 = time()
    with Pool(processes=n_cpu) as pool:
        #results = list(tqdm(pool.imap_unordered(main_snic, tuple(tasks_snic)), total=len(tasks_snic)))
        results = list(tqdm(pool.imap_unordered(main_hopf, tuple(tasks_hopf)), total=len(tasks_hopf)))
    results = [_i for temp in results for _i in temp]
    t2 = time()
    logger.info("Integration in {0:.2f} min".format((t2-t1)/60.))
    logger.info('___ saving to h5 ___')
    with h5py.File("datagen_hopf_pool_{0}.hdf5".format(num_systems), "w") as myh5:
        for _i,res in enumerate(results):
            param = res[0]
            arr = res[1]
            #print(arr)
            #print("idx ", _i, "; arr shape", arr.shape)
            dset = myh5.create_dataset(str(_i), data=arr)
            dset.attrs["params"] = param
    logger.info('###### all done. ######')