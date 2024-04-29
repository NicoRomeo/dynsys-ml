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

def dynsys_models(x0, T_samples, a,b,):
    dy = np.zeros_like(x0)
    def func(t, y):
        dy[0] = a[0]* y[0] + a[1] * y[1] + a[2]* y[0]**2 + a[3]* y[0]*y[1] + a[4]* y[1]**2  \
                + a[5]*y[0]**3 + a[6]*y[0]**2*y[1]+a[7]*y[0]*y[1]**2 + a[8]* y[1]**3
        dy[1] = b[0]*y[0] + b[1]* y[1] + b[2]* y[0]**2  + b[3]* y[0]*y[1] + b[4]* y[1]**2 \
                + b[5]*y[0]**3 + b[6]*y[0]**2*y[1]+b[7]*y[0]*y[1]**2 + b[8]* y[1]**3
        return dy
    def jac(t,y):
        return np.array([[a[0] + 2*a[2]* y[0] + a[3]*y[1]
                + 3*a[5]*y[0]**2 + 2*a[6]*y[0] *y[1]+a[7]*y[1]**2,
                a[1] * y[1] + a[3]* y[0] + 2*a[4]* y[1]  + a[6]*y[0]**2 +2* a[7]*y[0]*y[1] + 3*a[8]* y[1]**2],
        [b[0] + 2*b[2]* y[0]  + b[3]*y[1] + 3*b[5]*y[0]**2 + 2*b[6]*y[0]*y[1]+b[7]*y[1]**2,
          b[1]  + b[3]* y[0] + 2*b[4]* y[1]  + b[6]*y[0]**2 + 2*b[7]*y[0]*y[1] + 3*b[8]* y[1]**2
        ]])
    # be careful with integration t_eval...
    try:
        sol = scint.solve_ivp(func, (0, T_samples[-1]), x0, t_eval=T_samples, jac=jac, method='BDF', dense_output=False, max_step=1e-3)
        if not sol.success:
            print(sol.success)
        return sol.y.T, sol.success
    except RuntimeWarning:
        np.zeros(0), False

def sample_sys():
    # culls systems that flow to infinity far from the origin
    a = 1*np.random.randn(9)
    b = 1*np.random.randn(9)
    # impose a stable cubic form rdot < 0 [ a generalization of (x^2+y^2)*[-A, - B; B, -C] @ (x, y) ]
    #A1 = np.abs(np.random.randn())
    #A2 = np.abs(np.random.randn())
    #B1 = np.random.randn()
    #B2 = np.random.randn()
    #C1 = np.abs(np.random.randn())
    #C2 = np.abs(np.random.randn())
    # normalize timescale
    a =  a/np.abs(a[0])
    b = b/np.abs(a[0])
    # impose a stable cubic form rdot < 0 [ a generalization of (x^2+y^2)*[-A, - B; B, -C] @ (x, y) ]
    a[5] = -np.abs(a[5])
    #a[6] = -B1
    a[7] = -np.abs(a[7])
    #a[8] = -B2
    b[5] = -a[6] # B1
    b[6] = -np.abs(b[6]) #C1
    b[7] = - a[8] # B2
    b[8] = - np.abs(b[8]) #C2
    # normalize fields X and Y n=such that a[5] = -1, b[8] = -1
    X = np.sqrt(np.abs(1/a[5]))
    Y = np.sqrt(np.abs(1/b[8]))
    # apply field normalization factor
    a = a * np.array([1, Y/X, X, Y, Y*Y/X, X**2, X*Y, Y*Y, Y**3/X])
    b = b * np.array([X/Y, 1, X*X/Y, X, Y, X**3/Y, X*X, X*Y, Y**2])
    return a, b, X, Y

def main(seed):
    #print('Integrating i={0}'.format(rank))
    np.random.seed(seed)
    results = []
    trajs_i = []
    a, b, X, Y = sample_sys()
    parameters =  np.vstack([a,b])
    for j in range(num_IC):
        x0 = 2*(np.random.rand(2)-.5)
        x0[0] /= X
        x0[1] /= Y
        traj, success = dynsys_models(x0, times, a, b)
        if traj.shape[0] == len(times):
            traj[:,0] = X*traj[:,0]
            traj[:,1] = Y*traj[:,1]
            trajs_i.append(traj)
    job_out = parameters, np.array(trajs_i)
    results.append(job_out)
    return results


if __name__ == '__main__':
    
    # get number of samples to generate 
    # create an ensemble of FhN systems
    num_systems = int(sys.argv[1])
    num_IC = int(sys.argv[2])
    n_jobs = int(sys.argv[3]) # number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

    # ~ 120 systems /hour /core?
    
    t0 = time()
    print("+++++ N_CPU = {0} +++++".format(n_jobs))
    
    def make_randseed():
        return int(np.random.rand()*1e16 % 123456789)

    times = np.linspace(0, 10, 400)


    task_seeds = [make_randseed() for _ in range(num_systems)]
    logger.info('###### START INTEGRATION ######')
    t1 = time()
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap_unordered(main, tuple(task_seeds)), total=len(task_seeds), miniters=int(len(task_seeds)/100)))
    results = [_i for temp in results for _i in temp]
    t2 = time()
    logger.info("Integration in {0:.2f} min".format((t2-t1)/60.))
    logger.info('___ saving to h5 ___')
    with h5py.File("datagen_generic_pool_{0}.hdf5".format(num_systems), "w") as myh5:
        for _i,res in enumerate(results):
            param = res[0]
            arr = res[1]
            #print(arr)
            #print("idx ", _i, "; arr shape", arr.shape)
            dset = myh5.create_dataset(str(_i), data=arr)
            dset.attrs["params"] = param
    logger.info('###### all done. ######')