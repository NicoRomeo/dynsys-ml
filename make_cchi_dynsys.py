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
    # be careful with integration t_eval...
    sol = scint.solve_ivp(func, (0, T_samples[-1]), x0, t_eval=T_samples, method='RK45', dense_output=False, max_step=1e-3)
    return sol.y.T

def parse_parameters(txtstring):
    split_str = txtstring.split(" ")
    a_str = split_str[:8]
    b_str = split_str[8:]
    a = np.array([1.0]+[float(p) for p in a_str])
    b = np.array([float(p) for p in b_str])
    return a, b


def main(a, b, times, seed):
    #print('Integrating i={0}'.format(rank))
    np.random.seed(seed)
    results = []
    trajs_i = []
    parameters =  np.vstack([a,b])
    for j in range(num_IC):
        x0 = 2*(np.random.rand(2)-.5)
        traj = dynsys_models(x0, times, a, b)
        if traj.shape[0] == len(times):
            trajs_i.append(traj)
    job_out = parameters, np.array(trajs_i)
    results.append(job_out)
    return results


if __name__ == '__main__':
    
    # get number of samples to generate 
    # create an ensemble of FhN systems
    num_systems = int(sys.argv[1])
    num_IC = int(sys.argv[2])
    
    #n_cpu = cpu_count()
    n_jobs = int(sys.argv[3]) # number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    n_cpu = n_jobs
    
    # ~ 600 systems /hour /core?
    
    t0 = time()
    print("+++++ N_CPU = {0} +++++".format(n_cpu))
    
    def make_randseed():
        return int(np.random.rand()*1e16 % 123456789)

    times = np.linspace(0, 1, 400)
    
    #load data from Chris
    file_path = "cchi_dynsys.txt"
    with open(file_path, "rb") as f:
        num_lines = sum(1 for _ in f)
    if num_systems == -1 or num_systems > num_lines:
        num_systems = num_lines
        
    pbar = tqdm(total=num_systems, file=open(os.devnull, 'w'))
    LOG_INTERVAL = num_systems // 10
    logger.info("###### Parsing ######")
    parameters = []
    with open(file_path, "r") as file:
        i = 0
        for line in file:
            a,b = parse_parameters(line)
            parameters.append( (a,b, times, make_randseed()))
            i += 1
            if i == num_systems:
                break
            # progress update
            #if pbar.n % LOG_INTERVAL == 0:
            #    pbar.set_description(f'Parsing :>')
            #logger.info(str(pbar))
            #pbar.update(1)
        
    #pbar.set_description(f'Parsing :>')
    #logger.info(str(pbar))
    #pbar.update(1)
    logger.info(">> Parsing complete. << ")
    
    logger.info('###### START INTEGRATION ######')
    #results = [main(p) for i, p in enumerate(parameters)]
    t1 = time()
    with Pool(processes=n_cpu) as pool:
        results = pool.starmap(main, tuple(parameters))
    results = [_i for temp in results for _i in temp]
    t2 = time()
    logger.info("Integration in {0:.2f} min".format((t2-t1)/60.))
    logger.info('___ saving to h5 ___')
    with h5py.File("datagen_cchi_pool_{0}.hdf5".format(num_systems), "w") as myh5:
        for _i,res in enumerate(results):
            param = res[0]
            arr = res[1]
            #print(arr)
            #print("idx ", _i, "; arr shape", arr.shape)
            dset = myh5.create_dataset(str(_i), data=arr)
            dset.attrs["params"] = param
    logger.info('###### all done. ######')