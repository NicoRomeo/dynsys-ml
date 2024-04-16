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

from mpi4py import MPI
# Use default communicator. No need to complicate things.
COMM = MPI.COMM_WORLD

def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

def fhn_models(x0, T_samples, a,b, I, tau):
    dx = np.zeros_like(x0)
    def func(t,x):
        dx[0] =  x[0] *( 1.0 - x[0]**2 / 3. ) - x[1] + I
        dx[1] = (x[0] + a - b * x[1]) /tau
        return dx
    sol = scint.solve_ivp(func, (0, T_samples[-1]), x0, t_eval=T_samples, dense_output=False, max_step=1e-2)
    return sol.y.T


def main(parameters):
        # Collect whatever has to be done in a list. Here we'll just collect a list of
    # numbers. Only the first rank has to do this.
    if COMM.rank == 0:
        # Split into however many cores are available.
        jobs = split(parameters, COMM.size)
    else:
        jobs = None

    # Scatter jobs across cores.
    jobs = COMM.scatter(jobs, root=0)
    print('###### START INTEGRATION ######')
    # Now each rank just does its jobs and collects everything in a results list.
    # Make sure to not use super big objects in there as they will be pickled to be
    # exchanged over MPI.
    results = []
    for job in jobs:
        # Do something meaningful here...
        a, b, tau, I, _ = job
        trajs_i = []
        for j in range(num_IC):
            x0 = 2*(np.random.rand(2)-.5)
            trajs_i.append(fhn_models(x0, times, a, b, I, tau))
        job_out = job, np.array(trajs_i)
        results.append(job_out)

    # Gather results on rank 0.
    results = MPI.COMM_WORLD.gather(results, root=0)

    if COMM.rank == 0:
        # Flatten list of lists. What a weird 1-liner
        results = [_i for temp in results for _i in temp]

        print('___ saving to h5 ___')
        with h5py.File("datagen_{0}.hdf5".format(num_perclass), "w") as myh5:
            for _i,res in enumerate(results):
                dset = myh5.create_dataset(_i, data=res)
        print('###### all done. ######')

if __name__ == '__main__':
    
    # get number of samples to generate 
    # create an ensemble of FhN systems
    num_perclass = int(sys.argv[1])
    num_systems = 4* int(sys.argv[1])
    num_IC = int(sys.argv[2])

    times = np.linspace(0, 50, 100)

    # generate systems
    if COMM.rank == 0:
        a_list = np.random.rand(num_systems)
        b_list = np.random.rand(num_systems)
        tau_list = 20*np.random.rand(num_systems)
        I_list = 0.3*np.random.randn(num_systems)+0.05

        labels_list = []
        label_dict = {0:'MS-NE', 1:'LC', 2:'MS-E', 3:'BS'}
        # make a few monostables - non excitable: tau small, b small: label 0

        for i in range(num_perclass):
            a_list[i] = 0.2*(1+0.2*np.random.randn())
            b_list[i] = 0.5*(1+0.2*np.random.randn())
            tau_list[i] = 0.125*(1+0.2*np.random.randn())
            I_list[i] = 0.5
            labels_list.append(0)


        # make a few limit cycles - excitable: tau bigger, b small
        for i in range(num_perclass, 2*num_perclass):
            a_list[i] = 0.2*(1+0.2*np.random.randn())
            b_list[i] = 0.5*(1+0.2*np.random.randn())
            tau_list[i] = 1*(1+0.2*np.random.randn())
            I_list[i] = 0.5
            labels_list.append(1)

        # make a few monostable - excitable: tau bigger, b large, a and I s.t. only 1 fp
        for i in range(2*num_perclass, 3*num_perclass):
            a_list[i] = 0.8*(1+0.2*np.random.randn())
            b_list[i] = 1*(1+0.2*np.random.randn())
            tau_list[i] = 12.5*(1+0.2*np.random.randn())
            I_list[i] = 0.1
            labels_list.append(2)

        # make a few bistable: tau bigger, b large, a and I s.t. 3 fp
        for i in range(3*num_perclass, num_systems):
            a_list[i] = 0.8*(1+0.2*np.random.randn())
            b_list[i] = 2*(1+0.2*np.random.randn())
            tau_list[i] = 12.5*(1+0.2*np.random.randn())
            I_list[i] = 0.5
            labels_list.append(3)
            
        parameters = [ (a_list[_i], b_list[_i], tau_list[_i], I_list[_i], labels_list[_i])
                      for _i in range(num_systems)]
        main(parameters)