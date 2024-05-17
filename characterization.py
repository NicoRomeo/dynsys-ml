import numpy as np
from scipy.optimize import root
from sklearn.cluster import AgglomerativeClustering

def trajsorter_nofunc(traj, rtol=5e-2, tau=10):
    x = traj[:,0]
    y = traj[:,1]
    #print('x shape', x.shape)
    x_com, y_com = np.mean(x), np.mean(y)
    x_scale, y_scale = np.max(x-x_com), np.max(y-y_com)
    r = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y - y_com, x - x_com)
    # there are only 3 possible outcomes of trajectory classification here:
    # 1 - convergence to stable fp; also return estimate of fp location
    # 2 - if convergence to stable fp: is the trajectory excitable?
    # 3 - else, limit cycle
    # To determine if trajectory is convergent: 
    # is the mean position of the last few points close to the last point?
    xf, yf = x[-tau:], y[-tau:]
    #print('xf shape', xf.shape)
    res_x, res_y = np.abs(np.mean(xf)-xf[-1]), np.abs(np.mean(yf)-yf[-1])
    #print(res_x.shape)
    if (res_x/x_scale) + (res_y/y_scale) < rtol:
        # cv to fixed point at estimate position 
        fp_est = xf[-1], yf[-1] # estimated fp position
        # is the trajectory excitable? 
        # for this, check if trajectory ever points away from the fixed point
        xdot, ydot = np.diff(x), np.diff(y)
        xr, yr = x[:-1] - x[-1], y[:-1] - y[-1]
        if (xr * xdot + yr*ydot > 0 ).any():
            # excitable (or at least non-monotonic decay)
            return 2, fp_est
        else:
            return 1, fp_est
    # else, limit cycle
    return 3, (x_com, y_com)

def trajsorter(traj, fps, rtol=1e-2, extol=0):
    x = traj[:,0]
    y = traj[:,1]
    #print('x shape', x.shape)
    x_com, y_com = np.mean(x), np.mean(y)
    x_scale, y_scale = np.max(x-x_com), np.max(y-y_com)
    scale = 0.5*(x_scale + y_scale)
    r = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y - y_com, x - x_com)
    # there are only 3 possible outcomes of trajectory classification here:
    # 1 - convergence to stable fp; also return estimate of fp location
    # 2 - if convergence to stable fp: is the trajectory excitable?
    # 3 - else, limit cycle
    # To determine if trajectory is convergent: 
    # is the mean position of the last few points close to the last point?

    # find the closest (stable) fixed point to last timepoint
    if fps.size > 0:
        dists_to_fp = np.sqrt((x[-1]-fps[:,0])**2 + (y[-1]-fps[:,1])**2)
        idx_min = np.argmin(dists_to_fp)
        mindist = dists_to_fp[idx_min]
    else: # if there are no stable fixed points, this must be a LC
        return 3, (x_com, y_com)
    
    # if the closest fixed point is very close, declare trajectory as converged
    #print(res_x.shape)
    if mindist/scale < rtol:
        # cv to fixed point at estimate position 
        fp_est = fps[idx_min] # estimated fp position
        # is the trajectory excitable? 
        # for this, check if trajectory ever points away from the fixed point
        xdot, ydot = np.diff(x), np.diff(y)
        xr, yr = x[:-1] - x[-1], y[:-1] - y[-1]
        if (xr * xdot + yr*ydot > extol*scale**2 ).any():
            # excitable (or at least non-monotonic decay)
            return 2, fp_est
        else:
            return 1, fp_est
    # else, limit cycle
    return 3, (x_com, y_com)

def classify_system_simple(trajs, param):
    num_traj = trajs.shape[0]
    types = []
    fps = []
    for i in range(num_traj):
        cat, fp = trajsorter_nofunc(trajs[i,:,:], rtol=5e-2, tau=10)
        types.append(cat)
        fps.append(fp)
    types = np.array(types)
    ratio_ms = np.sum(types == 1)/num_traj
    ratio_ex = np.sum(types == 2)/num_traj
    ratio_LC = np.sum(types == 3)/num_traj
    # establish number of stable fps
    stable_fps = np.array([fp for i, fp in enumerate(fps) if types[i] != 3])
    # --> cluster stable fp based on a distance criterion
    if stable_fps.size == 0:
        num_fp = 0
    else:
        clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=np.std(trajs[:])/10).fit_predict(stable_fps)
        # number of classes
        num_fp  = len(np.unique(clustering))
    results = {'ratio_ms':ratio_ms, 'ratio_ex':ratio_ex, 'ratio_LC':ratio_LC, 'num_fp':num_fp}
    return results

def fixedpoints_generic(param, n0=30, w0=30):
    a = param[:,0]
    b = param[:,1]
    def func(y):
        dy = np.zeros(2)
        dy[0] = a[0]* y[0] + a[1] * y[1] + a[2]* y[0]**2 + a[3]* y[0]*y[1] + a[4]* y[1]**2  \
                + a[5]*y[0]**3 + a[6]*y[0]**2*y[1]+a[7]*y[0]*y[1]**2 + a[8]* y[1]**3
        dy[1] = b[0]*y[0] + b[1]* y[1] + b[2]* y[0]**2  + b[3]* y[0]*y[1] + b[4]* y[1]**2 \
                + b[5]*y[0]**3 + b[6]*y[0]**2*y[1]+b[7]*y[0]*y[1]**2 + b[8]* y[1]**3
        return dy
    def jac(y):
        return np.array([[a[0] + 2*a[2]* y[0] + a[3]*y[1]
                + 3*a[5]*y[0]**2 + 2*a[6]*y[0] *y[1]+a[7]*y[1]**2,
                a[1] * y[1] + a[3]* y[0] + 2*a[4]* y[1]  + a[6]*y[0]**2 +2* a[7]*y[0]*y[1] + 3*a[8]* y[1]**2],
        [b[0] + 2*b[2]* y[0]  + b[3]*y[1] + 3*b[5]*y[0]**2 + 2*b[6]*y[0]*y[1]+b[7]*y[1]**2,
          b[1]  + b[3]* y[0] + 2*b[4]* y[1]  + b[6]*y[0]**2 + 2*b[7]*y[0]*y[1] + 3*b[8]* y[1]**2
        ]])
    # solve for roots
    
    guesses = w0*(np.random.rand(n0+1, 2)-.5) #
    guesses[0,:] = 0.
    sols = []
    for k in range(guesses.shape[0]):
        sol = root(func, guesses[k] ,method='lm', jac=jac)
        if sol.success:
            sols.append(sol.x)
    sols = np.unique(np.array(sols).round(decimals=4), axis=0)
    stability = [np.trace(jac(x)) for x in sols]
    return sols, stability

def fixedpoints_snic(param):
    k = param[0]
    a = param[1]
    def func(y):
        dy = np.zeros(2)
        dy[0] = k * y[0] + (1- a)*y[1] - y[0]**3 - y[0]*y[1]**2 
        dy[1] = (1 + a)*y[0] + k * y[1] - y[0]**2*y[1] -  y[1]**3
        return dy
    def jac(y):
        return np.array([[k - 3.0 *y[0]**2 - y[1]**2, 1.0-a - 2.0 *y[0]*y[1]],
                         [1.0+a - 2.0*y[0]*y[1], k - y[0]**2 - 3.0*y[1]**2]])
    # solve for roots
    guesses = 30*(np.random.rand(31, 2)-.5) #
    guesses[0,:] = 0.
    #[trajs[i,-1,:] for i in range(trajs.shape[0])]
    sols = []
    for i in range(guesses.shape[0]):
        sol = root(func, guesses[i] ,method='hybr', jac=jac, tol=1e-7)
        if sol.success:
            sols.append(sol.x)
    sols = np.unique(np.array(sols).round(decimals=4), axis=0)
    #print([jac(x) for x in sols])
    stability = np.array([np.trace(jac(x)) for x in sols])
    return sols, stability

def fixedpoints_hopf(param):
    a = param
    def func(y):
        dy = np.zeros(2)
        dy[0] = a* y[0] - y[1] -y[0]**3 -y[0]*y[1]**2 
        dy[1] = y[0] + a * y[1] - y[0]**2*y[1] -  y[1]**3
        return dy
    def jac(y):
        return np.array([[a - 3*y[0]**2 -y[1]**2,
                -1 -2*y[0]*y[1]],
        [1- 2*y[0]*y[1],
          a - y[0]**2 -3*y[1]**2
        ]])
    # solve for roots
    guesses = 30*(np.random.rand(31, 2)-.5) #
    guesses[0,:] = 0.
    sols = []
    for k in range(guesses.shape[0]):
        sol = root(func, guesses[k] ,method='lm', jac=jac)
        if sol.success:
            sols.append(sol.x)
    sols = np.unique(np.array(sols).round(decimals=4), axis=0)
    stability = np.array([np.trace(jac(x)) for x in sols])
    #print(stability)
    return sols, stability
    

def classify_system_hopf(trajs, param, rtol, extol):
    num_traj = trajs.shape[0]
    types = []
    fps, stability = fixedpoints_hopf(param)
    stable_fps = fps[stability<0.,:]
    
    for i in range(num_traj):
        cat, fp = trajsorter(trajs[i,:,:], stable_fps, rtol=rtol, extol=0.1)
        types.append(cat)
        #fps.append(fp)
    types = np.array(types)
    ratio_ms = np.sum(types == 1)/num_traj
    ratio_ex = np.sum(types == 2)/num_traj
    ratio_LC = np.sum(types == 3)/num_traj
    if stable_fps.size == 0:
        num_fp = 0
    else:
        num_fp  = stable_fps.shape[0]
    results = {'ratio_ms':ratio_ms, 'ratio_ex':ratio_ex, 'ratio_LC':ratio_LC, 'num_fp':num_fp}
    return results

def classify_system_snic(trajs, param, rtol=1e-2, extol=0.):
    num_traj = trajs.shape[0]
    types = []
    fps, stability = fixedpoints_snic(param)
    print('{0} Fixed points found:\n'.format(fps.shape[0]), fps)
    print('TrJ:\n', stability)
    stable_fps = fps[stability<0.,:]
    
    for i in range(num_traj):
        cat, fp = trajsorter(trajs[i,:,:], stable_fps, rtol=rtol, extol=extol)
        types.append(cat)
        #fps.append(fp)
    types = np.array(types)
    ratio_ms = np.sum(types == 1)/num_traj
    ratio_ex = np.sum(types == 2)/num_traj
    ratio_LC = np.sum(types == 3)/num_traj
    if stable_fps.size == 0:
        num_fp = 0
    else:
        num_fp  = stable_fps.shape[0]
    results = {'ratio_ms':ratio_ms, 'ratio_ex':ratio_ex, 'ratio_LC':ratio_LC, 'num_fp':num_fp}
    return results

def classify_system(trajs, param, rtol=1e-2, extol=0., n0=30, w0=30):
    num_traj = trajs.shape[0]
    types = []
    fps, stability = fixedpoints_generic(param, n0=n0, w0=w0)
    stable_fps = fps[stability<0.,:]
    
    for i in range(num_traj):
        cat, fp = trajsorter(trajs[i,:,:], stable_fps, rtol=rtol, extol=0.1)
        types.append(cat)
        #fps.append(fp)
    types = np.array(types)
    ratio_ms = np.sum(types == 1)/num_traj
    ratio_ex = np.sum(types == 2)/num_traj
    ratio_LC = np.sum(types == 3)/num_traj
    if stable_fps.size == 0:
        num_fp = 0
    else:
        num_fp  = stable_fps.shape[0]
    results = {'ratio_ms':ratio_ms, 'ratio_ex':ratio_ex, 'ratio_LC':ratio_LC,
               'num_fp':num_fp, 'fps':fps, 'stability':stability}
    return results