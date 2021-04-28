# -*- coding: iso-8859-1 -*-

import os 
import operator as op
import numpy as np
# from scipy.stats import norm
from scipy.optimize import minimize
# from scipy.optimize import approx_fprime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel
from modules.OnlineGP import OGP
try:
    from scipy.optimize import basinhopping
    basinhoppingQ = True
except:
    basinhoppingQ = False
try:
    from .parallelstuff import *
    multiprocessingQ = True
    basinhoppingQ = False
except:
    print ('failed to import parallelstuff')
    basinhoppingQ = False
    multiprocessingQ = False
import time
from copy import deepcopy
from itertools import chain

class BayesOpt:
    """
    Contains the Bayesian optimization class with the following methods:
    acquire(): Returns the point that maximizes the acquisition function.
        For 'testEI', returns the index of the point instead.
        For normal acquisition, currently uses the bounded L-BFGS optimizer.
            Haven't tested alternatives much.
    best_seen(): Uses the model to make predictions at every observed point,
        returning the best-performing (x,y) pair. This is more robust to noise
        than returning the best observation, but could be replaced by other,
        faster methods.
    OptIter(): The main method for Bayesian optimization. Maximizes the
        acquisition function, then uses the interface to test this point and
        update the model.
    """
    
    def __init__(self, model, target_func, acq_func='EI', xi=0.0, alt_param=-1, m=200, bounds=None, iter_bound=False, prior_data=None, start_dev_vals=None, dev_ids=None, searchBoundScaleFactor=None, optimize_kernel_on_the_fly = None, verboseQ=False):
        """        
        Initialization parameters:
        --------------------------
        model: an object with methods 'predict', 'fit', and 'update'
                surrogate model to use
        interface: an object which supplies the state of the system and
            allows for changing the system's x-value.
            Should have methods '(x,y) = intfc.getState()' and 'intfc.setX(x_new)'.
            Note that this interface system is rough, and used for testing and
                as a placeholder for the machine interface.
        acq_func: specifies how the optimizer should choose its next point.
            'PI': uses probability of improvement. The interface should supply y-values.
            'EI': uses expected improvement. The interface should supply y-values.
            'UCB': uses GP upper confidence bound. No y-values needed.
            'testEI': uses EI over a finite set of points. This set must be
                provided as alt_param, and the interface need not supply
                meaningful y-values.
        xi: exploration parameter suggested in some Bayesian opt. literature
        alt_param: currently only used when acq_func=='testEI'
        m: the maximum size of model; can be ignored unless passing an untrained
            SPGP or other model which doesn't already know its own size
        bounds: a tuple of (min,max) tuples specifying search bounds for each
            input dimension. Generally leads to better performance.
            Has a different interpretation when iter_bounds is True.
        iter_bounds: if True, bounds the distance that can be moved in a single
            iteration in terms of the length scale in each dimension. Uses the
            bounds variable as a multiple of the length scales, so bounds==2
            with iter_bounds==True limits movement per iteration to two length
            scales in each dimension. Generally a good idea for safety, etc.
        prior_data: input data to train the model on initially. For convenience,
            since the model can be trained externally as well.
            Assumed to be a pandas DataFrame of shape (n, dim+1) where the last
                column contains y-values.
        optimize_kernel_on_the_fly: if not None, int which indicated the iteration number to start kernel optimization.
            Currently works for RBF only.
        """
        self.model = model
        self.m = m
        self.bounds = bounds
        self.searchBoundScaleFactor = 1.
        if type(searchBoundScaleFactor) is not type(None):
            try:
                self.searchBoundScaleFactor = abs(searchBoundScaleFactor)
            except:
                print(('BayesOpt - ERROR: ', searchBoundScaleFactor, ' is not a valid searchBoundScaleFactor (scaling coeff).'))
        self.iter_bound = iter_bound 
        self.prior_data = prior_data # for seeding the GP with data acquired by another optimizer
        self.target_func = target_func
        self.optimize_kernel_on_the_fly = optimize_kernel_on_the_fly 
        self.verboseQ = verboseQ
        if self.optimize_kernel_on_the_fly is not None: print('Run BO w/ kernel optimization on the fly')
        try: 
            self.mi = self.target_func.mi
        except:
            self.mi = self.target_func
        self.acq_func = (acq_func, xi, alt_param)
        #self.ucb_params = [0.24, 0.4] # [nu,delta] worked well for LCLS
        self.ucb_params = [2., None] # if we want to used a fixed scale factor of the standard deviation
        self.max_iter = 100
        self.alpha = 1.0 #control the ratio of exploration to exploitation in AI acuisition function
        self.kill = False
        self.ndim = np.array(start_dev_vals).size
        self.multiprocessingQ = multiprocessingQ # speed up acquisition function optimization
        self.dev_ids = dev_ids
        self.start_dev_vals = start_dev_vals
        self.pvs = self.dev_ids

        try:
            # get initial state
            print('Supposed to be grabbing initial machine state...')
            (x_init, y_init) = self.getState()
            print('x_init',x_init)
            print('y_init',y_init)
            self.X_obs = np.array(x_init)
            self.Y_obs = [y_init]
            self.current_x = np.array(np.array(x_init).flatten(), ndmin=2)
        except:
            print('BayesOpt - ERROR: Could not grab initial machine state')
        
        # calculate length scales
        try:
            self.lengthscales = self.model.lengthscales
        except:
            print('WARNING - GP.bayesian_optimization.BayesOpt: Using some unit length scales cause we messed up somehow...')
            self.lengthscales = np.ones(len(self.dev_ids))
        
        # make a copy of the initial params
        self.initial_hyperparams = {}
        self.initial_hyperparams['precisionMatrix'] = np.diag(1./copy.copy(self.lengthscales)**2)
        self.initial_hyperparams['noise_variance'] = copy.copy(self.model.noise_var) 
        self.initial_hyperparams['amplitude_covar'] = copy.copy(self.model.amplitude_covar)
        
        #initiate optimized hypers
        self.hyperparams_opt_all = {}
        self.hyperparams_opt_all['noise_variance'] = [copy.copy(self.model.noise_var)]
        self.hyperparams_opt_all['amplitude_covar'] = [copy.copy(self.model.amplitude_covar)]
        self.hyperparams_opt_all['precisionMatrix'] = [1./copy.copy(self.lengthscales)**2]
        
        if self.verboseQ:
            print('Using prior mean function of ', self.model.prmean)
            print('Using prior mean parameters of ', self.model.prmeanp)
        
        
    def getState(self):
        """
        get current state of the machine
        """
        x_vals, y_val = self.mi.getState()
        return x_vals, y_val

    
    def terminate(self, devices):
        """
        Sets the position back to the location that seems best in hindsight.
        It's a good idea to run this at the end of the optimization, since
        Bayesian optimization tries to explore and might not always end in
        a good place.
        """
        print(("TERMINATE", self.x_best))
        if(self.acq_func[0] == 'EI'):
            # set position back to something reasonable
            for i, dev in enumerate(devices):
                dev.set_value(self.x_best[i])
            #error_func(self.x_best)
        if(self.acq_func[0] == 'UCB'):
            # UCB doesn't keep track of x_best, so find it
            (x_best, y_best) = self.best_seen()
            for i, dev in enumerate(devices):
                dev.set_value(x_best[i])
                
    def sk_kernel(self, hypers_dict):
    
        amp = hypers_dict['amplitude_covar']
        lengthscales = np.diag(hypers_dict['precisionMatrix'])**-0.5
        noise_var = hypers_dict['noise_variance']
        
        se_ard = Ck(amp)*RBF(length_scale=lengthscales, length_scale_bounds=(1e-6,10))
        noise = WhiteKernel(noise_level=noise_var, noise_level_bounds=(1e-9, 1))  # noise terms
        
        sk_kernel = se_ard 
        if self.noiseQ:
            sk_kernel += noise
        t0 = time.time()        
        gpr = GaussianProcessRegressor(kernel=sk_kernel, n_restarts_optimizer=5)
        print("Initial kernel: %s" % gpr.kernel)
        
#         self.ytrain = [y[0][0] for y in self.Y_obs]
        
        gpr.fit(self.X_obs, np.array(self.Y_obs).flatten())
        print('SK fit time is ',time.time() - t0)
        print("Learned kernel: %s" % gpr.kernel_)
        print("Log-marginal-likelihood: %.3f" % gpr.log_marginal_likelihood(gpr.kernel_.theta))
        #print(gpr.kernel_.get_params())
        
        
        if self.noiseQ:
            # RBF w/ noise
            sk_ls = gpr.kernel_.get_params()['k1__k2__length_scale']
            sk_amp = gpr.kernel_.get_params()['k1__k1__constant_value']
            sk_loklik = gpr.log_marginal_likelihood(gpr.kernel_.theta)
            sk_noise = gpr.kernel_.get_params()['k2__noise_level']
        
        else:
            #RBF w/o noise
            sk_ls = gpr.kernel_.get_params()['k2__length_scale']
            sk_amp = gpr.kernel_.get_params()['k1__constant_value']
            sk_loklik = gpr.log_marginal_likelihood(gpr.kernel_.theta)
            sk_noise = 0

        # make dict
        sk_hypers = {}
        sk_hypers['precisionMatrix'] =  np.diag(1./(sk_ls**2)) 
        sk_hypers['noise_variance'] = sk_noise
        sk_hypers['amplitude_covar'] = sk_amp


        return sk_loklik, sk_hypers            
                
    def optimize_kernel_hyperparameters(self, noiseQ = False):
        """
       Optimize the kernel hyperparameters before acuiring the next point.
       This method optimizes the kernel twice - starting from the initial or last hyperparamters.
       Then compares the log likelihood and re-build the GP model using the most likely hypers.
       Note:  sk learn can't deal with matrix. so we can only optimize on lengthscales.
        """
        self.noiseQ = noiseQ
        # optimize kernel using SK learn from initial hyperparams
        print('optimize on initial hyperparams')
        sk_loklik0, sk_hypers0 = self.sk_kernel(self.initial_hyperparams)

        # optimize kernel using SK learn from current hyperparams        
        self.current_hyperparams = {}
        self.current_hyperparams['precisionMatrix'] = np.diag(1./self.model.lengthscales**2)
        self.current_hyperparams['noise_variance'] = self.model.noise_var
        self.current_hyperparams['amplitude_covar'] = self.model.amplitude_covar
                                    
        print('optimize on last hyperparams seen so far')
        sk_loklik, sk_hypers = self.sk_kernel(self.current_hyperparams)

        # compare likelihoods and choose best hyperparams
        if sk_loklik > sk_loklik0:
            hyperparams_opt = sk_hypers 
        else:
            hyperparams_opt  = sk_hypers0
            
        
        for key in hyperparams_opt:
            if key == 'precisionMatrix':
                self.hyperparams_opt_all[key] = np.array(list(chain(self.hyperparams_opt_all[key],                            [hyperparams_opt[key].diagonal()]))) 
            else:
                self.hyperparams_opt_all[key] = list(chain(self.hyperparams_opt_all[key], [hyperparams_opt[key]]))
        

        if self.verboseQ: print('hyperparams_opt ',hyperparams_opt)

        # create new OnlineGP model - overwrites the existing one
        if self.verboseQ: print('sanity dim check: ',self.model.dim == self.X_obs.shape[1])
            
        self.model = OGP(self.model.dim, hyperparams = hyperparams_opt, maxBV = self.model.maxBV, covar = self.model.covar) 
#         ,weighted=self.model.weighted, maxBV=self.model.maxBV) #, prmean=self.model.prmean, prmeanp=self.model.prmeanp, prvar=self.model.prvar, prvarp=self.model.prvarp , proj=self.model.proj,thresh=self.model.thresh, sparsityQ=self.model.sparsityQ, verboseQ=self.model.verboseQ)
        
        
        # initialize new model on current data
        self.model.fit(self.X_obs, np.array(self.Y_obs).flatten(), self.X_obs.shape[0])



    def minimize(self, error_func, x):
        """
        weighting for exploration vs exploitation in the GP 
        at the end of scan, alpha array goes from 1 to zero.
        """
        inverse_sign = -1
        self.current_x = np.array(np.array(x).flatten(), ndmin=2)
        self.X_obs = np.array(self.current_x)
        self.Y_obs = [np.array([[inverse_sign*error_func(x)]])]
        
        # iterate though the GP method
        for i in range(self.max_iter):
            # get next point to try using acquisition function
            x_next = self.acquire()
            
            if self.optimize_kernel_on_the_fly is not None:
                if i > self.optimize_kernel_on_the_fly:
                    print('****** Optimizing kerenl hyperparams')
                    self.optimize_kernel_hyperparameters()    

            y_new = error_func(x_next.flatten())
            if self.opt_ctrl.kill:
                print('WARNING - BayesOpt: Killing Bayesian optimizer...')
                break
            y_new = np.array([[inverse_sign *y_new]])

            # change position of interface
            x_new = deepcopy(x_next)
            self.current_x = x_new

            # add new entry to observed data
            self.X_obs = np.concatenate((self.X_obs, x_new), axis=0)
            self.Y_obs.append(y_new)

            # update the model (may want to add noise if using testEI)
            self.model.update(x_new, y_new)

            
    def OptIter(self,pause=0):
        """
        runs the optimizer for one iteration
        """
        
        # get next point to try using acquisition function
        x_next = self.acquire()
        if(self.acq_func[0] == 'testEI'):
            ind = x_next
            x_next = np.array(self.acq_func[2].iloc[ind,:-1],ndmin=2)    
        
        # change position of interface and get resulting y-value
        self.mi.setX(x_next)
        if(self.acq_func[0] == 'testEI'):
            (x_new, y_new) = (x_next, self.acq_func[2].iloc[ind,-1])
        else:
            (x_new, y_new) = self.mi.getState()
        # add new entry to observed data
        self.X_obs = np.concatenate((self.X_obs,x_new),axis=0)
        self.Y_obs.append(y_new)
        
        # update the model (may want to add noise if using testEI)
        self.model.update(x_new, y_new)# + .5*np.random.randn())
            
            
    def ForcePoint(self,x_next):
        """
        force a point acquisition at our discretion and update the model
        """
        
        # change position of interface and get resulting y-value
        self.mi.setX(x_next)
        if(self.acq_func[0] == 'testEI'):
            (x_new, y_new) = (x_next, self.acq_func[2].iloc[ind,-1])
        else:
            (x_new, y_new) = self.mi.getState()
        # add new entry to observed data
        self.X_obs = np.concatenate((self.X_obs,x_new),axis=0)
        self.Y_obs.append(y_new)
        
        # update the model (may want to add noise if using testEI)
        self.model.update(x_new, y_new)

        
    def best_seen(self):
        """
        Checks the observed points to see which is predicted to be best.
        Probably safer than just returning the maximum observed, since the
        model has noise. It takes longer this way, though; you could
        instead take the model's prediction at the x-value that has
        done best if this needs to be faster.

        Not needed for UCB so do it the fast way (return max obs)
        """
        if(self.acq_func[0] == 'UCB'):
            mu = self.Y_obs
        else:
            (mu, var) = self.model.predict(self.X_obs)
            mu = [self.model.predict(np.array(x,ndmin=2))[0] for x in self.X_obs]

        (ind_best, mu_best) = max(enumerate(mu), key=op.itemgetter(1))
        return (self.X_obs[ind_best], mu_best)

    
    def acquire(self):
        """
        Computes the next point for the optimizer to try by maximizing
        the acquisition function. If movement per iteration is bounded,
        starts search at current position.
        """
        # look from best positions
        (x_best, y_best) = self.best_seen()
        self.x_best = x_best
        x_curr = self.current_x[-1]
        x_start = x_best
            
        ndim = x_curr.size # dimension of the feature space we're searching NEEDED FOR UCB
        try:
            nsteps = 1 + self.X_obs.shape[0] # acquisition number we're on  NEEDED FOR UCB
        except:
            nsteps = 1

        # check to see if this is bounding step sizes
        if(self.iter_bound or True):
            if(self.bounds is None): # looks like a scale factor
                self.bounds = 1.0

            bound_lengths = self.searchBoundScaleFactor * 3. * self.lengthscales # 3x hyperparam lengths
            relative_bounds = np.transpose(np.array([-bound_lengths, bound_lengths]))
            
            iter_bounds = np.transpose(np.array([x_start - bound_lengths, x_start + bound_lengths]))

        else:
            iter_bounds = self.bounds
  
        # options for finding the peak of the acquisition function:
        optmethod = 'L-BFGS-B' # L-BFGS-B, BFGS, TNC, and SLSQP allow bounds whereas Powell and COBYLA don't
        maxiter = 1000 # max number of steps for one scipy.optimize.minimize call
        try:
            nproc = mp.cpu_count() # number of processes to launch minimizations on
        except:
            nproc = 1
        niter = 1 # max number of starting points for search
        niter_success = 1 # stop search if same minima for 10 steps
        tolerance = 1.e-4 # goal tolerance

        # perturb start to break symmetry?
        #x_start += np.random.randn(lengthscales.size)*lengthscales*1e-6

        # probability of improvement acquisition function
        if(self.acq_func[0] == 'PI'):
            aqfcn = negProbImprove
            fargs=(self.model, y_best, self.acq_func[1])

        # expected improvement acquisition function
        elif(self.acq_func[0] == 'EI'):
            aqfcn = negExpImprove
            fargs = (self.model, y_best, self.acq_func[1], self.alpha)

        # gaussian process upper confidence bound acquisition function
        elif(self.acq_func[0] == 'UCB'):
            aqfcn = negUCB
            fargs = (self.model, ndim, nsteps, self.ucb_params[0], self.ucb_params[1])

        # maybe something mitch was using once? (can probably remove)
        elif(self.acq_func[0] == 'testEI'):
            # collect all possible x values
            options = np.array(self.acq_func[2].iloc[:, :-1])
            (x_best, y_best) = self.best_seen()

            # find the option with best EI
            best_option_score = (-1,1e12)
            for i in range(options.shape[0]):
                result = negExpImprove(options[i],self.model,y_best,self.acq_func[1])
                if(result < best_option_score[1]):
                    best_option_score = (i, result)

            # return the index of the best option
            return best_option_score[0]

        else:
            print('WARNING - BayesOpt: Unknown acquisition function.')
            return 0

        try:
            if(self.multiprocessingQ): # multi-processing to speed search

#                 neval = 2*int(10.*2.**(ndim/12.))
#                 nkeep = 2*min(8,neval)

                neval = int(3) 
                nkeep = int(2)

                # add the 10 best points seen so far (largest Y_obs)
                nbest = 3 # add the best points seen so far (largest Y_obs)
                nstart = 2 # make sure some starting points are there to prevent run away searches
                
                yobs = np.array([y[0][0] for y in self.Y_obs])
                isearch = yobs.argsort()[-nbest:]
                for i in range(min(nstart,len(self.Y_obs))): #
                    if np.sum(isearch == i) == 0: # not found in list
                        isearch = np.append(isearch, i)
                        isearch.sort() # sort to bias searching near earlier steps

                v0s = None
                
                for i in isearch:
#                 """
#                 parallelgridsearch generates pseudo-random grid, then performs an ICDF transform
#                 to map to multinormal distrinbution centered on x_start and with widths given by hyper params
#                 """
                    vs = parallelgridsearch(aqfcn,self.X_obs[i],self.searchBoundScaleFactor * 0.6*self.lengthscales,fargs,neval,nkeep)
                                      
                    if type(v0s) == type(None):
                        v0s = copy.copy(vs)
                    else:
                        v0s = np.vstack((v0s,vs))

                v0sort = v0s[:,-1].argsort()[:nkeep] # keep the nlargest
                v0s = v0s[v0sort]
                
                x0s = v0s[:,:-1] # for later testing if the minimize results are better than the best starting point
                v0best = v0s[0]
                
                
                if basinhoppingQ:
                    # use basinhopping
                    bkwargs = dict(niter=niter,niter_success=niter_success, minimizer_kwargs={'method':optmethod,'args':fargs,'tol':tolerance,'bounds':iter_bounds,'options':{'maxiter':maxiter}}) # keyword args for basinhopping
                    res = parallelbasinhopping(aqfcn,x0s,bkwargs)
                
                else:
                    # use minimize
                    mkwargs = dict(bounds=iter_bounds, method=optmethod, options={'maxiter':maxiter}, tol=tolerance) # keyword args for scipy.optimize.minimize
                    res = parallelminimize(aqfcn,x0s,fargs,mkwargs,v0best,relative_bounds=relative_bounds)

            else: # single-processing

                if basinhoppingQ:
                    res = basinhopping(aqfcn, x_start,niter=niter,niter_success=niter_success, minimizer_kwargs={'method':optmethod,'args':(self.model, y_best, self.acq_func[1], self.alpha),'tol':tolerance,'bounds':iter_bounds,'options':{'maxiter':maxiter}})

                else:
                    res = minimize(aqfcn, x_start, args=(self.model, y_best, self.acq_func[1], self.alpha), method=optmethod,tol=tolerance,bounds=iter_bounds,options={'maxiter':maxiter})

                res = res.x
                
        except:
            raise
        return np.array(res,ndmin=2) # return resulting x value as a (1 x dim) vector
        

def negProbImprove(x_new, model, y_best, xi):
    """
    The probability of improvement acquisition function. Initial testing
    shows that it performs worse than expected improvement acquisition
    function for 2D scans (at least when alpha==1 in the fcn below). Alse
    performs worse than EI according to the literature.
    """
    (y_mean, y_var) = model.predict(np.array(x_new,ndmin=2))
    diff = y_mean - y_best - xi
    if(y_var == 0):
        return 0.
    else:
        Z = diff / np.sqrt(y_var)

    return -norm.cdf(Z)

def negExpImprove(x_new, model, y_best, xi,alpha = 1.0):
    """
    The common acquisition function, expected improvement. Returns the
    negative for the minimizer (so that EI is maximized). Alpha attempts
    to control the ratio of exploration to exploitation, but seems to not
    work well in practice. The terminate() method is a better choice.
    """
    (y_mean, y_var) = model.predict(np.array(x_new, ndmin=2))
    diff = y_mean - y_best - xi

    # Nonvectorizable. Can prob use slicing to do the same.
    if(y_var == 0):
        return 0.
    else:
        Z = diff / np.sqrt(y_var)

    EI = diff * norm.cdf(Z) + np.sqrt(y_var) * norm.pdf(Z)
    return alpha * (-EI) + (1. - alpha) * (-y_mean)


def negUCB(x_new, model, ndim, nsteps, nu = 1., delta = 1.):
    """
    GPUCB: Gaussian process upper confidence bound aquisition function
    Default nu and delta hyperparameters theoretically yield "least regret".
    Works better than "expected improvement" (for alpha==1 above) in 2D.

    input params
    x_new: new point in the dim-dimensional space the GP is fitting
    model: OnlineGP object
    ndim: feature space dimensionality (how many devices are varied)
    nsteps: current step number counting from 1
    nu: nu in the tutorial (see below)
    delta: delta in the tutorial (see below)
    
    
    GP upper confidence bound
    original paper: https://arxiv.org/pdf/0912.3995.pdf
    tutorial: http://www.cs.ubc.ca/~nando/540-2013/lectures/l7.pdf
    """

    if nsteps==0: nsteps += 1
    (y_mean, y_var) = model.predict(np.array(x_new,ndmin=2))
#     print('(y_mean, y_var) = ',(y_mean, y_var))
    if delta is None:
        GPUCB = y_mean + nu * np.sqrt(y_var)
    else:
        tau = 2.*np.log(nsteps**(0.5*ndim+2.)*(np.pi**2.)/3./delta)
        GPUCB = y_mean + np.sqrt(nu * tau * y_var)

    return -GPUCB[0]


