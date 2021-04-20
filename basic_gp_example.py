"""
This code will run an offline example as-is without any input arguments. Simply run the file in python.

In order to use it on a specific machine:
1. Change the *importlib.import_module('machine_interfaces.machine_interface_example')* to your machine interface name.
2. Select the name of your .npy file that contains your desired scan params, or change directly the parameters.
3. Choose if you want results saved by setting saveResultsQ = True/False.
   *Note:* If saveResultsQ is set to True, the scan data will be saved to the local directory called 'saved_results' and will have filename formatted as 'scan_YYYY-MM-DD-hhmmss.npy'
"""

import numpy as np
import importlib
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import warnings
warnings.simplefilter("ignore")

from modules.bayes_optimization import BayesOpt, negUCB, negExpImprove
from modules.OnlineGP import OGP

saveResultsQ = False

mi_module = importlib.import_module('machine_interfaces.machine_interface_example')


#load the dict that contains the parameters for the scan (control pv list, starting settings, and gp hyperparams)
scan_params_filename = 'my_scan_params.npy'
scan_params = np.load('params/'+scan_params_filename, allow_pickle=True).item()

#how long to wait between acquisitions
acquisition_delay = scan_params['acquisition_delay'] # how long to wait between acquisitions

#create the machine interface
dev_ids = scan_params['dev_ids']
start_point = 0*scan_params['start_point'] #if start_point is set to None, the optimizer will start from the current device settings.
mi = mi_module.machine_interface(dev_ids = dev_ids, start_point = start_point)  #an isotropic n-dimensional gaussian with amplitude=1, centered at the origin, plus gaussian background noise with std dev = 0.1

#create the gp
ndim = len(dev_ids)
# GP parameters
gp_precisionmat = scan_params['gp_precisionmat']
gp_amp = scan_params['gp_amp'] 
gp_noise_variance =scan_params['gp_noise'] 
hyperparams = {'precisionMatrix': gp_precisionmat, 'amplitude': gp_amp, 'noise_var': gp_noise_variance} 
gp = OGP(ndim, hyperparams)

#create the bayesian optimizer that will use the gp as the model to optimize the machine 
opt = BayesOpt(gp, mi, acq_func="UCB", start_dev_vals = mi.x, dev_ids = dev_ids)
opt.ucb_params = scan_params['ucb_params'] #set the acquisition function parameters
print('ucb_params',opt.ucb_params)

#run the gp search for some number of steps
Obj_state_s=[]

optimize_kernel_on_the_fly = None #optimize_kernel_on_the_fly is the iteration number to start optimize the kernel's hyperparmaters. If None, no optimization of the hypers during BO. 

Niter = 10
for i in range(Niter):
    clear_output(wait=True) 
    print ('iteration =', i)
    print ('current position:', mi.x, 'current objective value:', mi.getState()[1])
   
    Obj_state_s.append(mi.getState()[1][0])
    
    f = plt.figure(figsize=(20,3))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax.set_ylabel('Input controls',fontsize=12)
    ax.set_xlabel('Iteration',fontsize=12)
    for x, label in zip(opt.X_obs.T, opt.dev_ids):
        ax.plot(x,'.-',label = label)
    ax.legend()
    ax2.set_ylabel('Objective',fontsize=12)
    ax2.set_xlabel('Iteration',fontsize=12)
    ax2.plot(Obj_state_s,'.-')
    plt.show(); 
    
    if optimize_kernel_on_the_fly is not None:
        if i > optimize_kernel_on_the_fly:
            opt.optimize_kernel_hyperparameters()    

    opt.OptIter()
    time.sleep(acquisition_delay)   
    
#save results if desired
if saveResultsQ == True:
    timestr = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    try:
        os.mkdir('saved_results')
    except:
        pass
    results = {}
    results['scan_params'] = scan_params
    results['xs'] = opt.X_obs
    results['ys'] = np.array([y[0][0] for y in opt.Y_obs])
    results['time'] = timestr
    np.save('saved_results/scan_'+timestr, results)