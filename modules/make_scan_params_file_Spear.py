# -*- coding: iso-8859-1 -*-

#How to make your own scan parameter file:
#1) On line 10, choose a unique filename.
#2) On lines 15-36, replace the expressions to the right of the assignment statements with the desired values (maintain formatting)
#3) Run this file from the /basic_gp_stuff/ directory.

import numpy as np

filename = 'spear_sim_params.npy'

my_scan_params = {}

#device control keys for machine, list or 1d-array
my_scan_params['dev_ids'] = ['q1', 'q2', 'q3','q4','q5','q6','q7','q8','q9','q10','q11','q12','q13'] 

#gp amplitude parameter, float
my_scan_params['gp_amp'] = 0.11628

#gp noise std dev parameter, float 
my_scan_params['gp_noise'] = 0.01

#gp precision matrix, 2d-array of shape ndim x ndim where ndim = len(my_scan_params['dev_ids'])
my_scan_params['lengthscale'] = np.array([5.71,3.36,3.47,3.3,3.09,4.5,4.78,2.31,3.26,3.44,3.08,3.13,5.57]) # SK learn results on RCDS data
# 4*np.array(np.ones(len(my_scan_params['dev_ids'])))

my_scan_params['gp_precisionmat'] = np.diag(my_scan_params['lengthscale']**-2)

#control device settings from which to start the scan, 1d-array of len(ndim)
my_scan_params['start_point'] =np.array([-0.0103985530950442,-0.0456542206062778,0.000744423528543758,0.0192008242937595,-0.0345509351023979,0.00424757427214989,-0.00314017442732051,-0.0455976465860040,0.0242246057081527,0.00130194445222342,0.0191335800418534,-0.0340463413476130,-0.0119635904166963])
# np.array(np.zeros(len(my_scan_params['dev_ids']))) 

#UCB acquisition function parameters in order [nu, delta], list of length 2. If delta is None, ucb will do a fixed tail search using nu as a zscore
my_scan_params['ucb_params'] = [2., None] 

my_scan_params['offset'] = 0.5

my_scan_params['acquisition_delay'] = 0.0 # wait_time  #number of seconds to wait in between target function evaluations, float

np.save('params/'+filename, my_scan_params)



