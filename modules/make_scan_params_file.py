# -*- coding: iso-8859-1 -*-

#How to make your own scan parameter file:
#1) On line 10, choose a unique filename.
#2) On lines 15-36, replace the expressions to the right of the assignment statements with the desired values (maintain formatting)
#3) Run this file from the /basic_gp_stuff/ directory.

import numpy as np

filename = 'TopasOptimiserParams.npy'

my_scan_params = {}

#number of seconds to wait in between target function evaluations, float
my_scan_params['acquisition_delay'] = 0.0 

#device control keys for machine, list or 1d-array
my_scan_params['dev_ids'] = ['UpStreamHoleSize','DownStreamHoleSize']

#gp amplitude parameter, float
my_scan_params['gp_amp'] = 1.0

#gp noise std dev parameter, float 
my_scan_params['gp_noise'] = 0.01 

#gp precision matrix, 2d-array of shape ndim x ndim where ndim = len(my_scan_params['dev_ids'])
my_scan_params['gp_precisionmat'] = np.array([ [0.5]  ])

#control device settings from which to start the scan, 1d-array of len(ndim)
my_scan_params['start_point'] = np.array([1.0, 2.0])

#UCB acquisition function parameters in order [nu, delta], list of length 2. If delta is None, ucb will do a fixed tail search using nu as a zscore
my_scan_params['ucb_params'] = [1.0, None] 

np.save('../params/'+filename, my_scan_params)