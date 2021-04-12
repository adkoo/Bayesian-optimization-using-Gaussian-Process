# -*- coding: iso-8859-1 -*-

#How to make your own scan parameter file:
#1) On line 30, choose a unique filename.
#2) On lines 35-56, replace the expressions to the right of the assignment statements with the desired values (maintain formatting)
#3) Run this file from the /basic_gp_stuff/ directory.

import numpy as np
import logging
logging.basicConfig(format='[%(filename)s: line %(lineno)d] %(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)


def CheckScanParams(paramsfile):
    """
    Add in error handling here to check for correct dimensionality of various parameters
    :param paramsfile: the params file generated in this script
    :return: None
    """

    # check for correct dimensionality of gp_precisionmatrix:
    if not paramsfile['gp_precisionmat'].shape[1] == len(my_scan_params['dev_ids']):
        logger.error('incorrect number of parameters detected in gp_precisionmat')
    # check for correct dimensionality of start_point:
    if not len(my_scan_params['start_point'])  == len(my_scan_params['dev_ids']):
        logger.error('Number of device ids and starting parameters dont match')



filename = 'scan_params_example.npy' # choose a unique filename.

my_scan_params = {}

#number of seconds to wait in between target function evaluations, float
my_scan_params['acquisition_delay'] = 0.0 

#device control keys for machine, list or 1d-array
my_scan_params['dev_ids'] = ['sim_dev_0', 'sim_dev_1', 'sim_dev_2', 'sim_dev_3'] 

#gp amplitude parameter, float
my_scan_params['gp_amp'] = 1.0

#gp noise std dev parameter, float 
my_scan_params['gp_noise'] = 0.01 

#gp precision matrix, 2d-array of shape ndim x ndim where ndim = len(my_scan_params['dev_ids'])
my_scan_params['gp_precisionmat'] = np.array([ [0.5, 0, 0, 0],  
                                               [0, 0.5, 0, 0],
                                               [0, 0, 0.5, 0],
                                               [0, 0, 0, 0.5]  ])

#control device settings from which to start the scan, 1d-array of len(ndim)
my_scan_params['start_point'] = np.array([ 0., 0., 0., 0.]) 

#UCB acquisition function parameters in order [nu, delta], list of length 2. If delta is None, ucb will do a fixed tail search using nu as a zscore
my_scan_params['ucb_params'] = [2.0, None] 

CheckScanParams(my_scan_params)   # perform some basic error handling

np.save('params/'+filename, my_scan_params)