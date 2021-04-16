# -*- coding: iso-8859-1 -*-

import numpy as np
from machine_interfaces.injector.injector_surrogate import *

class machine_interface:
    def __init__(self, dev_ids = ['SOL1:solenoid_field_scale','CQ01:b1_gradient','SQ01:b1_gradient'], start_point = None):
        self.pvs = np.array(dev_ids)
        self.name = 'injectorSM' 
       
        #Load injector surrogate model
        self.Model = Surrogate_NN()
        self.Model.load_saved_model()
        self.Model.load_scaling()
        print('SM was set successfully!')
        
        # initizlie the reference point  with all the nominal model inputs -  set once only.
        # ref point contains laser radius, iris, sol, CQ, SQ. 
        #in the comment it is sim_pv, machine_pv
        ref_point = [[4.23867825e-01, # distgen:r_dist:sigma_xy:value'
                      3.06083484e+00, # 'distgen:t_dist:length:value',
                      2.50000000e+02, # 'distgen:total_charge:value',
                      2.45806452e-01, # 'SOL1:solenoid_field_scale',
                      7.13917676e-04, # 'CQ01:b1_gradient',
                      3.27285211e-04, # 'SQ01:b1_gradient',
                      5.80000000e+07, # 'L0A_scale:voltage',
                      -9.53597349e+00, # 'L0A_phase:dtheta0_deg',
                      7.00000000e+07, #  'L0B_scale:voltage',
                        9.85566222e+00]] #  'L0B_phase:dtheta0_deg'

        #convert to machine units
        ref_point = self.Model.sim_to_machine(np.asarray(ref_point))
        # This should be:
        #         [array([ 1.27160360e+00,  1.85505142e+00,  2.50000000e+02,  4.77969346e-01,
        #         -1.49922712e-03, -6.87298943e-04,  5.80000000e+01, -9.53597349e+00,
        #          7.00000000e+01,  9.85566222e+00])]

        
        #nested_list
        ref_point=[ref_point[0]]
        
        #make input array of length model_in_list
        self.x_in = np.empty((1,len(self.Model.model_in_list)))
        
        #fill in reference point around which to optimize
        self.x_in[:,:] = np.asarray(ref_point)
      
        if type(start_point) == type(None):
            current_x = np.zeros(len(self.pvs)) #replace with expression that reads current ctrl pv values (x) from machine
            self.setX(current_x)
        else: 
            self.setX(start_point)

    def setX(self, x_new):
        """
        set solenoid, SQ, CQ to values from optimization step
        """
        self.x = np.array(x_new, ndmin=2)
        self.x_in[:, self.Model.loc_in[self.pvs[0]]] = np.array(self.x)[0][0]
        self.x_in[:, self.Model.loc_in[self.pvs[1]]] = np.array(self.x)[0][1]
        self.x_in[:, self.Model.loc_in[self.pvs[2]]] = np.array(self.x)[0][2]
        
        
    def getState(self): 
        y_out = self.Model.pred_machine_units(self.x_in)
        emitx = y_out[:,self.Model.loc_out['norm_emit_x']] #grab norm_emit_x out of the model
        emity = y_out[:,self.Model.loc_out['norm_emit_y']] #grab norm_emit_y out of the model
        objective_state = np.sqrt(emitx*emity)[0]*10**6 #for um units
        objective_state += 0.001*np.random.normal() 
        return np.array(self.x, ndmin = 2), np.array([[-objective_state]])
    
"""
sim_name_to_pv_name 
{'distgen:r_dist:sigma_xy:value': 'IRIS:LR20:130:CONFG_SEL',
 'SOL1:solenoid_field_scale': 'SOLN:IN20:121:BDES',
 'CQ01:b1_gradient': 'QUAD:IN20:121:BDES',
 'SQ01:b1_gradient': 'QUAD:IN20:122:BDES',
 'L0A_phase:dtheta0_deg': 'ACCL:IN20:300:L0A_PDES',
 'L0B_phase:dtheta0_deg': 'ACCL:IN20:400:L0B_PDES',
 'L0A_scale:voltage': 'ACCL:IN20:300:L0A_ADES',
 'L0B_scale:voltage': 'ACCL:IN20:400:L0B_ADES',
 'QA01:b1_gradient': 'QUAD:IN20:361:BDES',
 'QA02:b1_gradient': 'QUAD:IN20:371:BDES',
 'QE01:b1_gradient': 'QUAD:IN20:425:BDES',
 'QE02:b1_gradient': 'QUAD:IN20:441:BDES',
 'QE03:b1_gradient': 'QUAD:IN20:511:BDES',
 'QE04:b1_gradient': 'QUAD:IN20:525:BDES',
 'distgen:t_dist:length:value': 'Pulse_length',
 'distgen:total_charge:value': 'Charge'}
 
 """