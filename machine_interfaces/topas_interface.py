# -*- coding: iso-8859-1 -*-
import numpy as np
import pickle
from XXXXX import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel

class machine_interface:
    def __init__(self, dev_ids, start_point = None): 
        self.name = 'topas_interface' 
        
        #setup topas simulation enviroment 
        self.GenerateTopasModel()
        self.RunTopasModel()
        
        #setup the devices name (input controls)
        self.pvs = np.array(dev_ids)
        if type(start_point) == type(None):
            current_x = np.zeros(len(self.pvs)) #replace with expression that reads current ctrl pv values (x) from machine
            self.setX(current_x)
        else: 
            self.setX(start_point)
 
    def setX(self, x_new):
        self.x = np.array(x_new, ndmin=2)
            

    def getState(self): #objective
        objective_state = self.BeamWidthObjective(self.x[0][0]) #BeamWidthObjective.py (in machine_interface folder)
        return np.array(self.x, ndmin = 2), np.array([[objective_state]])

    
