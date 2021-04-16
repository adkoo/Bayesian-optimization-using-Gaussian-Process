import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Activation, GaussianNoise, Reshape,  Conv2D, UpSampling2D
from tensorflow.keras import regularizers, datasets, layers, models
from tensorflow.keras.losses import mse 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
import pickle as pick
import pandas as pd
import json
import array
import random
from math import sqrt
import pickle

from IPython.display import clear_output

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,MinMaxScaler



class Surrogate_NN:
    
    def __init__(self, model_info_file = 'machine_interfaces/injector/model_info.json',
                 pv_info_file = 'machine_interfaces/injector/pvinfo.json',
                 take_log_out= True
                ):
        
        
        self.PATH = 'machine_interfaces/injector/'
        screen ='OTR2'
        NAME = 'v3b_cnsga_'

        with open(self.PATH+NAME+screen+'_list_dict.json') as json_file:
            json_names = json.load(json_file)

        #inputs and outputs in raw data
        output_names=json_names['out_'+screen+'_vars']
        input_names = json_names['input_vars']

        #load model info
        model_info = json.load(open(model_info_file))

        #inputs and outputs model is  trained on
        model_in_list = model_info['model_in_list']
        model_out_list = model_info['model_out_list']

        #dictionary of location of variables in array
        loc_in = model_info['loc_in']
        loc_out = model_info['loc_out']

        #inputs and outputs model is  trained on
        input_mins = model_info['train_input_mins']
        input_maxs = model_info['train_input_maxs']
        pv_info = json.load(open(pv_info_file))
        pv_to_sim_factor = pv_info['pv_to_sim_factor']
        sim_to_pv_factor = pv_info['sim_to_pv_factor']
        pv_unit = pv_info['pv_unit']
        pv_name_to_sim_name = pv_info['pv_name_to_sim_name']
        sim_name_to_pv_name = pv_info['sim_name_to_pv_name']
        
        #input variable names
        self.model_in_list = model_in_list
        self.model_out_list = model_out_list
        
        #dictionary mapping names to indices
        self.loc_in = {model_in_list[i]: np.arange(0,len(model_in_list))[i] for i in range(len(model_in_list))} 
        self.loc_out = {model_out_list[i]: np.arange(0,len(model_out_list))[i] for i in range(len(model_out_list))} 
        
        self.input_mins = input_mins
        self.input_maxs = input_maxs
        self.take_log_out = take_log_out
        
        self.debug = False
        
        self.pv_name_to_sim_name = pv_name_to_sim_name
        self.pv_to_sim_factor = pv_to_sim_factor
        self.sim_name_to_pv_name = sim_name_to_pv_name
        
        
    def pred_sim_units(self, x):
    
            x = self.transformer_x.transform(x)
            y = self.model_1.predict(x)
            y = self.transformer_y.inverse_transform(y)

            if self.take_log_out == True:
                return np.exp(y) #trained on log data
            
            else:
                return y
        
    def pred_machine_units(self, x):

            x_s = np.copy(x)
            
            for i in range(0,len(self.model_in_list)):
                x_s[:,self.loc_in[self.model_in_list[i]]] =  x[:,self.loc_in[self.model_in_list[i]]] * self.pv_to_sim_factor[self.sim_name_to_pv_name[self.model_in_list[i]]]

            if self.debug:
                print('small scale units',x_s)

            #scale for NN pred
            
            x_s =  self.transformer_x.transform(x_s)
            y = self.model_1.predict(x_s)
            y = self.transformer_y.inverse_transform(y)
            
            if self.take_log_out == True:
                return np.exp(y) #trained on log data
            
            else:
                return y

        
    def load_saved_model(self, model_path = 'machine_interfaces/injector/', model_name = 'model_OTR_rms_emit'):
        
            self.model_1 = load_model(model_path + model_name +'.h5')
            self.savepath = model_path +'figures/'
            
    def load_scaling(self,scalerfilex = 'transformer_x.sav', scalerfiley = 'transformer_y.sav'):
        
            
            if scalerfilex[-3:] == 'sav':
            
                self.transformer_x = pickle.load(open(self.PATH+scalerfilex, 'rb'))

                self.transformer_y = pickle.load(open(self.PATH+scalerfiley, 'rb'))
                
                
    ## functions to convert between sim and machine units for data
    def sim_to_machine(self,sim_vals):
        pv_vals = np.copy(sim_vals)

        for i in range(0,len(self.model_in_list)):
            pv_vals[:,self.loc_in[self.model_in_list[i]]]=np.asarray(sim_vals)[:,self.loc_in[self.model_in_list[i]]]/self.pv_to_sim_factor[self.sim_name_to_pv_name[self.model_in_list[i]]]

        return pv_vals


    def machine_to_sim(self,pv_vals):

        sim_vals = np.copy(pv_vals)

        for i in range(0,len(self.model_in_list)):
            sim_vals[:,self.loc_in[self.model_in_list[i]]] = np.asarray(pv_vals)[:,self.loc_in[self.model_in_list[i]]]*self.pv_to_sim_factor[self.sim_name_to_pv_name[self.model_in_list[i]]]

        return sim_vals
            
