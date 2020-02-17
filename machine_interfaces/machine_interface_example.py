# -*- coding: iso-8859-1 -*-

#This is the class that talks to the machine.

#To get it ready for a given machine, follow these steps. The steps will start from the bottom of the script and work upward so as to preserve referenced line numbers as you edit the code:

#1) On line 32, replace the expression to the right of the equal sign with an expression that queries the machine and returns a float representing the current objective value
#2) At line 29, add expressions to set the machine control pvs to the position called self.x -- Note: self.x is a 2-dimensional array of shape (1, ndim). To get the values as a 1d-array, use self.x[0]
#3) On line 22, replace the expression to the right of the equal sign with an expression that returns the current control pv values (i.e. the current x-position) from the machine as a 1-d array
#4) On line 20, replace the string to the right of the equal sign with whatever you want to call your machine. The name is inconsequential, except that you must not call it 'MultinormalInterface'
#5) At line 15, add any imports necessary for communicating with the machine in the expressions you have added as a result of the steps above

import numpy as np
# import tensorflow as tf
# import keras
# import keras.models.load_model as load_model
# # from keras.models import Sequential, Model,load_model
# model=load_model('peg.h5')
# scalerfile = 'transformer_x.sav'
# transformer_x = pickle.load(open(scalerfile, 'rb'))
# scalerfile = 'transformer_y.sav'
# transformer_y = pickle.load(open(scalerfile, 'rb'))
# print('loading scaler files')

class machine_interface:
    def __init__(self, dev_ids, start_point = None):
        self.pvs = np.array(dev_ids)
        self.name = 'basic_multinormal' #name your machine interface. doesn't matter what you call it as long as it isn't 'MultinormalInterface'.
        if type(start_point) == type(None):
            current_x = np.zeros(len(self.pvs)) #replace with expression that reads current ctrl pv values (x) from machine
            self.setX(current_x)
        else: 
            self.setX(start_point)

    def setX(self, x_new):
        self.x = np.array(x_new, ndmin=2)
        # add expressions to set machine ctrl pvs to the position called self.x -- Note: self.x is a 2-dimensional array of shape (1, ndim). To get the values as a 1d-array, use self.x[0]

    def getState(self): 
        objective_state = np.exp(-0.5*self.x[0].dot(np.eye(len(self.pvs))).dot(self.x[0])) + 0.01*np.random.normal() #replace with expression that returns float representing current objective value

        
        #         objective_state = error_func_tst()          
        return np.array(self.x, ndmin = 2), np.array([[objective_state]])
    
    
    
#     def error_func_tst(self,w1 = 1.0, w2 = -1.0, w5 = 0.3,w6 = -1.0):
#         #print(x)
#         #np.append(ins,x.reshape((1,3)),axis=0)
#         x = self.x
#         scale = np.array([1.08821033e-02, 1.00992249e-02, 4.37141815e-08,  1.79058524e-05]) # xrms, yrms, gsum, sigma_xy
#         ymin = -1
#         data_min = np.array([ 1.24920000e+01,  2.93010000e+01, 3.00054899e+06,-5.75928790e+04])  # xrms, yrms, gsum, sigma_xy
#         a = np.empty((1,16))
#         a[:]=np.array([-3.40000000e-01,  7.80000000e-01,  1.00000000e-01, -7.60000000e-01,
#             2.00000000e-02, -5.00753000e-01,  1.10000000e-01, -1.18813100e+00,
#             2.20000100e+00, -3.25999700e+00,  6.20000100e+00,  0.00000000e+00,
#             0.00000000e+00,  6.09094900e+01,  1.02743800e+01,  7.51183071e+04]).reshape((1,16))
#         a[:]=np.array([-3.40000000e-01,  7.80000000e-01,  1.00000000e-01, -7.60000000e-01,
#             2.00000000e-02, -8.34422000e-01,  1.10000000e-01, -1.12540400e+00,
#             2.20000000e+00, -3.26000000e+00,  6.16700000e+00,  0.00000000e+00,
#             0.00000000e+00,  6.18860300e+01,  7.67205000e+00,  9.44071445e+04]).reshape((1,16))


#         a[0,8]=x[0]
#         a[0,9]=x[1]
#         a[0,10]=x[2]

#         a= transformer_x.transform(a)

#         out = model.predict(a)
#         #print(out)

#         xrms= (out[0,0]   + data_min[0] * scale[0] - ymin)/scale[0]
#         yrms=(out[0,1]   + data_min[1] * scale[1] - ymin)/scale[1]
#         gsum=(out[0,4]  + data_min[2] * scale[2] - ymin)/scale[2]
#         sigma_xy=(out[0,5]   + data_min[3] * scale[3] - ymin)/scale[3]


#         sigma_xy_scaled = 0 + ((np.abs(sigma_xy)- 0)*(1/5000))
#         gsum_scaled = (0 + ((np.abs(gsum)- 8e6)*(1/8e7)))
#         xrms_scaled = (0 + ((np.abs(xrms)- 0)*(1/200)))
#         yrms_scaled = (0 + ((np.abs(yrms)- 0)*(1/200)))
   

#         cost_sxy = w5*sigma_xy_scaled

#         cost_sxy_gs_xrms_yrms = w5*sigma_xy_scaled + w6*gsum_scaled + w1*xrms+ w2*yrms
#         cost_sxy_gs_xrms_yrms_ratio = w5*sigma_xy_scaled + w6*gsum_scaled + w1*xrms_scaled/yrms_scaled


#         cost_xrms_yrms = w1*xrms_scaled+ w2*yrms_scaled

#         cost_xrms_over_yrms = w1*xrms_scaled/(-w2*yrms_scaled)

#         cost_gs_xrms_yrms = w6*gsum_scaled + w1*xrms_scaled + w2*yrms_scaled

#         cost_squared_ratio = (xrms/yrms)**2

#         obj = cost_sxy_gs_xrms_yrms

      

#         return obj 