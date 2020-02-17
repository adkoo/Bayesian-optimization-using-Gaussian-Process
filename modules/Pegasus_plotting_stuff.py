import numpy as np
import scipy.optimize as optimize
import time
import keras
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from keras.optimizers import SGD
import tensorflow
import pickle as pickle
import matplotlib.pyplot as plt
import json
from IPython.display import clear_output
import time


gsum_thresh=5e4
costs = []
xrmss = []
yrmss = []
xcs = []
ycs =[]
sigma_xys =[]
gsums =[]
q1s = []
q2s = []
q3s = []

def objfunc(xrms,yrms,xc,yc,sigma_xy,gsum,w1,w2,w3,w4,w5,w6):
    
    sigma_xy_scaled = 0 + ((np.abs(sigma_xy)- 0)*(1/5000))
    gsum_scaled = (0 + ((np.abs(gsum)-  gsum_thresh)*(1/gsum_thresh)))
    xrms_scaled = (0 + ((np.abs(xrms)- 0)*(1/200)))
    yrms_scaled = (0 + ((np.abs(yrms)- 0)*(1/200)))
    
    
    #if gsum < 1.0e7:
    #    penalty=np.abs(1/(gsum_scaled))
    #elif gsum> 1.0e7:
    #    penalty =0
    
    print(w1,w2,w3,w4,w5,w6)
    print(xrms,yrms,xc,yc,sigma_xy,gsum)
        
    cost_sxy = w5*sigma_xy_scaled
    
    cost_sxy_gs_xrms_yrms = w5*sigma_xy_scaled + w6*gsum_scaled + w1*xrms_scaled+ w2*yrms_scaled
    cost_sxy_gs_xrms_yrms_ratio = w5*sigma_xy_scaled + w6*gsum_scaled + w1*xrms_scaled/yrms_scaled
    
    cost_xrms_yrms = w1*xrms_scaled+ w2*yrms_scaled
    
    cost_xrms_over_yrms = w1*xrms_scaled/(-w2*yrms_scaled)
    
    cost_gs_xrms_yrms = w6*gsum_scaled + w1*xrms_scaled + w2*yrms_scaled
    
    cost_squared_ratio = (xrms/yrms)**2
    
    obj =  w5*sigma_xy_scaled + w1*((xrms_scaled)/(yrms_scaled)) + w3*xc + w4*yc

    #be sure to change if using pimax vs drz
    if gsum < gsum_thresh:
        penalty = w6*gsum_scaled
        obj = obj + penalty
        
    return obj

def func_machine(w1,w2,w3,w4,w5,w6):
               
    cursor.execute("rollback")
    cursor.execute("SELECT xrms,yrms,xc,yc,sigma_xy,gaussian_sum,rudyshot FROM image_analysis order by rudyshot desc limit "+ str(num_to_fetch))
    pimax = cursor.fetchall()

    xrms_all=np.asarray(pimax)[:,0]
    yrms_all=np.asarray(pimax)[:,1]
    xc_all=np.asarray(pimax)[:,2]
    yc_all=np.asarray(pimax)[:,3]
    gsum_all=np.asarray(pimax)[:,5]
    sigma_xy_all=np.asarray(pimax)[:,4]
    
    remove_list = []
    
    for i in range(0,len(gsum_all)):
        if gsum_all[i] <  gsum_thresh:
            remove_list.append(i)
    
    xrms_all=np.delete(xrms_all,remove_list)
    yrms_all=np.delete(yrms_all,remove_list)
    sigma_xy_all=np.delete(sigma_xy_all,remove_list)
    gsum_all=np.delete(gsum_all,remove_list) 
    xc_all=np.delete(xc_all,remove_list)
    yc_all=np.delete(yc_all,remove_list) 
        
    xrms=np.mean(xrms_all)
    yrms=np.mean(yrms_all)
    gsum=np.mean(gsum_all)
    sigma_xy=np.mean(sigma_xy_all)
    xc=np.mean(xc_all)
    yc=np.mean(yc_all)
    
    
    obj = objfunc(xrms,yrms,xc,yc,sigma_xy,gsum,w1,w2,w3,w4,w5,w6)

    return xrms, yrms, xc, yc, sigma_xy,gsum, obj



def store2(q11,q21,q31,w1,w2,w3,w4,w5,w6):
        x=[q11,q21,q31]
        #costs.append(error_func_machine_noset(w1=1.0, w2=-1.0,w5=0.0,w6=-0.2))
        
        q1s.append(q11)
        q2s.append(q21)
        q3s.append(q31)
        
        xrms1,yrms1,xc1,yc1,sigma_xy1,gsum1,cost1 = func_machine(w1,w2,w3,w4,w5,w6) # to do combine with above function call
        
        xrmss.append(xrms1)
        yrmss.append(yrms1)
        xcs.append(xc1)
        ycs.append(yc1)
        gsums.append(gsum1)
        sigma_xys.append(sigma_xy1)
        costs.append(cost1)
        
        clear_output(wait=True)
         
        f = plt.figure(figsize=(25,3))
        ax = f.add_subplot(141)
        ax2 = f.add_subplot(142)
        ax3 = f.add_subplot(143)
        ax4 = f.add_subplot(144)
        ax.plot(q1s)
        ax.set_ylabel('Q1',fontsize=12)
        ax2.plot(q2s, 'r')
        ax2.set_ylabel('Q2',fontsize=12)
        ax3.plot(q3s, 'g')
        ax3.set_ylabel('Q3',fontsize=12)
        ax4.plot(costs, 'k')
        ax4.set_ylabel('cost',fontsize=12)
        plt.show();

        f = plt.figure(figsize=(30,3))
        ax = f.add_subplot(151)
        ax2 = f.add_subplot(152)
        ax3 = f.add_subplot(153)
        ax4 = f.add_subplot(154)
        ax5 = f.add_subplot(155)
        ax.plot(xrmss)
        ax.set_ylabel('xrms',fontsize=12)
        ax2.plot(yrmss, 'r')
        ax2.set_ylabel('yrms',fontsize=12)
        ax3.plot(sigma_xys, 'g')
        ax3.set_ylabel('sigma_xy',fontsize=12)
        ax4.plot(gsums, 'k')
        ax4.set_ylabel('gsum',fontsize=12)
        ax5.plot(xcs, 'k')
        ax5.plot(ycs, 'r')
        ax5.set_ylabel('centroids',fontsize=12)
        plt.show();
        

        return costs,q1s,q2s,q3s,xrmss,yrmss,xcs,ycs,sigma_xys,gsums

