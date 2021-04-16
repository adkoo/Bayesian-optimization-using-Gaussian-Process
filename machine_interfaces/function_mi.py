# -*- coding: iso-8859-1 -*-

import numpy as np

class machine_interface:
    def __init__(self, dev_ids, start_point = None, funcobj=['booth']):
        self.pvs = np.array(dev_ids)
        self.name = 'function_mi' 
        self.funcobj = funcobj
        self.my_counter=0
        self.my_x=[]
        self.my_y=[]
        if type(start_point) == type(None):
            current_x = np.zeros(len(self.pvs)) 
            self.setX(current_x)
        else: 
            self.setX(start_point)
        
    def func(self,x):
        if self.funcobj == 'gaussian':
            y= self.a + self.b*np.exp(-0.5*np.dot(x,np.dot(self.Sigma,np.transpose(x)))) 
        elif self.funcobj == 'x_4':
            y = np.dot(x,x)
            y *= y 
        elif self.funcobj == 'x_2':
            y= -np.dot(x,x)
        elif self.funcobj == 'x_2_sin':
            y= -np.sin(3*x) - np.dot(x,x) + 0.7*x   
        elif self.funcobj == 'booth':
            print('here')
            y = (x[0]+2*x[1]-7)**2 + (x[1]+2*x[0]-5)**2 
        print('here',self.funcobj)
        self.my_counter+=1
        self.my_x += [x]  
        self.my_y += [y]
        
        return y  # + 0.0001*np.random.normal() 

    def setX(self, x_new):
        self.x = np.array(x_new, ndmin=2)
       
    def getState(self): 
        objective_state = self.func(self.x[0])
        return np.array(self.x, ndmin = 2), np.array([[objective_state]])
    
 
    