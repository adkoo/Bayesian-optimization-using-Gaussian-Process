# -*- coding: iso-8859-1 -*-

#This is the class that talks to the machine.

#To get it ready for a given machine, follow these steps. The steps will start from the bottom of the script and work upward so as to preserve referenced line numbers as you edit the code:

#1) On line 32, replace the expression to the right of the equal sign with an expression that queries the machine and returns a float representing the current objective value
#2) At line 29, add expressions to set the machine control pvs to the position called self.x -- Note: self.x is a 2-dimensional array of shape (1, ndim). To get the values as a 1d-array, use self.x[0]
#3) On line 22, replace the expression to the right of the equal sign with an expression that returns the current control pv values (i.e. the current x-position) from the machine as a 1-d array
#4) On line 20, replace the string to the right of the equal sign with whatever you want to call your machine. The name is inconsequential, except that you must not call it 'MultinormalInterface'
#5) At line 15, add any imports necessary for communicating with the machine in the expressions you have added as a result of the steps above


import numpy as np


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
        objective_state = np.exp(-0.5*self.x[0].dot(np.eye(len(self.pvs))).dot(self.x[0])) + 0.1*np.random.normal() #replace with expression that returns float representing current objective value
        return np.array(self.x, ndmin = 2), np.array([[objective_state]])