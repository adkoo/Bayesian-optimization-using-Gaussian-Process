# Bayesian optimization using Gaussian Process

Bayesian Optimization using Gaussian Process to optimize an objective function with respect to input controls.

For the most basic usage, run basic_gp_example.py, which is an example code to run as-is without any input arguments. 

*NOTE:* This code is written for python 2.7

## 

This tool was developed primarily for particle accelerators with various machine interfaces. 
It has the capability to run on:
1. Real machine (see 'Pegasus_UCLA.py' example)
2. Simulation (see 'Spear3_SLAC.py' example)
3. Surrogate model of the machine (see 'APS_ANL.py' example)
4. Analytic function (see 'basic_example.py') 

## Instructions to build your own optimizer

**Step 1.** Build your 'machine interface' file:
                   This file will contain the class object that will represent your target function. 
        To build it, open machine_interfaces/machine_interface_example.py and follow the step-by-step instructions in the comments at the top of the file.

**Step 2.**  Build your 'scan_params.npy' file:
                   This is the .npy file that will contain the settings for the optimizer to load and use. To build it, open modules/make_scan_params_file.py and follow the step-by-step instructions in the comments at the top of the file.

**Step 3.**  Run the optimization: 
                   For the most basic usage, open basic_gp_example.py and follow the step-by-step instructions in the comments at the top of the file.
      
      

