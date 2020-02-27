# Bayesian optimization using Gaussian Process

Bayesian Optimization using Gaussian Process to optimize an objective function with respect to input controls.

*NOTE:* This code is written for python 2.7

## 

This tool was developed primarly for particle accelerators with various machine interfaces. 
It has the capbabilty to run on:
1. Real machine (see 'Pegasus_UCLA.py' example)
2. Simulation (see 'Spear3_SLAC.py' example)
3. Surrogate model of the machine (see 'APS_ANL.py' example)
4. analytic function (see 'basic_example.py')


## Instructions
Step 1) Build your 'machine interface' file:
        This file will contain the class object that will represent your target function
        To build it, open machine_interfaces/machine_interface_example.py and
        follow the step-by-step instructions in the comments at the top of the file.

Step 2) Build your 'scan_params.npy' file:
        This is the .npy file that will contain the settings for the optimizer to load and use.
        To build it, open modules/make_scan_params_file.py and
        follow the step-by-step instructions in the comments at the top of the file.

Step 3) Run the optimization:
        -For the most basic usage, open basic_gp_example.py and 
        follow the step-by-step instructions in the comments at the top of the file.
        -Alternatively, you can try running gp_bayesopt_gui.py
        This should open a graphical user interface.
        Select your machine_interface and scan_params files from the dropdown menus as the top.
        Load the selections by clicking the button labeled 'load'.
        The selected scan parameters and machine state should then be displayed below.
        Usage is fairly self-explanatory, but here are some basic notes:
        The 'start scan' button always starts a scan from the current machine state.
        If you want to start the scan from device values shown in the 'Saved Device Values --' section,
        first hit reset, which returns the machine to those settings.
        If the 'save scan results' checkbox is checked, results will be saved to local directory 
        'saved_results/' and will be formatted as filename='scan_YYYY-MM-DD-hhmmss'. 
        The file is saved when the scan is terminated (whether by clicking 'stop scan' or 'exit'),
        so the time shown in the filename will be the time the scan was terminated.


basic_gp_example.py is an example code to run as-is without any input arguments. Simply run the file in python.
