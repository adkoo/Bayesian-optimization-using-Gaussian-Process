# -*- coding: iso-8859-1 -*-



import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import Tkinter as tk  # python 2.7
import ttk            # python 2.7
import sys
import os
import importlib
from datetime import datetime
from modules.bayes_optimization import BayesOpt, negUCB, negExpImprove
from modules.OnlineGP import OGP

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self,master)
        self.master = master
        self.params_dir = 'params/'
        self.createWidgets()
        self.master.protocol('WM_DELETE_WINDOW', self.exit)

    def createWidgets(self):
        # create all of the main containers
        left_frame = tk.Frame(self.master)#, width=240, height=height)
        right_frame = tk.Frame(self.master, bg='gray2')#, width=width, height=height)
        left_frame.grid(row=0, column=0)
        right_frame.grid(row=0, column=1)

        #partition the right frame
        top_right_frame = tk.Frame(right_frame)#, width=width, height=height/2.)
        bot_right_frame = tk.Frame(right_frame)#, width=width, height=height/2.)
        top_right_frame.pack()
        bot_right_frame.pack()

        #partition the left frame
        self.top_left= tk.Frame(left_frame)
        self.upper_left = tk.Frame(left_frame)
        self.mid_left = tk.Frame(left_frame)
        self.lower_left = tk.Frame(left_frame)
        self.bot_left = tk.Frame(left_frame)
        self.very_bot_left = tk.Frame(left_frame)
        self.top_left.pack(fill=tk.X, expand=1)
        self.upper_left.pack(fill=tk.X, expand=1)
        self.mid_left.pack(fill=tk.X, expand=1)
        self.lower_left.pack(fill=tk.X, expand=1)
        self.bot_left.pack()
        self.very_bot_left.pack()

        #Build the top left frame
        self.machine_interface_file = tk.StringVar(self.top_left)
        self.loaded_machine = tk.StringVar(self.top_left)
        self.scan_params_file = tk.StringVar(self.top_left)
        self.loaded_params = tk.StringVar(self.top_left)
        self.machine_interface_file.set('machine_interface_example') # default value
        self.scan_params_file.set("scan_params_example.npy") # default value
        self.loaded_machine.set(self.machine_interface_file.get())
        self.loaded_params.set(self.scan_params_file.get())
        option_menu_label = tk.Label(self.top_left, text='Choose Scan Parameter File:')
        self.option_menu = tk.OptionMenu(self.top_left, self.scan_params_file, *os.listdir('params'))
        option_menu_label.grid(row=1, column=0, sticky='w')
        self.option_menu.grid(row=1, column=1, sticky='e')
        mi_option_menu_label = tk.Label(self.top_left, text='Choose Machine Interface:')
        mi_list = [item[:-3] for item in os.listdir('machine_interfaces') if (item[-3:]=='.py' and item[0]!='_') ]
        self.mi_option_menu = tk.OptionMenu(self.top_left, self.machine_interface_file, *mi_list)
        mi_option_menu_label.grid(row=0, column=0, sticky=tk.W)
        self.mi_option_menu.grid(row=0, column=1, sticky=tk.E)
        self.load_button = tk.Button(master=self.top_left, text='Load', command=self.load)
        self.load_button.grid(row=0, column=2, sticky=tk.W, rowspan=2)
        config_label0 = tk.Label(self.top_left, text='Currently loaded:')
        config_label1 = tk.Label(self.top_left, textvariable=self.loaded_machine)
        config_label2 = tk.Label(self.top_left, text=' with ')
        config_label3 = tk.Label(self.top_left, textvariable=self.loaded_params)
        config_label0.grid(row=3, column=0)
        config_label1.grid(row=2, column=1)
        config_label2.grid(row=3, column=1)
        config_label3.grid(row=4, column=1)


        #build the upper-mid left frame
        self.amp_textvar = tk.StringVar(self.upper_left)
        self.noise_textvar = tk.StringVar(self.upper_left)
        self.precisionmat_textvar = tk.StringVar(self.upper_left)
        self.ucb_params_textvar = tk.StringVar(self.upper_left)
        hyp_label = tk.Label(self.upper_left, text='GP Hyperparameters --')
        amp_label = tk.Label(self.upper_left, text='Amplitude=')
        noise_label = tk.Label(self.upper_left, text='Noise=')
        precisionmat_label = tk.Label(self.upper_left, text='Precision Matrix=')
        ucb_params_label = tk.Label(self.upper_left, text='UCB params [nu, delta]=')

        hyp_label.grid(row=0, column=0, sticky=tk.W)
        amp_label.grid(row=1, column=0, sticky=tk.E)
        noise_label.grid(row=2, column=0, sticky=tk.E)
        precisionmat_label.grid(row=3, column=0, sticky=tk.N+tk.E)
        ucb_params_label.grid(row=4, column=0, sticky=tk.E)

        amp_val_label = tk.Label(self.upper_left, textvariable=self.amp_textvar)
        noise_val_label = tk.Label(self.upper_left, textvariable=self.noise_textvar)
        precisionmat_val_label = tk.Label(self.upper_left, textvariable=self.precisionmat_textvar)
        ucb_params_val_label = tk.Label(self.upper_left, textvariable=self.ucb_params_textvar)

        amp_val_label.grid(row=1, column=1, sticky=tk.W)
        noise_val_label.grid(row=2, column=1, sticky=tk.W)
        precisionmat_val_label.grid(row=3, column=1, sticky=tk.W)
        ucb_params_val_label.grid(row=4, column=1, sticky=tk.W)


        #build the bottom left frame
        self.start_button=tk.Button(master=self.bot_left, text="start scan", command=lambda: self.start_scan())
        self.start_button.grid(row=2,column=0)

        self.stop_button=tk.Button(master=self.bot_left, text="stop scan", command=self.stop_scan)
        self.stop_button.grid(row=2,column=1)
        self.stop_button.config(relief=tk.SUNKEN, state=tk.DISABLED)

        self.reset_button = tk.Button(master=self.bot_left, text='reset', command=self.revert_devices_to_saved)
        self.reset_button.grid(row=2, column=2)

        self.exit_button=tk.Button(master=self.bot_left, text="exit", command=self.exit)
        self.exit_button.grid(row=2,column=3)

        self.error_textvar = tk.StringVar(self.very_bot_left)
        self.error_label = tk.Label(master=self.very_bot_left, textvariable=self.error_textvar)
        self.error_label.pack()

        self.save_resultsQ = tk.IntVar()
        tk.Checkbutton(self.very_bot_left, text='Save scan results', var=self.save_resultsQ, command=self.save_results).pack()


        #build the top right frame
        fig1=plt.figure(figsize=(5,3), dpi=80)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        self.ax1=fig1.add_subplot(111)
        self.canvas1=FigureCanvasTkAgg(fig1,master=top_right_frame)
        self.canvas1.get_tk_widget().pack()#fill=tk.BOTH, expand=1)

        #build the bottom right frame
        fig2=plt.figure(figsize=(5,3), dpi=80)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        self.ax2=fig2.add_subplot(111)
        self.canvas2=FigureCanvasTkAgg(fig2,master=bot_right_frame)
        self.canvas2.get_tk_widget().pack()#fill=tk.BOTH, expand=1)

        self.plot()



        #load the variables and finish building
        self.load()




    def exit(self):
        try:self.stop_scan()
        except: pass
        self.master.destroy()
        plt.close('all')
        sys.exit()

    def update_saved_vals(self):
        self.update_machine_state()
        for i, textvar in enumerate(self.saved_device_val_textvars):
            textvar.set(self.device_val_textvars[i].get())
            try:
                self.saved_values[i] = float(self.device_val_textvars[i].get())
            except:
                pass

    def build_mid_left_frame(self):
        for widget in self.mid_left.winfo_children():
            widget.destroy()
        saved_vals_label = tk.Label(self.mid_left, text='Saved Device Values --')
        self.update_saved_vals_button = tk.Button(self.mid_left, text='Update to Current Vals', command=self.update_saved_vals)
        saved_vals_label.grid(row=0,column=0, sticky=tk.W)
        self.update_saved_vals_button.grid(row=0,column=1, sticky=tk.E)
        self.saved_device_val_textvars = []
        for i, dev_id in enumerate(self.dev_ids):
            device_label = tk.Label(self.mid_left, text=dev_id+'=')
            self.saved_device_val_textvars += [tk.StringVar(self.mid_left)]
            device_val_label = tk.Label(self.mid_left, textvariable=self.saved_device_val_textvars[i])
            device_label.grid(row=1+i, column=0, sticky=tk.E)
            device_val_label.grid(row=1+i, column=1, sticky=tk.W)

    def build_lower_left_frame(self):
        for widget in self.lower_left.winfo_children():
            widget.destroy()
        self.update_time_textvar = tk.StringVar(self.lower_left)
        self.objective_val_textvar = tk.StringVar(self.lower_left)
        machine_state_label = tk.Label(self.lower_left, text='Machine State -- updated at:')
        machine_state_time = tk.Label(self.lower_left, textvariable=self.update_time_textvar)
        self.machine_state_refresh_button = tk.Button(master=self.lower_left, text='refresh', command=self.update_machine_state)
        objective_label = tk.Label(self.lower_left, text='Objective Function Value=')
        objective_val = tk.Label(self.lower_left, textvariable=self.objective_val_textvar)        
        machine_state_label.grid(row=0, column=0, sticky=tk.E)
        machine_state_time.grid(row=0, column=1, sticky=tk.W)
        self.machine_state_refresh_button.grid(row=0, column=2, sticky=tk.W)
        objective_label.grid(row=1, column=0, sticky=tk.E)
        objective_val.grid(row=1, column=1, sticky=tk.W)

        self.device_val_textvars = []
        for i, dev_id in enumerate(self.dev_ids):
            device_label = tk.Label(self.lower_left, text=dev_id+'=')
            self.device_val_textvars += [tk.StringVar(self.lower_left)]
            device_val_label = tk.Label(self.lower_left, textvariable=self.device_val_textvars[i])
            device_label.grid(row=2+i, column=0, sticky=tk.E)
            device_val_label.grid(row=2+i, column=1, sticky=tk.W)

    def load(self):
        try:

            self.loaded_machine.set(self.machine_interface_file.get())
            self.loaded_params.set(self.scan_params_file.get())
            self.mi_module = importlib.import_module('machine_interfaces.'+self.machine_interface_file.get())

            #load the dict that contains the parameters for the scan (control pv list, starting settings, and gp hyperparams)
            scan_params = np.load(self.params_dir+self.scan_params_file.get(), allow_pickle=True).item()

            #create the machine interface
            self.dev_ids = scan_params['dev_ids']
            self.saved_values = scan_params['start_point'] #if start_point is set to None, the optimizer will start from the current device settings.

            self.build_mid_left_frame()
            self.build_lower_left_frame()

            if type(self.saved_values) != type(None):
                for i, textvar in enumerate(self.saved_device_val_textvars):
                    textvar.set(self.saved_values[i])
            else: 
                self.update_saved_vals()




            self.mi = self.mi_module.machine_interface(dev_ids = self.dev_ids, start_point = None) #an isotropic n-dimensional gaussian with amplitude=1, centered at the origin, plus gaussian background noise with std dev = 0.1

            #create the gp
            self.ndim = len(self.dev_ids)
            self.acquisition_delay = scan_params['acquisition_delay']
            self.gp_precisionmat = scan_params['gp_precisionmat']
            self.gp_amp = scan_params['gp_amp'] 
            self.gp_noise = scan_params['gp_noise']
            self.ucb_params = scan_params['ucb_params']

            self.file_error(None)
        
        except:
#             pass
            self.file_error('Invalid machine interface or parameter file. Try again.')
            #create the machine interface
            self.dev_ids = None
            self.saved_values = None #if start_point is set to None, the optimizer will start from the current device settings.
            self.mi = None

            #create the gp
            self.ndim = None
            self.gp_precisionmat = None
            self.gp_amp = None 
            self.gp_noise = None
        self.amp_textvar.set(str(self.gp_amp))
        self.noise_textvar.set(str(self.gp_noise))
        self.precisionmat_textvar.set(str(self.gp_precisionmat))
        self.ucb_params_textvar.set(str(self.ucb_params))
        try:
            self.update_machine_state()
        except:
            pass

    def file_error(self, message):
        if type(message) == type(None):
            self.error_textvar.set('')
        else:
            self.error_textvar.set('Error Message: '+message)

    def revert_devices_to_saved(self):
        self.mi.setX(self.saved_values)
        self.update_machine_state()
        self.plot()

    def update_machine_state(self):
        self.update_time_textvar.set(datetime.now().strftime('%H:%M:%S'))
        try:
            x, y = self.mi.getState()
            x = x[0]; y=y[0][0]
            self.objective_val_textvar.set(round(y,2))
            for i, textvar in enumerate(self.device_val_textvars):
                textvar.set(round(x[i],2))
        except:
            self.objective_val_textvar.set('Error')
            for i, textvar in enumerate(self.device_val_textvars):
                textvar.set('Error')

    def stop_scan(self):
        if self.continueScanning==False:
            return
        self.stop_button.config(relief = tk.SUNKEN, state=tk.DISABLED)
        self.start_button.config(relief=tk.RAISED, state=tk.NORMAL)
        self.reset_button.config( state=tk.NORMAL)
        self.update_saved_vals_button.config(state=tk.NORMAL)
        self.machine_state_refresh_button.config( state=tk.NORMAL)
        self.load_button.config(state=tk.NORMAL)
        self.continueScanning=False
        if self.save_resultsQ.get() == True:
            self.save_results()

    def plot(self, xs=None, ys=None):
        self.ax1.clear()
        self.ax2.clear()
        if type(xs) != type(None) and type(ys) != type(None):
            self.ax1.plot(ys)
            self.ax2.plot(xs)
        self.ax1.set_title('Objective Function Monitor')
        self.ax1.set_xlabel('Step Number')
        self.ax1.set_ylabel('Objective Function Value')
        self.ax2.set_title('Control Device Displacements')
        self.ax2.set_xlabel('Step Number')
        self.ax2.set_ylabel('Displacement')
        self.canvas1.draw()
        self.canvas2.draw()

    def scan(self):
        if self.continueScanning:
            self.opt.OptIter()
            ys = [y[0][0] for y in self.opt.Y_obs]
            xs = np.copy(self.opt.X_obs)
            xs -= xs[0];
            self.plot(xs=xs, ys=ys)
            self.update_machine_state()
            self.master.after(int(self.acquisition_delay*1000), self.scan)

    def make_opt(self):

        #create the machine interface
        #if start_point is set to None, the optimizer will start from the current device settings.
        self.mi = self.mi_module.machine_interface(dev_ids = self.dev_ids, start_point = self.mi.x) #an isotropic n-dimensional gaussian with amplitude=1, centered at the origin, plus gaussian background noise with std dev = 0.1

        #create the gp
        hyps = [self.gp_precisionmat, np.log(self.gp_amp), np.log(self.gp_noise**2)] #format the hyperparams for the OGP
        gp = OGP(self.ndim, hyps)

        #create the bayesian optimizer that will use the gp as the model to optimize the machine 
        opt = BayesOpt(gp, self.mi, acq_func="UCB", start_dev_vals = self.mi.x, dev_ids = self.dev_ids)
        opt.ucb_params = self.ucb_params
        return opt

    def start_scan(self):
        self.stop_button.config(relief=tk.RAISED, state=tk.NORMAL)
        self.start_button.config(relief=tk.SUNKEN, state=tk.DISABLED)
        self.reset_button.config( state=tk.DISABLED)
        self.update_saved_vals_button.config(state=tk.DISABLED)
        self.machine_state_refresh_button.config( state=tk.DISABLED)
        self.load_button.config(state=tk.DISABLED)
        sys.stdout = open(os.devnull, "w")
        self.opt = self.make_opt()
        sys.stdout = sys.__stdout__
        self.continueScanning = True
        self.scan()
        
    def save_results(self):
        timestr = datetime.now().strftime('%Y-%m-%d-%H%M%S')
        try: os.mkdir('saved_results')
        except: pass
        results = {}
        results['scan_params'] = np.load(self.params_dir+self.scan_params_file.get(), allow_pickle=True).item()
        results['xs'] = self.opt.X_obs
        results['ys'] = np.array([y[0][0] for y in self.opt.Y_obs])
        results['time'] = timestr
        np.save('saved_results/scan_'+timestr, results)

            
        
root=tk.Tk()
root.title('Bayesian Optimization GP GUI')
root.geometry('{}x{}'.format(840,560))
app=Application(master=root)
app.mainloop()