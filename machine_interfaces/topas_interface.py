# -*- coding: iso-8859-1 -*-
import numpy as np
import pickle
from scipy.optimize import minimize
import subprocess
import sys, os
from pathlib import Path
import logging
from matplotlib import pyplot as plt
import numpy as np




## add my PhaserGeom Repository to path:
sys.path.append('/home/brendan/python/Phaser_Models/topas')
sys.path.append('/home/brendan/python/Phaser_Models/PhaserGeometry')
sys.path.append('/home/brendan/python/Phaser_Models/AnalysisCodes')

from PhaserBeamLine import PhaserBeamLine
from PhaseSpaceAnalyser import ElectronPhaseSpace
from TopasScriptGenerator2 import TopasScriptGenerator
from DoseAnalyser import WaterTankData
# from XXXXX import *
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel

class machine_interface:
    def __init__(self, dev_ids, start_point=None):
        self.name = 'topas_interface' 
        
        #setup topas simulation enviroment 
        self.GenerateTopasModel()
        self.RunTopasModel()
        
        #setup the devices name (input controls)
        self.pvs = np.array(dev_ids)
        if start_point is None:
            current_x = np.zeros(len(self.pvs)) #replace with expression that reads current ctrl pv values (x) from machine
            self.setX(current_x)
        else: 
            self.setX(start_point)
 
    def setX(self, x_new):
        self.x = np.array(x_new, ndmin=2)
            

    def getState(self): #objective
        """
        Return the objective value for the optimisation problem
        """

        objective_state = self.BeamWidthObjective(self.x[0][0]) #BeamWidthObjective.py (in machine_interface folder)
        return np.array(self.x, ndmin = 2), np.array([[objective_state]])


class bcolors:
    """
    This is just here to enable me to print pretty colors to the linux terminal
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SingleChannelOptimiser:
    """
    the purpose of this code is to run a optimsiation of the central channel for the sphinx collimator.
    The goal is to maximise the dose/dose rate while maintaining TargetBeamWidth. This is a work in progress.

    .. figure:: ../doc_source/_static/07012021_BeamletWidths.png
        :align: center
        :alt: FreeCAD setup
        :scale: 100%
        :figclass: align-center
    """

    def __init__(self, BaseDirectory, SimulationName, TargetBeamWidth=7, debug=False):
        """
        Set up the parameters to be optimised and bounds.
        """

        self.BaseDirectory = BaseDirectory
        if not os.path.isdir(BaseDirectory):
            logging.error(
                f'{bcolors.FAIL}Input BaseDirectory "{BaseDirectory}" does not exist. Exiting. {bcolors.ENDC}')
            sys.exit()
        self.SimulationName = SimulationName
        SimName = Path(self.BaseDirectory) / self.SimulationName
        if not os.path.isdir(SimName):
            os.mkdir(SimName)
        self.TargetBeamWidth = TargetBeamWidth
        self.MaxItterations = 100
        self.Itteration = 0
        self.Nthreads = -6  ## number of threads to run. -3 means use all but 3. 0 means use all.
        # the starting values of our optimisation parameters are defined from the default geometry
        self.ParameterNames = ['USHS', 'BSI',
                               'CT']  # these are just labels, they don't have to be the same as the real names
        self.StartingValues = [0.5, 2, 100]
        self.UpperBounds = [1, 3, 150]
        self.LowerBounds = [.25, .5, 50]
        self.CollThicknessLowerLimit = 75  # lower collimator limit in mm
        self.CollThicknessUpperLimit = 150  # upper collimator limit  in mm
        self.AllObjectiveFunctionValues = []
        self.CheckInputData()
        if debug:
            self.BremPSlocation = '../QuickDirtyBremPhaseSpace/Results/brem_PhaseSpace_xAng_0.00_yAng_0.00_dum'
        else:
            self.BremPSlocation = '../QuickDirtyBremPhaseSpace/Results/brem_PhaseSpace_xAng_0.00_yAng_0.00'
        # set some boundaries for parameters to optimse:

    def CheckInputData(self):
        """
        check that the user has put in reasonable parameters
        """
        # do the number of parameters match?
        if not np.size(self.ParameterNames) == np.size(self.StartingValues):
            print(f'{bcolors.FAIL} size of ParameterNames does not match size of .StartingValues{bcolors.ENDC}')
            sys.exit()
        if not np.size(self.StartingValues) == np.size(self.UpperBounds):
            print(f'{bcolors.FAIL} size of StartingValues does not match size of UpperBounds{bcolors.ENDC}')
            sys.exit()
        if not np.size(self.UpperBounds) == np.size(self.LowerBounds):
            print(f'{bcolors.FAIL} size of UpperBounds does not match size of LowerBounds{bcolors.ENDC}')
            sys.exit()

        for i, Paramter in enumerate(self.ParameterNames):
            if self.StartingValues[i] < self.LowerBounds[i]:
                print(f'{bcolors.FAIL}For {Paramter}, Starting value {self.StartingValues[i]} is less than '
                      f'Lower bound {self.LowerBounds[i]}{bcolors.ENDC}')
                sys.exit()
            elif self.StartingValues[i] > self.UpperBounds[i]:
                print(f'{bcolors.FAIL}For {Paramter}, Starting value {self.StartingValues[i]} is greater '
                      f'than upper bound {self.UpperBounds[i]}{bcolors.ENDC}')

    def GenerateTopasModel(self, x):
        """
        Generates a topas model with the latest parameters as well as a shell script called RunAllFiles.sh to run it.

        You are required to manually code the values into the call to PhaserBeamLine
        """

        # if RunAllFiles exists, delete it
        ShellScriptLocation = str(Path(self.BaseDirectory) / self.SimulationName / 'RunAllFiles.sh')
        if os.path.isfile(ShellScriptLocation):
            # I don't think I should need to do this; the file should be overwritten if it exists, but this doesn't
            # seem to be working so deleting it.
            os.remove(ShellScriptLocation)
        self.ShellScriptLocation = ShellScriptLocation

        PhaserGeom = PhaserBeamLine(verbose=False, coll_UpStreamHoleSize=x[0], BeamletSizeAtIso=x[1],
                                    coll_CollimatorThickness=x[2])
        ParameterString = f'run_{self.Itteration}'
        for i, Paramter in enumerate(self.ParameterNames):
            ParameterString = ParameterString + f'_{Paramter}_{x[i]:1.1f}'
        self.TopasScript = TopasScriptGenerator(self.BaseDirectory, self.SimulationName, PhaserGeom,
                                                AnglesToRun='Central', WaterTankVoxelSizeZ=5,
                                                PhysicsSettings='g4em-standard_opt0',
                                                UseMappedMagneticField=False, UseBremSource=True, Overwrite=True,
                                                UsePhaseSpaceSource=False,
                                                ParametricVariable=ParameterString,
                                                BremPhaseSpaceLocation=self.BremPSlocation,
                                                WT_ScoringFieldSize=self.TargetBeamWidth * 3,
                                                Nthreads=self.Nthreads,GenerateScripts=False)

    def RunTopasModel(self):
        """
        This invokes a bash subprocess to run the current model
        """
        print(f'{bcolors.OKBLUE}Running file: \n{self.ShellScriptLocation}')
        ShellScriptPath = str(Path(self.BaseDirectory) / self.SimulationName)
        cmd = subprocess.run(['bash', self.ShellScriptLocation], cwd=ShellScriptPath)
        print(f'{bcolors.OKBLUE}Analysis complete{bcolors.ENDC}')

        # update the definition of current model
        self.CurrentWTdata = self.TopasScript.WT_PhaseSpaceName_current

    def ReadTopasResults(self):
        """
        Read in topas results and extract quantities of interest

        Returns two parameters:
        BeamletWidth (double, mm)
        MaxDose (double, AU)
        """

        DataLocation = str(Path(self.BaseDirectory) / self.SimulationName / 'Results')

        Dose = WaterTankData(DataLocation, self.CurrentWTdata,
                             AbsDepthDose=True, MirrorData=False)
        # store the relevant quantities:
        self.BeamletWidth = Dose.BeamletWidthsGauss[0]
        self.MaxDose = Dose.MaxDose[0]

    def BeamWidthObjective(self, x):
        """
        Define a beam width objective function, which should basically do:
        Maximise dose while maintaining beamwidth = targetBeamWidth

        At the moment all pretty crude, mainly designed to get the workflow up and running.
        """
        if self.Itteration > self.MaxItterations:
            logging.warning(f'{bcolors.WARNING} Max number of itterations exceeded; terminating code')
            sys.exit()
        # check that 'guess' is within our boundaries
        for i, xval in enumerate(x):
            if xval < self.LowerBounds[i]:

                try:
                    self.DummyOF + self.DummyOF + 100  # make sure this keeps getting bigger if it keeps going in the wrong direction
                except AttributeError:
                    self.DummyOF = self.AllObjectiveFunctionValues[-1] + 100
                print(
                    f'{bcolors.WARNING}Values below lower bound attempted, returning high cost function: '
                    f'{self.DummyOF}{bcolors.ENDC}')
                self.UpdateOptimisationLogs(x, self.DummyOF)
                self.AllObjectiveFunctionValues.append(self.DummyOF)
                self.AllParameterValues = np.vstack([self.AllParameterValues, x])
                self.PlotResults()
                self.Itteration = self.Itteration + 1
                return self.DummyOF
            if xval > self.UpperBounds[i]:
                try:
                    self.DummyOF + self.DummyOF + 100  # make sure this keeps getting bigger if it keeps going in the wrong direction
                except AttributeError:
                    self.DummyOF = self.AllObjectiveFunctionValues[-1] + 100
                print(
                    f'{bcolors.WARNING}Values above lower bound attempted, returning high cost function:'
                    f' {self.DummyOF}{bcolors.ENDC}')
                self.UpdateOptimisationLogs(x, self.DummyOF)
                self.AllObjectiveFunctionValues.append(self.DummyOF)
                self.AllParameterValues = np.vstack([self.AllParameterValues, x])
                self.PlotResults()
                self.Itteration = self.Itteration + 1
                return self.DummyOF

        self.GenerateTopasModel(x)
        self.RunTopasModel()
        self.ReadTopasResults()

        if self.Itteration == 0:
            # normalise such that our first 'max dose' parameter is 1, and subsequent calls are normlaised to this
            self.StartingDose = self.MaxDose
            self.MaxDose = 1
        else:
            self.MaxDose = self.MaxDose / self.StartingDose

        try:
            # if nothing goes wrong, this is the place we want to end up; this is where the objective function is
            # defined
            w1 = 200
            w2 = 500
            w3 = 100
            w4 = 200

            DeltaBeamletWidth = abs(self.TargetBeamWidth - self.BeamletWidth)
            CollimatorThicknessMinTerm = min(0, self.CollThicknessLowerLimit - x[2]) ** 2
            CollimatorThicknessMaxTerm = min(0, self.CollThicknessUpperLimit - x[2]) ** 2
            OF = (-w1 * self.MaxDose) + (DeltaBeamletWidth * w2) + (
                        w4 * CollimatorThicknessMaxTerm)  # the more negative this function the better the answer
            self.AllObjectiveFunctionValues.append(OF)
            try:
                self.AllParameterValues = np.vstack([self.AllParameterValues, x])
            except AttributeError:
                self.AllParameterValues = x
            self.UpdateOptimisationLogs(x, OF)
            self.PlotResults()
            try:
                if OF > self.DummyOF:
                    self.DummyOF = OF
            except AttributeError:
                pass
            self.Itteration = self.Itteration + 1
            return OF
        except:
            print(f'{bcolors.FAIL}Something went wrong when calculating objective function. Heres the data:')
            for i, parameter in enumerate(self.ParameterNames):
                print(f'{parameter}: {x[i]}')
            raise
            print(f'Will just return a high value of OF so the algorithm keeps running')
            try:
                self.DummyOF + self.DummyOF + 100  # make sure this keeps getting bigger if it keeps going in the wrong direction
            except AttributeError:
                self.DummyOF = self.AllObjectiveFunctionValues[-1]
            self.UpdateOptimisationLogs(x, self.DummyOF)
            self.PlotResults()
            self.Itteration = self.Itteration + 1

            return self.DummyOF

    def UpdateOptimisationLogs(self, x, OF):
        """
        Just a simple function to keep track of the objective function in the logs folder
        """

        LogFile = Path(self.BaseDirectory) / self.SimulationName
        LogFile = LogFile / 'logs'
        LogFile = str(LogFile / 'OptimisationLogs.txt')
        with open(LogFile, 'a') as f:
            Entry = f'Itteration: {self.Itteration}'
            for i, Parameter in enumerate(self.ParameterNames):
                Entry = Entry + f', {Parameter}: {x[i]}'
            Entry = Entry + f', BeamletWidth: {self.BeamletWidth}'
            Entry = Entry + f', Dose: {self.MaxDose}'
            Entry = Entry + f', ObjectiveFunction: {OF}\n'
            f.write(Entry)
        print(f'{bcolors.OKGREEN}{Entry}{bcolors.ENDC}')

    def PlotResults(self):

        if self.Itteration < 2:
            # dont waste my time
            return

        ItterationVector = np.arange(self.Itteration + 1)
        FigureLocation = Path(self.BaseDirectory) / self.SimulationName
        FigureLocation = FigureLocation / 'logs'
        FigureLocation = str(FigureLocation / 'OptimisationLogs.png')

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        axs[0].plot(ItterationVector, self.AllObjectiveFunctionValues)
        axs[0].set_xlabel('Itteration number')
        axs[0].set_ylabel('Objective function')
        axs[0].grid(True)

        MinValue = np.argmin(self.AllObjectiveFunctionValues)
        axs[0].plot(ItterationVector[MinValue], self.AllObjectiveFunctionValues[MinValue], 'r-x')

        LegendStrings = []
        for i, ParameterVals in enumerate(self.AllParameterValues.T):
            LegendStrings.append(f'{self.ParameterNames[i]}: InitVal = {self.StartingValues[i]}')
            ParameterVals = ParameterVals / ParameterVals[0]
            axs[1].plot(ItterationVector, ParameterVals)

        axs[1].legend(LegendStrings)
        for ParameterVals in self.AllParameterValues.T:
            ParameterVals = ParameterVals / ParameterVals[0]
            axs[1].plot(ItterationVector[MinValue], ParameterVals[MinValue], 'r-x')
        axs[1].set_xlabel('Itteration number')
        axs[1].set_ylabel('Parameter value [mm]')
        axs[1].grid(True)

        SaveLoc = Path(self.BaseDirectory) / self.SimulationName
        SaveLoc = SaveLoc / 'logs'
        plt.savefig(SaveLoc / 'Results.png')

    def RunOptimisation(self):
        """
        Use the scipy.optimize.minimize module to perform the optimisation.
        Not that this function repeatedly call BeamWidthObjective; this is
        where all the 'magic' happens
        """

        res = minimize(self.BeamWidthObjective, self.StartingValues, method='Nelder-Mead',
                       options={'xatol': 1e-1, 'fatol': 1e-1, 'disp': True})

    def RunOptimisationAdi(self):
        """
        This uses Adi's suggestions for implementation of the simplex optimisation algorithm.
        It's not working yet.
        """
        lengthscales = np.array([.1, .3])
        lengthscales *= np.sign(np.random.randn(lengthscales.size))
        isim = np.zeros((len(lengthscales) + 1, len(lengthscales)))
        isim[0, :] = self.StartingValues
        for i in range(len(x)):
            vertex = np.zeros(len(x))
            vertex[i] = lengthscales[i]
            isim[i + 1, :] = x + vertex  # vertex

        # fmin(self.BeamletWidth, self.StartingValues, maxiter=self.MaxItterations,
        #               maxfun=self.MaxItterations, xtol=0.1, ftol=1, initial_simplex=isim,
        #               disp=True, full_output=True)


class spear_sim_interface:
    def __init__(self, dev_ids, precision_matrix=None, start_point=None):

        # setup spear simulation enviroment
        self.mlab = matlabThread()
        self.mlab.run()
        self.mlab.mlabinterface.eval('clearvars')  # clear variables
        self.mlab.mlabinterface.eval('setpathspear3')  # set
        self.mlab.mlabinterface.eval('spear_sim_run')
        print('set spear3 simulation successfully!  ')

        self.pvs = np.array(dev_ids)
        self.name = 'spear_sim_interface'
        if type(start_point) == type(None):
            self.x = np.array(np.zeros(len(self.pvs)), ndmin=2)
        else:
            self.x = np.array(start_point, ndmin=2)
        if type(precision_matrix) == type(None):
            self.precision_matrix = np.eye(len(self.pvs))
        else:
            self.precision_matrix = precision_matrix

        self.noise_std = 0.0  # FIX ME! this is the g_noise variable in the matlab sim

        self.ratio = 0.00245246  # current = SkewK_LOCO/0.00245246;

    def setX(self, x_new):
        self.x = np.array(x_new)  # this is in machine units [-30,30] mA
        self.x_sim = self.x / 60 + 0.5  # Multiply by ratio = 0.00245246 to convert to gradient in the range (-0.1,0.1) [m**-2] and normalize between 0 to 1 for the simulation
        self.mlab.mlabinterface.workspace.x0 = np.array(np.array(self.x_sim).flatten(), ndmin=2)

    #         print ('self.x',self.x)
    #         print ('self.x_sim', self.x_sim)
    #         print ('self.mlab.mlabinterface.workspace.x0 ',self.mlab.mlabinterface.workspace.x0 )

    def getState(self):
        self.x_sim = self.x / 60 + 0.5  # simulation quads range is between [0,1]
        self.mlab.mlabinterface.workspace.x0 = np.array(np.array(self.x_sim).flatten(), ndmin=2)
        self.mlab.mlabinterface.eval('obj = spear_sim_obj_median(x0)')  # spear_sim_func
        self.objective_state0 = - self.mlab.mlabinterface.workspace.obj  # + self.noise_std * np.random.randn()  # we want to maximuze, so take neg of objective
        objective_state = self.objective_state0 + self.noise_std * np.random.randn()  # we want to maximuze, so take neg of objective
        print('current state: ', np.array(self.x, ndmin=2), np.array([[objective_state]]))

        return np.array(self.x, ndmin=2), np.array([[objective_state]])

    def getState0(self):
        return np.array(self.objective_state0)

    def get_value(self, device):
        index = np.arange(0, len(self.pvs), 1)[np.array(self.pvs) == device]
        return self.x[-1][index][0]


class matlabThread(QtCore.QThread):
    def __init__(self, parent=None):
        QtCore.QThread.__init__(self, parent)

    def run(self):
        self.mlabinterface = matlab_wrapper.MatlabSession()



if __name__ == '__main__':
    BaseDirectory = os.path.expanduser("~") + '/Dropbox (Sydney Uni)/Projects/PhaserSims/topas'
    SimulationName = 'SingleCollimatorOptimisation_test3'
    # Optimisation files will be stored BaseDirectory/SimulationName.
    # BaseDirectory must already exist. SimulationName name will be created if it doesn't exist.
    SCO = SingleChannelOptimiser(BaseDirectory, SimulationName, debug=True)
    SCO.RunOptimisation()
    # SCO.PlotLogFile()
    #



