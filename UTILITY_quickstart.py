from pytao import Tao
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import re
import io
from os import path,environ
import pandas as pd
import random
from IPython.display import display, clear_output, update_display
import bayes_opt
import shutil
from scipy.special import jn as besselj

import scipy
from scipy.optimize import curve_fit

from pmd_beamphysics import ParticleGroup
#from pmd_beamphysics.statistics import resample_particles
import pmd_beamphysics.statistics

from UTILITY_plotMod import plotMod, slicePlotMod
from UTILITY_linacPhaseAndAmplitude import getLinacMatchStrings, setLinacPhase, setLinacGradientAuto
from UTILITY_modifyAndSaveInputBeam import modifyAndSaveInputBeam
from UTILITY_setLattice import setLattice, getBendkG, getQuadkG, getSextkG, setBendkG, setQuadkG, setSextkG, setXOffset, setYOffset
from UTILITY_impact import runImpact
from UTILITY_OpenPMDtoBmad import OpenPMD_to_Bmad

import os

filePathGlobal = None

def initializeTao(
    filePath = None,
    lastTrackedElement = "end",
    csrTF = False,
    inputBeamFilePathSuffix = None,
    numMacroParticles = None,
    loadDefaultLatticeTF = True,
    defaultsFile = None, 
    runImpactTF = False,
    impactGridCount = 32
):

    #######################################################################
    #Set file path
    #######################################################################
    global filePathGlobal 
    
    if not filePath:
        filePath = os.getcwd()
        
    os.environ['FACET2_LATTICE'] = filePath
    filePathGlobal = filePath
    
    print('Environment set to: ', environ['FACET2_LATTICE']) 

    
    #######################################################################
    #Launch and configure Tao
    #######################################################################
    tao=Tao('-init {:s}/bmad/models/f2_elec/tao.init -noplot'.format(environ['FACET2_LATTICE'])) 
    tao.cmd("set beam add_saved_at = DTOTR, XTCAVF, M2EX")

    tao.cmd(f'set beam_init track_end = {lastTrackedElement}') #See track_start and track_end values with `show beam`
    print(f"Tracking to {lastTrackedElement}")

    tao.cmd(f'call {filePath}/bmad/models/f2_elec/scripts/Activate_CSR.tao')
    if csrTF: 
        tao.cmd('csron')
        print("CSR on")
    else:
        tao.cmd('csroff')
        print("CSR off")


    if loadDefaultLatticeTF:
        print("Overwriting lattice with setLattice() defaults")
        setLattice(tao, verbose = True,  defaultsFile = defaultsFile) #Set lattice to my latest default config
        
    else:
        print("Base Tao lattice")
    

    #######################################################################
    #Import or generate input beam file
    #######################################################################
    if runImpactTF:
        if not numMacroParticles:
            print("Define numMacroParticles to run Impact")
            return
                  
        runImpact(
            filePath = filePath,
            gridCount = impactGridCount,
            numMacroParticles = numMacroParticles
        )

        inputBeamFilePath = f'{filePath}/beams/ImpactBeam.h5'

    else:
        if inputBeamFilePathSuffix:
            inputBeamFilePath = f'{filePath}{inputBeamFilePathSuffix}'
            
        else: #If tracking wasn't requested and a beamfile wasn't specified just grab a random beam... assume the user only wants to do single-particle sims
            print("WARNING! No beam file is specified!")
            inputBeamFilePath = f'{filePath}/beams/activeBeamFile.h5'
            

        if numMacroParticles:
            print(f"Number of macro particles = {numMacroParticles}")
        else:
            print(f"Number of macro particles defined by input file")
    
    modifyAndSaveInputBeam(
            inputBeamFilePath,
            numMacroParticles = (None if runImpactTF else numMacroParticles)
    )
    
    tao.cmd(f'set beam_init position_file={filePath}/beams/activeBeamFile.h5')
    tao.cmd('reinit beam')
    
    return tao

def trackBeam(
    tao,
    trackStart = "L0AFEND",
    trackEnd = "end",
    laserHeater = False,
    centerBC14 = False,
    centerBC20 = False,
    verbose = False,
    **kwargs,
):
    global filePathGlobal

    #For backwards compatibility, want this function to leave activeBeamFile unmodified
    #Introducing patchBeamFile for piecewise tracking
    #Note that, although initializeTao() asks for inputBeamFilePathSuffix, it will /always/ write it to activeBeamFile


    #Having second thoughts about this. Might be excessively paranoid?
    #shutil.copy(f'{filePathGlobal}/beams/activeBeamFile.h5', f'{filePathGlobal}/beams/patchBeamFile.h5') 
    #tao.cmd(f'set beam_init position_file={filePathGlobal}/beams/patchBeamFile.h5')
    #tao.cmd('reinit beam')

    #Instead, stick with always starting with activeBeamFile?
    tao.cmd(f'set beam_init position_file={filePathGlobal}/beams/activeBeamFile.h5')
    tao.cmd('reinit beam')
    if verbose: print("Loaded activeBeamFile.h5")
    
    tao.cmd(f'set beam_init track_start = {trackStart}')
    tao.cmd(f'set beam_init track_end = {trackEnd}')
    if verbose: print(f"Set track_start = {trackStart}, track_end = {trackEnd}")

    
    if laserHeater:
        #Will track from start to HTRUNDF, get the beam, modify it, export it, import it, update track_start and track_end
        tao.cmd(f'set beam_init track_end = HTRUNDF')
        if verbose: print(f"Set track_end = HTRUNDF")
        
        if verbose: print(f"Tracking!")
        tao.cmd('set global track_type = beam') #set "track_type = single" to return to single particle
        tao.cmd('set global track_type = single') #return to single to prevent accidental long re-evaluation

        P = getBeamAtElement(tao, "HTRUNDF", tToZ = False)

        PAfterLHmodulation, deltagamma, t = addLHmodulation(P, **kwargs,);
        
        writeBeam(PAfterLHmodulation, f'{filePathGlobal}/beams/patchBeamFile.h5')
        if verbose: print(f"Beam with LH modulation written to patchBeamFile.h5")

        tao.cmd(f'set beam_init position_file={filePathGlobal}/beams/patchBeamFile.h5')
        tao.cmd('reinit beam')
        if verbose: print("Loaded patchBeamFile.h5")

        tao.cmd(f'set beam_init track_start = HTRUNDF')
        tao.cmd(f'set beam_init track_end = {trackEnd}')
        if verbose: print(f"Set track_start = HTRUNDF, track_end = {trackEnd}")

    if centerBC14:
        tao.cmd(f'set beam_init track_end = BEGBC14_1')
        if verbose: print(f"Set track_end = BEGBC14_1")

        if verbose: print(f"Tracking!")
        tao.cmd('set global track_type = beam') #set "track_type = single" to return to single particle
        tao.cmd('set global track_type = single') #return to single to prevent accidental long re-evaluation

        P = getBeamAtElement(tao, "BEGBC14_1", tToZ = False)

        PMod = centerBeam(P)
        
        writeBeam(PMod, f'{filePathGlobal}/beams/patchBeamFile.h5')
        if verbose: print(f"Beam centered at BEGBC14 written to patchBeamFile.h5")

        tao.cmd(f'set beam_init position_file={filePathGlobal}/beams/patchBeamFile.h5')
        tao.cmd('reinit beam')
        if verbose: print("Loaded patchBeamFile.h5")

        tao.cmd(f'set beam_init track_start = BEGBC14_1')
        tao.cmd(f'set beam_init track_end = {trackEnd}')
        if verbose: print(f"Set track_start = BEGBC14_1, track_end = {trackEnd}")

    if centerBC20:
        tao.cmd(f'set beam_init track_end = BEGBC20')
        if verbose: print(f"Set track_end = BEGBC20")

        if verbose: print(f"Tracking!")
        tao.cmd('set global track_type = beam') #set "track_type = single" to return to single particle
        tao.cmd('set global track_type = single') #return to single to prevent accidental long re-evaluation

        P = getBeamAtElement(tao, "BEGBC20", tToZ = False)

        PMod = centerBeam(P)
        
        writeBeam(PMod, f'{filePathGlobal}/beams/patchBeamFile.h5')
        if verbose: print(f"Beam centered at BEGBC20 written to patchBeamFile.h5")

        tao.cmd(f'set beam_init position_file={filePathGlobal}/beams/patchBeamFile.h5')
        tao.cmd('reinit beam')
        if verbose: print("Loaded patchBeamFile.h5")

        tao.cmd(f'set beam_init track_start = BEGBC20')
        tao.cmd(f'set beam_init track_end = {trackEnd}')
        if verbose: print(f"Set track_start = BEGBC20, track_end = {trackEnd}")

    if verbose: print(f"Tracking!")
    tao.cmd('set global track_type = beam') #set "track_type = single" to return to single particle
    tao.cmd('set global track_type = single') #return to single to prevent accidental long re-evaluation

    if verbose: print(f"trackBeam() exiting")


    #For backwards compatibility, return to activeBeamFile. Might be unnecessary
    # tao.cmd(f'set beam_init position_file={filePathGlobal}/beams/activeBeamFile.h5')
    # tao.cmd('reinit beam')

def trackBeamLEGACY(tao):
    #This is the pre-2024-08-23 version of trackBeam(), retained for debugging purposes. Can be deleted
    
    tao.cmd('set global track_type = beam') #set "track_type = single" to return to single particle
    tao.cmd('set global track_type = single') #return to single to prevent accidental long re-evaluation

def getBeamAtElement(tao, eleString, tToZ = True):
    P = ParticleGroup(data=tao.bunch_data(eleString))
    P = P[P.status == 1]

    #Naive implementation for "typical" beams. ParticleGroup has .drift_to_z but I couldn't get it to work...
    if tToZ:
        P.z = -299792458 * P["delta_t"]
        #P.t = 0 * P.t #I haven't decided the best practice for this yet. Technically the beam is not self-consistent without t being set to zero but not doing so is convenient for backwards compatibility
        
    return P

def getDriverAndWitness(P):
    #See, e.g. "2024-07-01 Nudge Macroparticle Weights.ipynb" for details
    
    weights = np.sort(np.unique(P.weight))
    if len(weights) != 2:
        print("WARNING! Expected drive/witness structure not found")
        return
    PWitness = P[P.weight == weights[0]]
    PDrive = P[P.weight == weights[1]]
    return PDrive, PWitness

def writeBeam(P, fileName):
    """ Writes the beam as an h5 with E. Cropp's timeOffset fix """
    P.write(fileName)
    OpenPMD_to_Bmad(fileName)

def makeBeamActiveBeamFile(P):
    global filePathGlobal
    #P.write(f"{filePathGlobal}/beams/activeBeamFile.h5")
    writeBeam(P, f"{filePathGlobal}/beams/activeBeamFile.h5")

def smallestInterval(nums, percentage=0.9):
    """Give the smallest interval containing a desired percentage of provided points"""
    nums = nums.copy()  # Create a copy to avoid modifying the original list
    nums.sort()
    n = len(nums)
    k = int(n * percentage)
    min_range = float('inf')
    interval = (None, None)
    
    for i in range(n - k + 1):
        current_range = nums[i + k - 1] - nums[i]
        if current_range < min_range:
            min_range = current_range
            interval = (nums[i], nums[i + k - 1])
    
    return interval[1]-interval[0]


def smallestIntervalImpliedSigma(nums, percentage=0.9):
    interval = smallestInterval(nums, percentage)
    intervalToSigmaFactor = scipy.special.erfinv(percentage) * (2 * np.sqrt(2))
    return interval/intervalToSigmaFactor

#See "Discussion of alternative emittance and spot size calculations.ipynb"
#See also "2024-07-01 RMS vs FWHM at PENT.ipynb"
def smallestIntervalImpliedEmittanceModelFunction(z, sigmax, sigmaxp, rho):
    return np.sqrt(sigmax**2 + 2 * z * rho * sigmax * sigmaxp + z**2 * sigmaxp**2)

def smallestIntervalImpliedEmittance(P, plane = "x", percentage = 0.9, verbose = False):
    zValues = np.arange(-20, 20, 0.1)
    if plane == "x":
        sigmaXResults = [ smallestIntervalImpliedSigma(P.x + z * P.xp, percentage = percentage) for z in zValues]
        sigmaXResultsExact = [ np.std(P.x + z * P.xp) for z in zValues]
    elif plane == "y":
        sigmaXResults = [ smallestIntervalImpliedSigma(P.y + z * P.yp, percentage = percentage) for z in zValues]
        sigmaXResultsExact = [ np.std(P.y + z * P.yp) for z in zValues]
    else:
        return

    #sigmaXResults = [ smallestIntervalImpliedSigma(P.x + z * P.xp, percentage = percentage) for z in zValues]
    #sigmaXResultsExact = [ np.std(P.x + z * P.xp) for z in zValues]
    
    
    # Fit the model to the data
    popt, pcov = curve_fit(smallestIntervalImpliedEmittanceModelFunction, zValues, sigmaXResults, p0=[1, 1, 0])
    
    # Extract optimal parameters
    sigmax_opt, sigmaxp_opt, rho_opt = popt


    
    emit_opt = np.sqrt( sigmax_opt**2 * sigmaxp_opt**2 - (rho_opt * sigmax_opt * sigmaxp_opt)**2 )


    if verbose:
        print(f"""True sigma_x, sigma_xp, rho: {P.std("x")}, {P.std("xp")}, {P["cov_x__xp"] / (P.std("x") * P.std("xp"))}""")
        print(f"Optimizer parameters: sigma_x = {sigmax_opt}, sigma_xp = {sigmaxp_opt}, rho = {rho_opt}")
    

        plt.scatter(zValues, sigmaXResultsExact, label='True rms')
        plt.scatter(zValues, sigmaXResults, label='Inferred rms')
        plt.plot(zValues, smallestIntervalImpliedEmittanceModelFunction(zValues, *popt), label='Fitted function', color='red')
        plt.xlabel('Drift [m]')
        plt.ylabel('Sigma_x [m]')
        plt.legend()
        plt.show()

        print(f"""Actual emittance: \t {P["norm_emit_x"]}""")
        print(f"""Fit emittance: \t\t {emit_opt * P["mean_gamma"]}""")

    return emit_opt * P["mean_gamma"]
    
def displayMatrix(matrix):
    display(pd.DataFrame(matrix).style.hide(axis="index").hide(axis="columns"))

def getMatrix(tao, start, end, print = False):
    """Return first order transport matrix from start to end. Optionally print in a human readable format"""
    
    transportMatrix = (tao.matrix(start, end))["mat6"]

    if print:
        #display(pd.DataFrame(transportMatrix).style.hide(axis="index").hide(axis="columns"))
        displayMatrix(transportMatrix)
        
    return transportMatrix

def addLHmodulation(
    inputBeam, 
    #Elaser, 
    showplots=False,
    laserHeater_laserEnergy = 0.5e-3,
    laserHeater_sigma_t =  (2 / 2.355) * 1e-12,
    laserHeater_offset = -0.5
):
    """ From C. Emma, 2024-08-23 """
    # Hardcode FACET-II laser and undulator parameters
    # Laser parameters
    Elaser = laserHeater_laserEnergy
    lambda_laser = 760e-9
    sigmar_laser = 200e-6
    sigmat_laser = laserHeater_sigma_t # (2 / 2.355) * 1e-12
    Plaser = Elaser / np.sqrt(2 * np.pi * sigmat_laser**2)
    offset = laserHeater_offset #-0.5  # laser to e-beam offset if you want you can add it
    # Undulator parameters
    K = 1.1699
    lambdaw = 0.054
    Nwig = 9
    # Electron beam
    outputBeam = inputBeam.copy()
    x = inputBeam.x - np.mean(inputBeam.x)
    y = inputBeam.y - np.mean(inputBeam.y)
    gamma = inputBeam.gamma
    gamma0 = np.mean(inputBeam.gamma)
    t = inputBeam.t-np.mean(inputBeam.t);
    # Calculated parameters
    lambda_r = lambdaw / 2 / gamma0**2 * (1 + K**2 / 2)  # Assumes planar undulator
    omega = 2 * np.pi * 299792458 / lambda_r  # Resonant frequency
    JJ = besselj(0, K**2 / (4 + 2 * K**2)) - besselj(1, K**2 / (4 + 2 * K**2))
    totalLaserEnergy = np.sqrt(2 * np.pi * sigmat_laser**2) * Plaser
    # Laser is assumed Gaussian with peak power Plaser
    # This formula from eq. 8 Huang PRSTAB 074401 2004
    mod_amplitude = np.sqrt(Plaser / 8.7e9) * K * lambdaw * Nwig / gamma0 / sigmar_laser * JJ
    #print(mod_amplitude / np.sqrt(Plaser))
    # offset = 1.0  # temporal offset between laser and e-beam in units of laser wavelengths
    # Calculate induced modulation deltagamma
    deltagamma = mod_amplitude * np.exp(-0.25 * (x**2 + y**2) / sigmar_laser**2) * \
                 np.sin(omega * t + offset * 2 * np.pi) * \
                 np.exp(-0.5 * ((t - offset * sigmat_laser) / sigmat_laser)**2)
    outputBeam.gamma = inputBeam.gamma + deltagamma
    return outputBeam, deltagamma, t

def centerBeam(P):
    PMod = P
    PMod.x = P.x - np.mean(P.x)
    PMod.y = P.y - np.mean(P.y)
    PMod.px = P.px - np.mean(P.px)
    PMod.py = P.py - np.mean(P.py)

    return PMod