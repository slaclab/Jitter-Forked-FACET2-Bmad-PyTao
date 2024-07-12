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

from pmd_beamphysics import ParticleGroup
#from pmd_beamphysics.statistics import resample_particles
import pmd_beamphysics.statistics

from UTILITY_plotMod import plotMod, slicePlotMod
from UTILITY_linacPhaseAndAmplitude import getLinacMatchStrings, setLinacPhase, setLinacGradientAuto
from UTILITY_modifyAndSaveInputBeam import modifyAndSaveInputBeam
from UTILITY_setLattice import setLattice, getBendkG, getQuadkG, getSextkG, setBendkG, setQuadkG, setSextkG
from UTILITY_impact import runImpact

import os

filePathGlobal = None

def initializeTao(
    filePath = None,
    lastTrackedElement = "end",
    csrTF = False,
    inputBeamFilePathSuffix = None,
    numMacroParticles = None,
    loadDefaultLatticeTF = True,
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
        setLattice(tao, verbose = True) #Set lattice to my latest default config
        
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

def trackBeam(tao):
    tao.cmd('set global track_type = beam') #set "track_type = single" to return to single particle
    tao.cmd('set global track_type = single') #return to single to prevent accidental long re-evaluation

def getBeamAtElement(tao, eleString):
    P = ParticleGroup(data=tao.bunch_data(eleString))
    P = P[P.status == 1]
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

def makeBeamActiveBeamFile(P):
    global filePathGlobal
    P.write(f"{filePathGlobal}/beams/activeBeamFile.h5")

def smallestInterval(nums, percentage=0.9):
    """Give the smallest interval containing a desired percentage of provided points"""
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