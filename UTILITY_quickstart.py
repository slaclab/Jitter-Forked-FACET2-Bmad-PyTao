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

from UTILITY_plotNMM import plotNMM, slicePlotNMM
from UTILITY_linacPhaseAndAmplitude import getLinacMatchStrings, setLinacPhase, setLinacGradientAuto
from UTILITY_modifyAndSaveInputBeam import modifyAndSaveInputBeam
from UTILITY_setLattice import setLattice, getBendkG, getQuadkG, getSextkG

import os



def initializeTao(
    filePath = None,
    lastTrackedElement = "end",
    csrTF = False,
    inputBeamFilePathSuffix = None,
    numMacroParticles = None,
    loadDefaultLatticeTF = True,
):

    if not filePath:
        filePath = os.getcwd()
    os.environ['FACET2_LATTICE'] = filePath
    
    print('Environment set to: ', environ['FACET2_LATTICE']) 
    
    tao=Tao('-init {:s}/bmad/models/f2_elec/tao.init -noplot'.format(environ['FACET2_LATTICE'])) 

    tao.cmd("set beam add_saved_at = DTOTR, XTCAVF, M2EX")


    tao.cmd(f'set beam_init track_end = {lastTrackedElement}') #Can see present track_start and track_end values with `show beam`
    print(f"Tracking to {lastTrackedElement}")

    tao.cmd(f'call {filePath}/bmad/models/f2_elec/scripts/Activate_CSR.tao')
    if csrTF: 
        tao.cmd('csron')
        print("CSR on")
    else:
        tao.cmd('csroff')
        print("CSR off")
    

    inputBeamFilePath = f'{filePath}{inputBeamFilePathSuffix}'

    if numMacroParticles:
        print(f"Number of macro particles = {numMacroParticles}")
    else:
        print(f"Number of macro particles defined by input file")
    
    modifyAndSaveInputBeam(
        inputBeamFilePath,
        numMacroParticles = numMacroParticles
    )
    
    tao.cmd(f'set beam_init position_file={filePath}/beams/activeBeamFile.h5')
    tao.cmd('reinit beam')

    if loadDefaultLatticeTF:
        #Set lattice to my latest default config
        setLattice(tao)
        print("Loading default setLattice() values")
    else:
        print("Base Tao lattice")
    
    return tao

def trackBeam(tao):
    tao.cmd('set global track_type = beam') #set "track_type = single" to return to single particle
    tao.cmd('set global track_type = single') #return to single to prevent accidental long re-evaluation