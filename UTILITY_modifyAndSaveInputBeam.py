from pmd_beamphysics import ParticleGroup
import numpy as np
import os
import random


def modifyAndSaveInputBeam(
    inputBeamFilePath,
    betaX = None,
    alphaX = None,
    betaY = None,
    alphaY = None,
    numMacroParticles = None,
    timeCenterTF = True):

    #Import
    P = ParticleGroup(inputBeamFilePath)

    #Downsample
    #if numMacroParticles:
    #    P.data.update(resample_particles(P, n=numMacroParticles))
    #PROBLEM! Built-in resampling smushes everything down to a single particle weight. No good for me since I'm using those to keep track of driver/witness
    #Instead, since the weights are ~equal, just pick a random subset then rescale their weights
    initialImportSize = np.size(P.x)
    if numMacroParticles:
        numMacroParticles = int(numMacroParticles)
        P = P[random.sample(range(initialImportSize), numMacroParticles)]
        P.weight = P.weight * (initialImportSize / numMacroParticles)
    

    #Time center
    if timeCenterTF:
        P.t=P.t-np.mean(P.t) #This is OK because present beam doesn't have different weights; np.unique(P.weight)

    #Apply linear matching
    if (betaX is not None) and (alphaX is not None):
        P.twiss_match(
              plane='x',
              beta = betaX,
              alpha = alphaX,
              inplace=True)

    if (betaY is not None) and (alphaY is not None):
        P.twiss_match(
              plane='y',
              beta = betaY,
              alpha = alphaY,
              inplace=True)

    filePath = os.getcwd()
    #Write as the active file
    P.write(f'{filePath}/beams/activeBeamFile.h5')

    #Also return the beam object
    return P
