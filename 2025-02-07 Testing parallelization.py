import multiprocessing
import os

from UTILITY_quickstart import *


def worker(overrides):
    importedDefaultSettings = loadConfig("setLattice_configs/2024-10-22_oneBunch.yml")
    csrTF = True




    
    tao = initializeTao(
        inputBeamFilePathSuffix = importedDefaultSettings["inputBeamFilePathSuffix"],
        
        csrTF = csrTF,
        numMacroParticles=1e4,
    
        scratchPath = "/tmp",
        randomizeFileNames = True
    )




    
    activeSettings = importedDefaultSettings | overrides

    setLattice(tao, **activeSettings)
    trackBeam(tao, **activeSettings)
    
    P = getBeamAtElement(tao, "ENDBC14_2")
    BC14Length = smallestIntervalImpliedSigma(P.z)
    
    P = getBeamAtElement(tao, "PENT")
    PENTLength = smallestIntervalImpliedSigma(P.z)

    return f"""{overrides["L2PhaseSet"]}, {BC14Length}, {PENTLength}"""

    

if __name__ == "__main__":
    num_workers = 8
    tasks = [ {"L2PhaseSet" : L2PhaseSet} for L2PhaseSet in np.arange(-40,-30,0.2) ] 

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker, tasks)

    for res in results:
        print(res)