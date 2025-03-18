# import multiprocessing as mp
# from mpi4py import MPI
import xopt
import concurrent.futures
import os
import time
from UTILITY_quickstart import *

multiplicity_count = 1 # 36 # 24
tasks_per_node = 1 # 140
num_tasks = int(multiplicity_count * tasks_per_node)
print(f"Multiplicity: {multiplicity_count}")
print(f"Tasks per node: {tasks_per_node}")
print(f"Number of jobs requested: {num_tasks}")
print(f"Available cores on node: {len(os.sched_getaffinity(0))}")

importedDefaultSettings = loadConfig(f'setLattice_configs/2025-02-25_oneBunch_baseline.yml')

#############################################
#        Make Synthetic Distribution        #
# From E. Cropp RedPill Helper_functions.py #
#############################################

def Gaussian_Dist_Maker(n,mu,sigma,lSig,rSig):
    """
    This function returns a truncated gaussian distribution of quasi-random particles.  This uses the Halton series
    
    Argument:
    n -- int number of particles
    mu -- float: center of distribution/mean
    sigma -- float: std of distribution
    lSig -- float number of sigma at which to truncate Gaussian left
    rSig -- float number of sigma at which to truncate Gaussian right
    """
    # Check inputs
    try: n = int(n)
    except: raise ValueError("n is not an int!")
    
    try: mu = float(mu)
    except: raise ValueError("mu is not a float!")
    
    try: sigma = float(sigma)
    except: raise ValueError("sigma is not a float!")
    
    try: lSig = float(lSig)
    except: raise ValueError("lSig is not a float!")
    
    try: rSig = float(rSig)
    except: raise ValueError("rSig is not a float!")
    
    
    # get and shuffle n samples from halton series
    h=scipy.stats.qmc.Halton(1)
    X0=h.random(n=n)
    np.random.shuffle(X0)
    
    # Make these into Gaussian and return
    X0=X0*(1-(1-scipy.stats.norm.cdf(lSig))-(1-scipy.stats.norm.cdf(rSig)))
    X0=X0+(1-scipy.stats.norm.cdf(lSig))
    GaussDist = mu + np.sqrt(2)*sigma*scipy.special.erfinv(2*X0-1)
    return np.squeeze(GaussDist)


##################################################
# Initialize nominal parameters and jitter range #
##################################################

# I think L0B, L1, L2, L3 amplitudes can be jittered by adding/subtracting to <Stage>EnergyOffset.
nominal_charge             = 1600 # pC
nominal_gun_theta0_deg     = 29.3-90
nominal_gun_rf_field_scale = 119/2.44885*1e6
nominal_L0A_Phase          = 29
nominal_L0A_Amp            = 30e6
nominal_L0B_Phase          = importedDefaultSettings['L0BPhaseSet']
nominal_L0BF_Amp           = 5.95e7
nominal_L1_Phase           = importedDefaultSettings['L1PhaseSet']
nominal_L1_Amp             = 0.335e9 - 0.125e9
nominal_L2_Phase           = importedDefaultSettings['L2PhaseSet']
nominal_L2_Amp             = 4.5e9 - 0.335e9
nominal_L3_Phase           = importedDefaultSettings['L3PhaseSet']
nominal_L3_Amp             = 10.0e9 - 4.5e9

# values obtained from `2024 run` column of https://docs.google.com/spreadsheets/d/1xeCUImz5uFSq6QA3wV91dG38s-8cyVXQMGw9hjPKa6M/edit?usp=sharing
charge_jitter_percent  = 2.3
gun_Phase_jitter       = 0.15
gun_Amp_jitter_percent = 0.25
L0A_Phase_jitter       = 0.1 # L0APhaseOffset kwarg passed directly into `initializeTao`. Note that it's an *offset*.
L0A_Amp_jitter_percent = 0.06
L0B_Phase_jitter       = 0.1
L0B_Amp_jitter_percent = 0.5
L1A_Phase_jitter       = 0.7
L1A_Amp_jitter_percent = 0.6
L1B_Phase_jitter       = 0.5
L1B_Amp_jitter_percent = 0.7
L2_Phase_jitter        = 0.4
L2_Amp_jitter_percent  = 0.3
L3_Phase_jitter        = 0.4
L3_Amp_jitter_percent  = 0.3

L_str = 'L2'

cal_data = {
    f'{L_str}PhaseSet':[nominal_L2_Phase - L2_Phase_jitter, nominal_L2_Phase + L2_Phase_jitter],
#    f'L1PhaseSet':[nominal_L1_Phase - L1A_Phase_jitter, nominal_L1_Phase + L1A_Phase_jitter],
}

# Beam output locations
locations = ['L0AFEND','ENDINJ','BEGL1F','ENDL1F','BC11CEND','ENDL2F','ENDBC14_2','ENDL3F_2','BEGFF20','ENDFF20','PENT']

# Diagnostic keys
diagnostic_keys = [
    'energy',
    'peak current 50',
    'peak current 100',
    'peak current 200',
    'sigma_x',
    'sigma_y',
    'sigma_z',
    'emittance_x',
    'emittance_y',
    'pmd_emittance_x',
    'pmd_emittance_y',
    'energy spread',
]

#################################
# Make evaluation points        #
#################################

cutoff_sigma = 3

# Quasi-random Gaussian
points = {}
for key in cal_data.keys():
    mu = np.mean(np.array(cal_data[key]))
    sigma = np.ptp(np.array(cal_data[key]))/(2*cutoff_sigma)
    dist = Gaussian_Dist_Maker(num_tasks,mu,sigma,cutoff_sigma,cutoff_sigma)
    points[key] = dist

# multiproc
# turn dict of arrays into array of dicts
# tasks = [
#    {key: points[key][i] for key in cal_data.keys()}
#    for i in range(num_tasks)
# ]

# xopt
points = pd.DataFrame(points)


##################################
# Parallelizable jitter function #
##################################

path_conda = '/global/homes/m/maxvarv/miniforge3/envs/bmad/bin/'

output_path = f'/pscratch/sd/m/maxvarv/Jitter_2025_02_25/xopt_24hr_{L_str}_Phase'
os.makedirs(output_path, exist_ok=True)

def worker(overrides):
    start_time = time.time()    

    # initialize diagnostics dict
    diagnostics = { loc: {key: 0.0 for key in diagnostic_keys} for loc in locations }
    
    tao, unique_ID = initializeTao(
            inputBeamFilePathSuffix = importedDefaultSettings["inputBeamFilePathSuffix"],
            GFILESuffix = f'2024-10-22_distgen_onebunch.yaml',
            csrTF = True,
            runImpactTF = False,
            # impactGridCount = 32,          # unused when `runImpactTF = False`
            # numMacroParticles = 5 * 32**3, # unused when `runImpactTF = False` (still technically used if uncommented, but we want to use the h5 input file to define this)
            # solenoidTValue = -0.41, # uncomment if explicitly running impact-T!
            # impactChargepC = point['ChargepC'], # adjust bunch charge (when jittering in impact-T),
            command = path_conda + 'ImpactTexe',    
            command_mpi = path_conda + 'ImpactTexe-mpi',
            mpi_run = '/global/u1/m/maxvarv/miniforge3/envs/bmad/bin/mpirun --map-by :OVERSUBSCRIBE -n {nproc} {command_mpi}',
            scratchPath = output_path,
            randomizeFileNames = True,
    	)

    #activeSettings = importedDefaultSettings | overrides
    setLattice(tao, **importedDefaultSettings)
    disableAutoQuadEnergyCompensation(tao)
    
    try:
        #setLattice(tao, **activeSettings)
        #disableAutoQuadEnergyCompensation(tao)
        #trackBeam(tao, **activeSettings)
        
        setLinacPhase(tao, L_str, overrides[f'{L_str}PhaseSet']) # adjust L2 Phase
        
        trackBeam(tao, **importedDefaultSettings)
    
        result = {'Unique ID': unique_ID}

        # multiproc
        # result.update(overrides)
        
        # make directory if it doesn't already exist
        os.makedirs(f'{output_path}/{unique_ID}', exist_ok=True)

        # multiproc
        # result_df = pd.DataFrame(result, index=[0])
        # result_df.to_csv(f'{output_path}/{unique_ID}/result.csv', index=False)
        
        for location in locations:
            P = getBeamAtElement(tao, location)

            diagnostics[location]['energy'          ] = P['mean_energy']
            diagnostics[location]['peak current 50' ] = P.slice_statistics(slice_key='t', n_slice=50)['current'].max()
            diagnostics[location]['peak current 100'] = P.slice_statistics(slice_key='t', n_slice=100)['current'].max()
            diagnostics[location]['peak current 200'] = P.slice_statistics(slice_key='t', n_slice=200)['current'].max()
            diagnostics[location]['sigma_x'         ] = smallestIntervalImpliedSigma(P.x) # horizontal spot size
            diagnostics[location]['sigma_y'         ] = smallestIntervalImpliedSigma(P.y) # vertical spot size
            diagnostics[location]['sigma_z'         ] = smallestIntervalImpliedSigma(P.z) # bunch length
            diagnostics[location]['emittance_x'     ] = smallestIntervalImpliedEmittance(P, plane = 'x', percentage = 0.9)
            diagnostics[location]['emittance_y'     ] = smallestIntervalImpliedEmittance(P, plane = 'y', percentage = 0.9)
            diagnostics[location]['pmd_emittance_x' ] = P.norm_emit_x
            diagnostics[location]['pmd_emittance_y' ] = P.norm_emit_y
            diagnostics[location]['energy spread'   ] = P.std('energy')/P.avg('energy')
            
    	    # write to output_path
            P.write(f'{output_path}/{unique_ID}/{location}.h5')
    finally:
        tao.close_subprocess()

        
        diagnostics = pd.DataFrame(diagnostics)
        diagnostics.to_csv(f'{output_path}/{unique_ID}/diagnostics.csv')
        
        # already captured by 'xopt_runtime'
        # compute_time = (time.time() - start_time) / 60
        # result['Compute time [min]'] = compute_time

        print(f'Run {unique_ID} elapsed time: {(time.time() - start_time) / 60:.1f} minutes')
        
    return result


if __name__ == '__main__':
   
    ##################################################
    # Define multiprocess object for parameter scans # 
    ##################################################
    
    ####################################################
    # single-node implementation
    ####################################################
    # with mp.Pool(tasks_per_node) as pool:
    #    results = pool.map(worker, tasks)
    
    # jitter_numbers = pd.DataFrame(results)
    # jitter_numbers.to_csv(f'{output_path}/jitter_numbers.csv', index=False)


    ####################################################
    # Define Low-Level Xopt Object for Parameter Scans # 
    ####################################################
    
    evaluator1=xopt.evaluator.Evaluator(
                                        function=worker, 
                                        function_kwargs={},
                                        max_workers=tasks_per_node,
                                        executor=concurrent.futures.ProcessPoolExecutor(max_workers=tasks_per_node),
                                        vectorized=False,
                                       )
    
    results = pd.DataFrame.from_dict( evaluator1.evaluate_data(points) )
    results.to_csv(f'{output_path}/xopt_out.csv', index=False)



