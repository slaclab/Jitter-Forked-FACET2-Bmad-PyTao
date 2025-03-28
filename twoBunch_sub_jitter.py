import xopt
import concurrent.futures
import os
import time
import sys
from UTILITY_quickstart import *
from UTILITY_setLattice import *
from UTILITY_linacPhaseAndAmplitude import getLinacMatchStrings, setLinacPhase, setLinacGradientAuto

multiplicity_count = 4
tasks_per_node = 60
num_tasks = int(multiplicity_count * tasks_per_node)
print(f"Multiplicity: {multiplicity_count}")
print(f"Tasks per node: {tasks_per_node}")
print(f"Number of jobs requested: {num_tasks}")
print(f"Available cores on node: {len(os.sched_getaffinity(0))}")

# Capture the number of calls index (given via slurm) to determine the slice of data to use
sim_call_count = int(sys.argv[1])

# Print the call index
print(f'sim_call_count={sim_call_count}')

importedDefaultSettings = loadConfig(f'setLattice_configs/2025-02-25_oneBunch_baseline.yml')

# Beam output locations
locations = ['L0AFEND','ENDINJ','BEGL1F','ENDL1F','BC11CEND','ENDL2F','ENDBC14_2','ENDL3F_2','BEGFF20','MFFF','XC1FF','YC1FF','YC2FF','XC3FF','ENDFF20','PENT']

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

path_conda = '/global/homes/m/maxvarv/miniforge3/envs/bmad/bin/'

output_path = '/pscratch/sd/m/maxvarv/twoBunch/L123_Energy_Offset_Scan'
os.makedirs(output_path, exist_ok=True)

# get evaluation points
eval_pts = pd.read_csv(f'{output_path}/evaluation_points.csv')
points = eval_pts[sim_call_count*num_tasks:(sim_call_count+1)*num_tasks]


##################################
# Parallelizable jitter function #
##################################

def worker(overrides):
    start_time = time.time()    

    # initialize diagnostics dict
    diagnostics = { loc: {key: 0.0 for key in diagnostic_keys} for loc in locations }
    beam_specs = {}
    
    tao, unique_ID = initializeTao(
            inputBeamFilePathSuffix = importedDefaultSettings["inputBeamFilePathSuffix"],
            GFILESuffix = 'distgen_twobunch.yaml',
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

    [L1MatchStrings, L2MatchStrings, L3MatchStrings, selectMarkers] = getLinacMatchStrings(tao)
    
    setLattice(tao, **importedDefaultSettings)
    disableAutoQuadEnergyCompensation(tao)
    
    try:
                
        # without energy compensation
        # setLinacPhase(tao, L_str, overrides[f'{L_str}PhaseSet']) # adjust L2 Phase

        setLinacGradientAuto( tao, L1MatchStrings, overrides['L1EnergyOffset'] + 0.335e9 - 0.125e9 )
        setLinacGradientAuto( tao, L2MatchStrings, overrides['L2EnergyOffset'] + 4.5e9 - 0.335e9 )
        setLinacGradientAuto( tao, L3MatchStrings, overrides['L3EnergyOffset'] + 10.0e9 - 4.5e9 )
        
        # setAllFinalFocusKickers(tao,
        #                 latticeSettings["XC1FFkG"],
        #                 latticeSettings["XC3FFkG"],
        #                 latticeSettings["YC1FFkG"],
        #                 latticeSettings["YC2FFkG"]
        #                 )

        trackBeam(tao)
    
        result = {'Unique ID': unique_ID}
        
        # make directory if it doesn't already exist
        os.makedirs(f'{output_path}/{unique_ID}', exist_ok=True)
        
        for location in locations:
            P = getBeamAtElement(tao, location)

            beam_specs[location] = getBeamSpecs(P)
            
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
        
        beam_specs = pd.DataFrame(beam_specs)
        beam_specs.to_csv(f'{output_path}/{unique_ID}/beam_specs.csv')

        # early_out = {'Unique ID': unique_ID}
        # early_out.update(overrides)
        # early_out = pd.DataFrame(early_out)
        # early_out.to_csv(f'{output_path}/{unique_ID}/early_out.csv')
        
        print(f'Run {unique_ID} elapsed time: {(time.time() - start_time) / 60:.1f} minutes')
        
    return result


if __name__ == '__main__':
    
    evaluator1=xopt.evaluator.Evaluator(
                                        function=worker, 
                                        function_kwargs={},
                                        max_workers=tasks_per_node,
                                        executor=concurrent.futures.ProcessPoolExecutor(max_workers=tasks_per_node),
                                        vectorized=False,
                                       )
    
    results = pd.DataFrame.from_dict( evaluator1.evaluate_data(points) )
    results.to_csv(f'{output_path}/xopt_out_{sim_call_count}.csv', index=False)



