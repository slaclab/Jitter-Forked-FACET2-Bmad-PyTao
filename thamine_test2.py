from UTILITY_QPAD import QPAD_sim
import numpy as np

# directory to store QPAD simulation output
sim_dir = './test1234'


# normalizing density
n0 = 6.73e16 * 1e6 # m^[-3]
sim = QPAD_sim(n0)
kp = sim.kp
wp = sim.wp



sim.init_grid(nr = 384, nz = 768, rmin = 0, rmax = 9/kp, zmin = -8/kp, zmax = 2/kp)

# Example: add bunch from openPMD file (qpad_beam_file.h5 created in directory as input to QPAD)
sim.add_openpmd_file_bunch('beamAt_PENT.h5', 'qpad_beam_file.h5', directory = sim_dir)


# Example: add 2nC gaussian bunch
# sim.add_gaussian_electron_bunch(2e-9, [0.5/kp, 0.5/kp, 0.5/kp])


# Example: add uniform pre-ionized plasma with density n0
# sim.add_uniform_plasma(number_density =n0) 

# Example: add neutral gas with density n0
# sim.add_uniform_neutral_gas(number_density = n0) 

z, nLi, nHe = sim.generate_Li_oven_profile()

# # add Lithium gas profile
sim.add_longitudinal_neutral_gas_profile(z, nLi, particle_type = 'Li', max_level = 1) # only consider first level of Lithium

# # add Helium gas profile
sim.add_longitudinal_neutral_gas_profile(z, nHe, particle_type = 'He', max_level = 1) # only consider first level of Helium




## field diagnostics (dump every 10 timesteps)
sim.add_field_diagnostics(data_list = ['Ez', 'rho', 'psi'], period = 10) # 

## add particle diagnotics (dump every timestep)
sim.add_particle_diagnostics(period = 1)

# generate QPAD input deck and data dumps (in sim_dir) and run code (specify path_to_qpad directory containing qpad.e)
sim.run_simulation(dt = 20/wp, tmax = 200.1/wp, sim_dir = sim_dir, path_to_qpad = '/sdf/group/facet/codes/QPAD/bin')