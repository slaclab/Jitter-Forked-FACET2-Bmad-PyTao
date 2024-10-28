
import UTILITY_QPAD_PICMI as picmi
import numpy as np
from pmd_beamphysics import ParticleGroup
import h5py
from scipy.optimize import fsolve
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.constants import physical_constants
import subprocess, os 

cst = picmi.constants


class QPAD_sim:

	# ParticleGroup
	P = None

	# list of species (beams, plasma, and neutrals)
	species_list = []

	# list of particles layouts (ppc data)
	layouts = []

	# Particle Diagnostics 
	if_beam = []
	part_diags = []

	# Field Diagnostics
	field_diags = []

	# simulation, solver and grid structs
	simulation, solver, grid = None, None, None

	"""
	Constructor

	Parameters
	----------
	n0: float
		Normalizing Density in units of m^{-3}

	"""
	def __init__(self, n0 = 1e17 * 1e6):
		self.n0 = n0 
		self.wp = np.sqrt(cst.q_e**2 * self.n0/(cst.ep0 * cst.m_e))
		self.kp = self.wp/cst.c

	"""
	Initialize Grid Paramters

	Parameters
	----------
	nr, nz: integer
		Number of grid cells along r and z, respectfully. 

	zmin, zmax: float
		Grid bounds along z.

	rmin, rmax: float
		Grid bounds along r. Note: rmin should always be zero (axis).

	n_modes: integer
		Number of azimuthal fourier modes m. The code
		uses (2m +1) grids. (1 zero mode + real and imaginary
		components for higher modes).
		
	"""
	def init_grid(self,nr = None, nz = None, zmin = None, zmax = None, rmin = 0, rmax = None, n_modes = 1):
		self.grid = picmi.CylindricalGrid(
			number_of_cells           = [nr, nz],
			lower_bound               = [0. , zmin],
			upper_bound               = [rmax, zmax],
			lower_boundary_conditions = ['open', 'open'],
			upper_boundary_conditions = ['open', 'open'],
			n_azimuthal_modes = n_modes,
			moving_window_velocity    = [0,cst.c])


		self.solver = picmi.ElectromagneticSolver( grid = self.grid )


	"""
	Add OpenPMD Beam File

	Parameters
	----------
	charge: float
	Total charge [C]

	bunch_rms_size: vector of length 3 of floats
		RMS bunch size along (x,y,z) [m]

	bunch_centroid_position: vector of length 3 of floats, default = [0, 0, 0]
		Bunch centroid position (x,y,z) [m]
	
	bunch_centroid_velocity: vector of length 3 of floats, default = [0, 0, 19569.47]
		RMS velocity in units of p/mc (unitless)

	bunch_rms_velocity: vector of length 3 of floats, default = [0, 0, 0]
		RMS velocity in units of sigma_p/mc (unitless)
		
	"""
	def add_openpmd_file_bunch(self, pmd_file_in, qpad_file_out, op = np.median, directory = '.'):
		assert self.grid is not None, Exception("Must initialize grid before adding OpenPMD bunch")
		
		## simulation paramters ##
		P = ParticleGroup(pmd_file_in)
		P = P[P.status == 1]
		q_grid_norm = (2 * np.pi * self.grid.dr**2 * self.grid.dz) 
		q_raw_norm = (cst.q_e * self.n0 )
		scale_q = 1.0/(q_grid_norm * q_raw_norm)
		scale_p = 1/(0.511e6)


		dataset = P.copy() # modify a copy of P
		dataset.x = dataset.x - op(dataset.x) # adjust beam x
		dataset.y = dataset.y - op(dataset.y) # adjust beam y
		dataset.z = cst.c * (op(dataset.t) - dataset.t) # calculate z and center beam at high current region

		x, y, z = self.kp * dataset.x, self.kp * dataset.y, self.kp * dataset.z
		px, py, pz = dataset['px']*scale_p, dataset['py']*scale_p, dataset['pz']*scale_p

		q = dataset.weight *scale_q

		hf = h5py.File(directory + '/' + qpad_file_out, 'w')
		hf.create_dataset('x1', data=x)
		hf.create_dataset('x2', data=y)
		hf.create_dataset('x3', data=z)
		hf.create_dataset('p1', data=px)
		hf.create_dataset('p2', data=py)
		hf.create_dataset('p3', data=pz)
		hf.create_dataset('q',data=-np.abs(q))
		hf.close()

		self.species_list.append(picmi.Species( particle_type = 'electron', 
			initial_distribution = picmi.OpenPMDFileDistribution(qpad_file_out)))
		self.if_beam.append(True)
		self.layouts.append(picmi.FileLayout(grid = self.grid))



	"""
	Add tri-Gaussian electron bunch

	Parameters
	----------
	charge: float
	Total charge [C]

	bunch_rms_size: vector of length 3 of floats
		RMS bunch size along (x,y,z) [m]

	bunch_centroid_position: vector of length 3 of floats, default = [0, 0, 0]
		Bunch centroid position (x,y,z) [m]
	
	bunch_centroid_velocity: vector of length 3 of floats, default = [0, 0, 19569.47]
		RMS velocity in units of p/mc (unitless)

	bunch_rms_velocity: vector of length 3 of floats, default = [0, 0, 0]
		RMS velocity in units of sigma_p/mc (unitless)

	ppc: list of 3 integers, default = [2, 1, 2]
		ppc along r, phi, and z

	num_theta: 
		integer, default = 8
		ppc along azimuthal direction
	
	
	"""
	def add_gaussian_electron_bunch(self, charge, bunch_rms_size, 
		bunch_centroid_position = [0, 0 ,0], bunch_centroid_velocity = [0, 0, 19569.47],
		 bunch_rms_velocity = [0, 0 ,0], ppc = [2, 1, 2], num_theta = 8):

		n_physical_particles = abs(int(charge/cst.q_e))

		dist = picmi.GaussianBunchDistribution(
			n_physical_particles = n_physical_particles,
			rms_bunch_size       = bunch_rms_size,
			rms_velocity         = [cst.c * x for x in bunch_rms_velocity],
			centroid_position    = bunch_centroid_position,
			centroid_velocity    = [cst.c * x for x in bunch_centroid_velocity] )
		
		self.species_list.append(picmi.Species( particle_type = 'electron', initial_distribution = dist))
		layout_dict = { picmi.codename + '_num_theta' : num_theta }
		self.layouts.append(picmi.GriddedLayout(
					grid = self.grid,
					n_macroparticle_per_cell = ppc, 
					**layout_dict))
		self.if_beam.append(True)


	"""
	Add uniform pre-ionized plasma

	Parameters
	----------
	number_density: float
	Plasma electron number density [m^-3]

	ppc: list of 2 integers, default = [4, 1]
		ppc along r and phi

	num_theta: 
		integer, default = 8
		ppc along azimuthal direction
	"""
	def add_uniform_plasma(self, number_density = 0, ppc = [4, 1], num_theta = 8):
		if(self.grid is None):
			print("Warning: Initialize grid before adding plasma")
			return
		self.species_list.append(picmi.Species(particle_type = 'electron', 
			initial_distribution = picmi.UniformDistribution(density = number_density) ))

		layout_dict = { picmi.codename + '_num_theta' : num_theta }
		self.layouts.append(picmi.GriddedLayout(
					grid = self.grid,
					n_macroparticle_per_cell = ppc, 
					**layout_dict))
		self.if_beam.append(False)


	"""
	Add uniform neutral gas (e.g. Li)

	Parameters
	----------
	number_density: float
		Number density of gas [m^-3].

	particle_type: string
		A string specifying an atom (e.g. Li, Ar...) as defined in
		the openPMD 2 species type extension, openPMD-standard/EXT_SpeciesType.md

	max_level: integer, optional 
		Specifies maximum ionization level.

	ppc: list of 2 integers, default = [4, 4]
		ppc along r and phi. ppc(1) * ppc(2) = total ionized macroelectrons 
		per cell (default is 16)

	num_theta: 
		integer, default = 8
		ppc along azimuthal direction
		
	"""
	def add_uniform_neutral_gas(self, number_density = 0, particle_type = 'Li', max_level = None, ppc = [4, 4], num_theta = 8):
		if(self.grid is None):
			print("Warning: Initialize grid before adding neutral gas")
			return
		assert self.grid is not None, Exception("Must initialize grid before adding Plasma")
		if(max_level is not None):
			neut_dict = { picmi.codename + '_ion_max' : max_level }
		else:
			neut_dict = {}

		self.species_list.append(picmi.Neutral(particle_type = particle_type, 
			initial_distribution = picmi.UniformDistribution(density = number_density), 
			**neut_dict ))

		layout_dict = { picmi.codename + '_num_theta' : num_theta }
		self.layouts.append(picmi.GriddedLayout(
					grid = self.grid,
					n_macroparticle_per_cell = ppc, 
					**layout_dict))
		self.if_beam.append(False)



	"""
	Add longitudinal neutral gas (e.g. Li)

	Parameters
	----------
	z: array of floats
		Longitudinal position of neutral gas profile [m].

	nz: array of floats
		Number density of neutral gas profile [m^-3].

	n0_factor: float
		Normalizing density factor [m^-3].

	particle_type: string
		A string specifying an atom (e.g. Li, Ar...) as defined in
		the openPMD 2 species type extension, openPMD-standard/EXT_SpeciesType.md

	max_level: integer, optional 
		Specifies maximum ionization level.

	ppc: list of 2 integers, default = [4, 4]
		ppc along r and phi. ppc(1) * ppc(2) = total ionized macroelectrons 
		per cell (default is 16)

	num_theta: 
		integer, default = 8
		ppc along azimuthal direction
		
	"""
	def add_longitudinal_neutral_gas_profile(self, z, nz,  particle_type = 'Li', max_level = None, ppc = [4, 4], num_theta = 8):
		if(self.grid is None):
			print("Warning: Initialize grid before adding neutral gas")
			return
		assert self.grid is not None, Exception("Must initialize grid before adding Plasma")
		if(max_level is not None):
			neut_dict = { picmi.codename + '_ion_max' : max_level }
		else:
			neut_dict = {}

		self.species_list.append(picmi.Neutral(particle_type = particle_type, 
			initial_distribution = picmi.PiecewiseDistribution(density = self.n0, piecewise_s = z, piecewise_fs = nz), 
			**neut_dict ))

		layout_dict = { picmi.codename + '_num_theta' : num_theta }
		self.layouts.append(picmi.GriddedLayout(
					grid = self.grid,
					n_macroparticle_per_cell = ppc, 
					**layout_dict))
		self.if_beam.append(False)







	"""
	Adds Raw Particle Diagnostic for beam dumps

	Parameters
	----------

	period: integer, default = 1
		Frequency of data dumps (1 dumps every timestep)

	period: integer, default = 1
		Sampling frequency of particles (1 dumps every particle, 2 dumps every other part)
		
	"""
	def add_particle_diagnostics(self, period = 1, psample = 1):
		part_diag_dict = { picmi.codename + '_sample' : 1}
		beam_list = []
		for i in range(len(self.species_list)):
			if(self.if_beam[i]):
				beam_list.append(self.species_list[i])
		self.part_diags.append(picmi.ParticleDiagnostic(period = period,
                             species = beam_list,
                              **part_diag_dict))

	"""
	Adds Field Diagnostic to data dumps

	Parameters
	----------
	data_list: list of strings
		Field Data to dump (e.g. ['Er', 'Ephi', 'Ez', 'Br', 'Bphi', 'Bz', 'psi'])

	period: integer, default = 1
		Frequency of data dumps (1 dumps every timestep)

	"""
	def add_field_diagnostics(self, data_list = [], period = 1):
		self.field_diags.append(picmi.FieldDiagnostic(data_list = data_list,
	                                   grid = self.grid,
	                                   period = period))


	"""
	Constructs simulation input file and runs QPADs

	Parameters
	----------
	dt: float
		Time step of simulation [s].

	tmax: float
		Maximum time of simulation [s].

	nodes: list of 2 integers, default = [1, 1]
		mpi procs along r and z
		
	"""
	def run_simulation(self,dt, tmax, nodes = [1, 1], sim_dir = '.', path_to_qpad = '.'):
		sim_dict = { picmi.codename + '_nodes' : nodes, picmi.codename + '_n0' : self.n0}
		self.simulation = picmi.Simulation(solver = self.solver, verbose = 1,
			time_step_size = dt, max_time = tmax, **sim_dict)

		for i in range(len(self.species_list)):
			self.simulation.add_species(species = self.species_list[i], layout = self.layouts[i])

		for i in range(len(self.field_diags)):
			self.simulation.add_diagnostic(self.field_diags[i])

		for i in range(len(self.part_diags)):
			self.simulation.add_diagnostic(self.part_diags[i])


		self.simulation.write_input_file(sim_dir+ '/qpinput.json')
		# todo add mpi script
		# subprocess ....
		env = dict(os.environ)
		env['LD_LIBRARY_PATH'] ='/sdf/group/facet/codes/qpad_libs_openmpi/json-fortran/build/lib:'  + env['LD_LIBRARY_PATH']
		procs = np.prod(nodes)
		subprocess.run(["mpirun", "-np", str(procs), path_to_qpad + "/qpad.e"], cwd = sim_dir, env=env)


		



	""" 
	Generates the plasma density of the lithium oven/helium as a function of z position.

	The position and density at each position is returned as output in the following order:

	[z array, Lithium density array, Helium density array]

	Args:
	    Nz: Number of positions in z.
	    Z: [m] Maximum z position to generate, inclusive.
	    P: [torr] Buffer gas pressure.
	    T_bkgd: [K] Temperature of the background He buffer gas.
	    l_He: [m] Length of He density to use from thermodynamics calulation. Interpolation
	        is used outside of this region.
	    filename_Li: Filename to output the Li density to.
	    filename_He: Filename to output the He density to.
	"""
	def generate_Li_oven_profile(self, Nz = 1001, Z = 0.6, P = 5.0, T_bkgd = 273.15, l_He = 0.44 ):

		# Calculation variables
		z = np.linspace(0.0, Z, Nz)
		n = np.zeros(Nz, dtype="double")
		center = Z / 2
		kB = physical_constants["Boltzmann constant"][0]


		# Lithium properties
		def Pv(T):
		    """Calculates the vapor pressure of the lithium gas as a function of temperature.

		    Args:
		        T: [K] temperature of the lithium gas.

		    Returns:
		        Pv: [torr] vapor pressure of the lithium.
		    """
		    T = T * 1e-3
		    return np.exp(-2.0532 * np.log(T) - 19.4268 / T + 9.4993 + 0.753 * T) / 133.0e-6


		def f(T):
		    return Pv(T) - P


		T = fsolve(f, 1000.0)[0]
		ne = 9.66e24 * P / T

		# Background lithium density - necessary for find He density
		Pv_bkgd = Pv(T_bkgd)
		n_bkgd = 9.66e24 * Pv_bkgd / T_bkgd

		# Uniform accelerating plasma
		length = 238e-3
		z_start = center - 0.5 * length
		z_end = center + 0.5 * length
		sel = (z > z_start) * (z < z_end)
		n[sel] = 1.0

		# Entrance ramp - error function
		ent_start = center - 400.0e-3
		s_ent = 22.0e-3
		sel = (z >= ent_start) * (z <= z_start)
		n[sel] = 0.5 * (1 + erf((z[sel] - z_start + 100.0e-3) / (np.sqrt(2) * s_ent)))
		n[sel] *= 1.0 / n[sel][-1]  # Make sure curve is continuous

		# Exit ramp - error function
		exit_end = center + 400.0e-3
		s_ext = 22.0e-3
		sel = (z >= z_end) * (z <= exit_end)
		n[sel] = 0.5 * (1 + erf(-(z[sel] - z_end - 100.0e-3) / (np.sqrt(2) * s_ext)))
		n[sel] *= 1.0 / n[sel][0]  # Make sure curve is continuous

		n *= ne
		n += n_bkgd

		# Save the Li plasma density file
		data = np.stack((z, n), axis=1)

		# Calculate He plasma density
		# First create interpolations to go from density to temperature and Li pressure
		T_int = np.linspace(200, 1200, 1001)
		P_int = Pv(T_int)
		n_int = 9.66e24 * P_int / T_int
		T_from_n = interp1d(n_int, T_int)
		P_from_n = interp1d(n_int, P_int)

		# Find the temperature and Li pressure along the oven, then calculate He density
		T_n = T_from_n(n)
		P_n = P_from_n(n)
		n_He = ((P - P_n) * 133.32236842) / (kB * T_n)
		n_He_bkgd = ((P) * 133.32236842) / (kB * T_bkgd)

		# Above meathod breaks down at low Li pressure, use linear interpolation from the ramps
		z_HeStart = center - 0.5 * l_He
		z_HeEnd = center + 0.5 * l_He
		nHe = np.zeros(Nz)
		sel = (z > z_HeStart) * (z < z_HeEnd)
		nHe[sel] = n_He[sel]

		# Extend linearly from the ends
		slope = (nHe[sel][10] - nHe[sel][0]) / (z[sel][10] - z[sel][0])
		selUp = z <= z_HeStart
		nHe[selUp] = slope * (z[selUp] - z[sel][0]) + nHe[sel][0]
		selDown = z >= z_HeEnd
		nHe[selDown] = -slope * (z[selDown] - z[sel][-1]) + nHe[sel][-1]

		# Set to background density
		sel = nHe > n_He_bkgd
		nHe[sel] = n_He_bkgd

		data = np.stack((z, nHe), axis=1)

		return [z, n, nHe]






		