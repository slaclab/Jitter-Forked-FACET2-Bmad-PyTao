import os
from UTILITY_quickstart import *

importedDefaultSettings = loadConfig(f'setLattice_configs/2025-02-25_oneBunch_baseline.yml')

num_calls          = 10 # 9
multiplicity_count = 36 # 8
tasks_per_node     = 140 # 139
num_tasks = int(num_calls * multiplicity_count * tasks_per_node)


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

# impact
nominal_charge             = 1600 # pC
nominal_gun_theta0_deg     = 29.3-90
nominal_gun_rf_field_scale = 119/2.44885*1e6
nominal_L0A_Phase          = 29
nominal_L0A_Amp            = 30e6

# bmad
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
L0A_Phase_jitter       = 0.1 # degS
L0A_Amp_jitter_percent = 0.06
L0B_Phase_jitter       = 0.1 # degS
L0B_Amp_jitter_percent = 0.5
L1A_Phase_jitter       = 0.7 # degS
L1A_Amp_jitter_percent = 0.6
L1B_Phase_jitter       = 0.5 # degS
L1B_Amp_jitter_percent = 0.7
L2_Phase_jitter        = 0.4 # degS
L2_Amp_jitter_percent  = 0.3
L3_Phase_jitter        = 0.4 # degS
L3_Amp_jitter_percent  = 0.3


# note that the linac amplitudes are jittered via *offsets* from the nominal, not set absolutely
cal_data = {
    'L0BPhaseSet':[nominal_L0B_Phase - L0B_Phase_jitter, nominal_L0B_Phase + L0B_Phase_jitter],
    'L1PhaseSet':[nominal_L1_Phase - L1A_Phase_jitter, nominal_L1_Phase + L1A_Phase_jitter], # pick the larger jitter of L1A and L1B
    'L2PhaseSet':[nominal_L2_Phase - L2_Phase_jitter, nominal_L2_Phase + L2_Phase_jitter],
    'L3PhaseSet':[nominal_L3_Phase - L3_Phase_jitter, nominal_L3_Phase + L3_Phase_jitter],
    'L0BEnergyOffset':[-L0B_Amp_jitter_percent * nominal_L0BF_Amp, L0B_Amp_jitter_percent * nominal_L0BF_Amp],
    'L1EnergyOffset':[-L1B_Amp_jitter_percent * nominal_L1_Amp, L1B_Amp_jitter_percent * nominal_L1_Amp], # pick the larger jitter of L1A and L1B
    'L2EnergyOffset':[-L2_Amp_jitter_percent * nominal_L2_Amp, L2_Amp_jitter_percent * nominal_L2_Amp],
    'L3EnergyOffset':[-L3_Amp_jitter_percent * nominal_L3_Amp, L3_Amp_jitter_percent * nominal_L3_Amp],
}


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

points = pd.DataFrame(points)

if __name__ == '__main__':
    
    output_path = '/pscratch/sd/m/maxvarv/Linac_phase_amp_jitter_2025_03_17'
    os.makedirs(output_path, exist_ok=True)
    
    points.to_csv(f'{output_path}/evaluation_points.csv', index=False)



