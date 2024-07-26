#===============================================================================
# Parameters
#===============================================================================

# System Parameters
sys_paras = {
    'seed': None,           # Seed for random number generator
    'L': int(64),           # Lattice size (number of sites along one dimension)
    'a': 1.0,               # Lattice spacing (distance between adjacent sites)
    'write_times': int(10)  # Number of times data is written to disk during simulation
}

# Simulation Parameters
sim_paras = {
    'T': 0.892,             # Simulation temperature (in reduced units)
    'FA': False,            # Whether to perform Fourier acceleration
    'm_FA': 0.1,               # Mass in the kernel in Fourier acceleration method
    'lfl': int(10),         # Number of leapfrog steps for each trajectory in HMC
    'num_traj': int(1e4),   # Total number of trajectories during sampling phase
    'lf_calib': True,       # Whether to perform automatic calibration of leapfrog parameters
    'log_freq': int(100),   # Frequency of logging simulation progress (in trajectories)
    'max_sep_t': int(25),   # Maximum space separation for correlation function calculation
    'folder_temp': r'path/to/folder_temp'  # Path to temporary working directory
}

# Leapfrog Calibration Parameters
calibration_paras = {
    'num_step_calib': int(500),  # Number of trajectories in each calibration iteration
    'acc_rate_upper': 0.75,      # Upper bound of acceptable HMC acceptance rate
    'acc_rate_lower': 0.55,      # Lower bound of acceptable HMC acceptance rate
    'acc_rate_ref': 0.65,        # Reference HMC acceptance rate
    'lfl_adj': int(1),           # Step size for adjusting number of leapfrog steps during calibration
    'num_calib': int(20),        # Maximum number of calibration iterations
    'lfl_lower': int(1),         # Minimum allowed number of leapfrog steps
    'lfl_upper': int(50)         # Maximum allowed number of leapfrog steps
}
