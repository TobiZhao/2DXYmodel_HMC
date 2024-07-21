# Parameters

sys_paras = {
    'seed': None,                   # seed for random number generator
    'L': int(64),                   # lattice size
    'a': 1.0,                       # lattice spacing
    'meas_freq': int(50),           # frequency of measuring raw data
    'write_times': int(10)          # times of data write-out
}

sim_paras = {
    'T': 0.892,                     # simulation temperature 
    'lfl': int(10),                 # number of leapfrog steps for each trajectory
    'num_traj': int(1e4),           # number of trajectories during sampling
    'lf_calib': True,               # whether to calibrate leapfrog parameters
    'log_freq': int(100),           # frequency of logging
    'max_sep_t': int(25),           # maximum of separation t when computing correlation function
    'folder_temp': r'path/to/folder_temp'  # path to working folder
}

calibration_paras = {
    'num_step_calib': int(500),     # number of trajectories of each calibration iteration
    'acc_rate_upper': 0.75,         # upper bound of acceptance rate in calibration
    'acc_rate_lower': 0.55,         # lower bound of acceptance rate in calibration
    'acc_rate_ref': 0.65,           # reference value of acceptance rate
    'lfl_adj': int(1),              # stepsize of adjusting lfl in every calbration iteration
    'num_calib': int(20),           # maximal number of iterations of calibration
    'lfl_lower': int(1),            # lower bound of lfl
    'lfl_upper': int(50)            # upper bound of lfl
}
