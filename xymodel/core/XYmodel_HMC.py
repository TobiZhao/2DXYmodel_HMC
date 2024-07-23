#===============================================================================
# HMC Simulation of 2D XY Model (Core Code)
#===============================================================================

import os
import numpy as np
from tqdm import tqdm

from xymodel import *
from ..parameters import *
from ..analysis.correlations import *
from ..observables.vortex import *
from ..observables.helicity_modulus import *

class XYSystem:
    """
    Represents the XY model system and implements the HMC simulation.
    """
    # Define keys for the data dictionary
    EMPTY_LIST_KEYS = ['num_traj', 'energy', 'magnetization', 'mx', 'my', 'acc_rate', 'delta_H', 
                       'sep_t', 'sep_n', 'vor_num', 'avor_num', 'vor_den', 'heli_mod', 
                       'avg_gm_buffer', 'var_gm_buffer']
    
    NONE_KEYS = ['avg_gm', 'var_gm']
    
    def __init__(self, seed=None, L=10, a=1.0, write_times=10, logger=None):
        """
        Initialize the XY system.

        Args:
            seed (int, optional): Seed for random number generation. Defaults to None.
            L (int, optional): Number of lattice sites in one dimension. Defaults to 10.
            a (float, optional): Lattice spacing. Defaults to 1.0.
            write_times (int, optional): Number of times to write data. Defaults to 10.
            logger (Logger, optional): Logger object. Defaults to None.
        """
        self.logger = logger
        
        if seed is not None:
            self.logger.info(f"Random Seed = {seed}")
            np.random.seed(seed)
        else:
            self.logger.info("Random Seed Not Used")
            
        # Lattice parameters
        self.L = L
        self.N = L**2
        self.a = a
        
        self.accepted_count = 0
        self.attempts_count = 0
        self.acc_rate = 0

        # Nearest neighbors, (right, down, left, top)
        self.nbr = {i: ((i // L) * L + (i + 1) % L,  # Right site
                        (i + L) % self.N,            # Down site
                        (i // L) * L + (i - 1) % L,  # Left site
                        (i - L) % self.N)            # Top site
                    for i in range(self.N)}
        
        # Initialization
        self.spin_config = np.random.random(self.N) * 2 * np.pi
        self.write_times = write_times
        self.delta_H = 0
        self.stop_flag = False
        
        # Data dictionary
        self.data = {
            'L': self.L,
            'N': self.N,
            'a': self.a,
            'T': 0.892,
            'write_times': self.write_times,
            **{key: [] for key in self.EMPTY_LIST_KEYS},
            **{key: None for key in self.NONE_KEYS}
        }
    
    def compute_energy(self, spin_config):
        """
        Compute the total energy of the system (float).
        """
        nbr_spin_diff = spin_config[:, None] - spin_config[np.array(list(self.nbr.values()))]
        energy = -np.sum(np.cos(nbr_spin_diff))
        return energy / 2  # Avoid double counting
    
    def compute_action(self, spin_config):
        """
        Compute the action of the system (float).
        """
        nbr_spin_diff = spin_config[:, None] - spin_config[np.array(list(self.nbr.values()))]
        action = -np.sum(np.cos(nbr_spin_diff) / self.data['T'])
        return action / 2

    def compute_Hamiltonian(self, spin_config, momentum):
        """
        Compute the Hamiltonian of the system (float).
        """
        ke = sum([p ** 2 for p in momentum]) / 2
        pe = self.compute_action(spin_config)
        
        H = ke + pe
        return H
    
    def compute_deriv(self, spin_config):
        """
        Compute the derivative of the action (ndarray).
        """
        nbr_spin_diff = spin_config[:, None] - spin_config[np.array(list(self.nbr.values()))]
        deriv = np.sum(np.sin(nbr_spin_diff) / self.data['T'], axis=1)
        return deriv
    
    def Metropolis_choice(self, delta_H):
        """
        Make a Metropolis-Hastings acceptance decision (bool).
        """
        if delta_H < 0:
            return True
        else:
            return np.random.rand() < np.exp(-delta_H)
            
    def leapfrog(self, theta_old, lfl = int(10)):
        '''
        Perform a leapfrog integration step, and return the updated configuration.
        
        Parameters:
        
        lfl: number of Leapfrog steps
        lfeps: Leapfrog stepsize
        
        (lfl * lfeps = 1 is kept)
        '''
        lfeps = 1 / lfl
        
        self.attempts_count += 1
        
        # Sample momenta from Gaussian distribution
        p_old = np.random.normal(0, 1, np.shape(theta_old))
        
        # First half-step for momenta
        deriv = self.compute_deriv(theta_old)
        delta_p_half = -0.5 * lfeps * deriv        
        p_new = p_old + delta_p_half
        
        # Full step for positions
        theta_new = theta_old + lfeps * p_new
        
        # (lfl-1) full steps for momenta and positions
        for _ in range(lfl - 1):
            deriv = self.compute_deriv(theta_new)
            delta_p = -lfeps * deriv
            p_new = p_new + delta_p
            delta_theta = lfeps * p_new
            theta_new = theta_new + delta_theta
        
        # Last half-step for momenta
        deriv = self.compute_deriv(theta_new)
        delta_p_half = -0.5 * lfeps * deriv
        p_new = p_new + delta_p_half

        # Compute the difference in Hamiltonian
        H_old = self.compute_Hamiltonian(theta_old, p_old)
        H_new = self.compute_Hamiltonian(theta_new, p_new)
        delta_H = H_new - H_old
        self.delta_H = delta_H
        
        # Metropolis acceptance step
        if self.Metropolis_choice(delta_H):
            self.accepted_count += 1
            return theta_new
        else:
            return theta_old

    def leapfrog_calibration(self, num_step_calib=500, acc_rate_upper=0.8, acc_rate_lower=0.6, acc_rate_ref=0.7, lfl_adj=1, num_calib=10, lfl_lower=1, lfl_upper=50):
        """
        Calibrate leapfrog parameters to control the acceptance rate.

        Parameters:
            num_step_calib (int): Number of trajectories for each calibration iteration.
            acc_rate_upper (float): Upper bound of acceptance rate.
            acc_rate_lower (float): Lower bound of acceptance rate.
            acc_rate_ref (float): Reference value of acceptance rate.
            lfl_adj (int): Step size for adjusting lfl.
            num_calib (int): Maximum number of calibration iterations.
            lfl_lower (int): Lower bound for lfl.
            lfl_upper (int): Upper bound for lfl.
        """
        for i in range(num_calib):
            # Get current Leapfrog parameters
            lfl_cur = sim_paras['lfl']
            lfeps_cur = 1 / lfl_cur
            
            # Run simulation to get acceptance rate
            spin_config_calib = self.spin_config
            self.accepted_count = self.attempts_count = 0
            
            for _ in range(num_step_calib):
                spin_config_calib = self.leapfrog(spin_config_calib, lfl=lfl_cur)
            
            self.acc_rate = self.accepted_count / self.attempts_count
            
            self.logger.info(f"Calibration iteration {i + 1}/{num_calib}: lfl = {sim_paras['lfl']}, lfeps = {1/sim_paras['lfl']:.4f}, acc_rate = {self.acc_rate:.4f}")

            # Adjust lfl if acceptance rate is out of range
            if self.acc_rate > acc_rate_upper or self.acc_rate < acc_rate_lower:
                d_acc_rate = acc_rate_ref - self.acc_rate
                sim_paras['lfl'] += int(np.sign(d_acc_rate) * lfl_adj)
                
                # Check if lfl is within allowed range
                if sim_paras['lfl'] < lfl_lower or sim_paras['lfl'] > lfl_upper:
                    self.logger.info('\nCalibration Failed: lfl out of allowed range')
                    return True
            else:
                self.logger.info(f'\nCalibration Completed: lfl = {lfl_cur}, lfeps = {lfeps_cur:.4f}, reference acc_rate = {self.acc_rate:.4f}')
                return False
        
        # Calibration failed if all iterations completed without success
        self.logger.info('\nCalibration Failed: acceptance rate still out of range')
        return True
        
    def measure(self, spin_config):
        '''
        Measure and save raw data of energy and magnetization.
        '''
        energy = self.compute_energy(spin_config) / self.N
        mx = np.sum(np.cos(spin_config)) / self.N
        my = np.sum(np.sin(spin_config)) / self.N
        m = np.sqrt(mx**2 + my**2) 

        self.data['energy'].append(energy)
        self.data['mx'].append(mx)
        self.data['my'].append(my)
        self.data['magnetization'].append(m)    
        
    def run_simulation(self, T=None, num_traj=int(1e4), lfl=10, lf_calib=True, write_times=10, log_freq=int(100), max_sep_t=5, vor_thld=0.01, folder_temp=None):
        """
        Run the simulation.

        Parameters:
            T (float, optional): Temperature. If None, use default value. Defaults to None.
            num_traj (int, optional): Number of trajectories. Defaults to 10000.
            lfl (int, optional): Number of leapfrog steps. Defaults to 10.
            lf_calib (bool, optional): Whether to calibrate leapfrog parameters. Defaults to True.
            write_times (int, optional): Number of times to write data. Defaults to 10.
            log_freq (int, optional): Frequency of logging. Defaults to 100.
            max_sep_t (int, optional): Maximum separation for correlation function. Defaults to 5.
            vor_thld (float, optional): Threshold for vortex identification. Defaults to 0.01.
            folder_temp (str, optional): Temporary folder for output. Defaults to None.
        """
        # Initialization
        spin_config = self.spin_config
        self.data['T'] = T
        raw_data_path = os.path.join(folder_temp, f"raw_data_T{T:.2f}.txt")
        
        self.data['avg_gm'] = np.zeros(max_sep_t)
        self.data['var_gm'] = np.zeros(max_sep_t)
        
        self.logger.info(f"Lattice Size L^2 = {self.L}^2 \nSimulation Temperature T = {T:.2f}")
        
        # Calibrate Leapfrog parameters if enabled
        if lf_calib:
            self.logger.info('-' * 100)
            self.logger.info('Calibration of Leapfrog Parameters Activated')
            self.stop_flag = self.leapfrog_calibration(**calibration_paras)
        else:
            self.logger.info('-' * 100)
            self.logger.info('Calibration of Leapfrog Parameters Not Activated')
            self.stop_flag = False
        
        if self.stop_flag:  # Terminate the simulation if calibration fails
            self.logger.info(f"Simulation at T = {T:.2f} Terminated")
            self.logger.info('=' * 100)
            return spin_config

        # Burn-in stage
        self.accepted_count = self.attempts_count = 0  # Reset counters
        self.logger.info('-' * 100)
        
        equi_steps = int(0.1 * num_traj)  # Number of trajectories for burn-in
        
        with tqdm(total=equi_steps, desc="Equilibrating", ncols=100) as pbar:
            for n in range(equi_steps):
                spin_config = self.leapfrog(spin_config, sim_paras['lfl'])
                if (n + 1) % log_freq == 0:
                    pbar.update(log_freq)
                    self.logger.realtime(f'Equilibrating: trajectories {n + 1} / {equi_steps}')           
                    
        self.logger.info(f"\nBurn-in Stage Completed (T = {T:.2f})")
        self.logger.info('-' * 100)
        
        # Sampling stage
        self.accepted_count = self.attempts_count = 0  # Reset counters
        write_count = 0
        
        with tqdm(total=num_traj, desc="Sampling", ncols=100, leave=False) as pbar:
            for t in range(num_traj): 
                spin_config = self.leapfrog(spin_config, lfl=sim_paras['lfl'])  # Evolve the system
                
                if (t + 1) % log_freq == 0:
                    pbar.update(log_freq)
                    self.logger.realtime(f'Sampling: {t + 1} / {num_traj}')
                
                self.data['num_traj'].append(t+1)  # Save current number of trajectories
                self.data['delta_H'].append(self.delta_H)  # Save current values of delta_H

                self.measure(spin_config)  # Measure and save raw data (energy and magnetization)

                self.acc_rate = self.accepted_count / self.attempts_count  # Compute and save acceptance rate
                self.data['acc_rate'].append(self.acc_rate)
                
                # Compute and save current average gamma values and variances
                gm = proc_corre(spin_config, max_sep_t)
                gm = np.array(gm)
                d_gm = gm - self.data['avg_gm']

                if t == 0:
                    self.data['avg_gm'] = gm
                    self.data['var_gm'] = np.zeros_like(gm)
                else:
                    self.data['avg_gm'] = (t * self.data['avg_gm'] + gm) / (t + 1)
                    self.data['var_gm'] = ((t - 1) / t) * self.data['var_gm'] + (d_gm ** 2) / t
                
                self.data['avg_gm_buffer'].append(self.data['avg_gm'])
                self.data['var_gm_buffer'].append(self.data['var_gm'])
                
                # Compute and save vortex data
                vor_num, avor_num, vor_den = compute_vor(spin_config, vor_thld)
                self.data['vor_num'].append(int(vor_num))
                self.data['avor_num'].append(int(avor_num))
                self.data['vor_den'].append(vor_den)
                
                # Compute and save the helicity modulus
                heli_mod = compute_heli_mod(spin_config, T)
                self.data['heli_mod'].append(heli_mod)

                # Raw data write-out
                if (t + 1) % (int(num_traj / write_times)) == 0:
                    with open(raw_data_path, 'a') as f:
                        if os.stat(raw_data_path).st_size == 0:  # Create header if file is newly created
                            f.write("# num_traj energy magnetization mx my acc_rate delta_H vor_num avor_num vor_den heli_mod avg_gm var_gm\n")
                        
                        for i in range(len(self.data['num_traj'])):  # Write data
                            avg_gm_str = np.array2string(self.data['avg_gm_buffer'][i], precision=16, separator=',', suppress_small=True, max_line_width=np.inf)
                            var_gm_str = np.array2string(self.data['var_gm_buffer'][i], precision=16, separator=',', suppress_small=True, max_line_width=np.inf)
                            
                            line = f"{self.data['num_traj'][i]} {self.data['energy'][i]:.16f} {self.data['magnetization'][i]:.16f} {self.data['mx'][i]:.16f} {self.data['my'][i]:.16f} {self.data['acc_rate'][i+int(write_count * num_traj / write_times)]:.16f} {self.data['delta_H'][i]:.16f} {self.data['vor_num'][i]} {self.data['avor_num'][i]} {self.data['vor_den'][i]:.16f} {self.data['heli_mod'][i]:.16f} {avg_gm_str} {var_gm_str}\n"
                            f.write(line)
                    
                    # Reset data buffers after write-out
                    for key in ['num_traj', 'energy', 'magnetization', 'mx', 'my', 'acc_rate', 'delta_H', 'vor_num', 'avor_num', 'vor_den', 'heli_mod', 'avg_gm_buffer', 'var_gm_buffer']:
                        self.data[key].clear()
                    
            self.logger.info(f"\n\nSampling Stage Completed (T = {T:.2f})")
            self.logger.info('=' * 100)
        
        return spin_config
