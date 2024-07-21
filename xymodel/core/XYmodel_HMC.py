# HMC simulation of two dimensional XY model (core code)

import os
import numpy as np
from tqdm import tqdm

from xymodel import *
from ..parameters import *
from ..analysis.correlations import *
from ..observables.vortex_func import *
from ..observables.helicity_modulus import *

# ================================================================================================

class XYSystem():
    def __init__(self, seed = None, L = 10, a = 1.0, meas_freq = int(50), write_times = int(10), logger=None):
        # set the logger
        self.logger = logger
        
        # set the seed for random number generation
        if seed is not None:
            self.logger.info(f"Random Seed = {seed}")
            np.random.seed(seed)
        else:
            self.logger.info("Random Seed Not Used")
            
        # lattice parameters
        self.L = L # number of lattice sites in a dimension
        self.N = L**2 # number of spins
        self.a = a # lattice spacing
        
        self.accepted_count = 0 # to count the number of accepted steps during sampling
        self.attempts_count = 0 # to count the number of attempts
        self.acc_rate = 0 # to save the acceptance rate during sampling

        # nearest neighbors, returning a tuple of 4 neighbors with the sequence of (right, down, left, top)
        L = self.L
        N = self.N

        self.nbr = {i : ((i // L) * L + (i + 1) % L, 
                         (i + L) % N,
                         (i // L) * L + (i - 1) % L, 
                         (i - L) % N) 
                    for i in list(range(N))} 
        
        # initialization
        self.spin_config = np.random.random(self.N) * 2 * np.pi # spin configuration
        self.meas_freq = meas_freq # frequency of measurement
        self.write_times = write_times # times (number of batches) of data write-out
        self.delta_H = 0 # to save delta Hamiltonian
        self.stop_flag = False # flag of terminating simulation
        
        # create a dictionary to save the data during sampling
        self.data = {
            'L': self.L,
            'N': self.N,
            'a': self.a,
            'T': 0.892,
            'meas_freq': self.meas_freq,
            'write_times': self.write_times,
            'num_traj': [],
            'energy': [],
            'magnetization': [],
            'mx': [],
            'my': [],
            'acc_rate': [],
            'delta_H': [],
            'sep_t': [],
            'sep_n': [],
            'vor_num': [],
            'avor_num': [],
            'vor_den': [],
            'heli_mod': [],
            'avg_gm': None, 
            'var_gm': None,
            'avg_gm_buffer': [],
            'var_gm_buffer': []
        }
        
    def compute_energy(self, spin_config):
        energy = np.zeros(np.shape(spin_config))
        for i, theta in enumerate(spin_config): # compute energy per spin
            energy[i] = - sum(np.cos(theta - spin_config[j]) for j in self.nbr[i])
        
        # total energy of the system
        energy_sys = sum(energy) / 2 # divided by 2 because each neighbor is computed twice
        return energy_sys

    def compute_action(self, spin_config):
        # compute action per spin by summing its interactions with 4 nearest neighbors
        action = np.zeros(np.shape(spin_config))
        for i, theta in enumerate(spin_config): 
            action[i] = - sum(np.cos(theta - spin_config[j]) / self.data['T'] for j in self.nbr[i])
        action = sum(action) / 2 # divided by 2 because each neighbor is computed twice
        return action

    def compute_Hamiltonian(self, spin_config, momentum):
        # kinetic energy of auxiliary momenta
        ke = sum([p ** 2 for p in momentum]) / 2
        
        # potential energy in Hamiltonian dynamics
        pe = self.compute_action(spin_config)
        
        # Hamiltonian
        H = ke + pe
        return H

    def compute_deriv(self, spin_config):
        deriv = np.zeros(np.shape(spin_config))
        for i, theta in enumerate(spin_config):
            deriv[i] = sum(np.sin(theta - spin_config[j]) / self.data['T'] for j in self.nbr[i])
        return deriv

    def Metropolis_choice(self, delta_H):
        # Metropolis-Hastings acceptance
        if delta_H < 0:
            return True
        else:
            rand_num = np.random.rand()
            return rand_num < np.exp(- delta_H)
    
    def leapfrog(self, theta_old, lfl = int(10)):
        # Leapfrog method
        
        '''
        Parameters:
        
        lfl: number of Leapfrog steps
        lfeps: Leapfrog stepsize
        
        (lfl * lfeps = 1 is kept)
        '''
        
        lfeps = 1 / lfl
        
        # update the attempts counting
        self.attempts_count += 1
        
        # sampling of momenta from Gaussian distribution
        p_old = np.random.normal(0, 1, np.shape(theta_old))
        
        # update momenta by a half step
        deriv = self.compute_deriv(theta_old)
        delta_p_half = - 0.5 * lfeps * deriv        
        p_new = p_old + delta_p_half
        
        # update angles by a full step
        theta_new = theta_old + lfeps * p_new
        
        # alternate updating momenta and angles by a full step for lfl-1 times
        for k in range(lfl - 1):
            deriv = self.compute_deriv(theta_new) # recompute derivative
            delta_p = - lfeps * deriv
            p_new = p_new + delta_p
            delta_theta = lfeps * p_new
            theta_new = theta_new + delta_theta
        
        # update momenta by a half step once again
        deriv = self.compute_deriv(theta_new) # recompute derivative
        delta_p_half = - 0.5 * lfeps * deriv
        p_new = p_new + delta_p_half

        #compute the difference of Hamiltonian
        H_old = self.compute_Hamiltonian(theta_old, p_old)
        H_new = self.compute_Hamiltonian(theta_new, p_new)
        
        delta_H = H_new - H_old
        
        # save the value of delta_H
        self.delta_H = H_new - H_old
        
        # Metropolis algorithm
        if self.Metropolis_choice(delta_H):
            self.accepted_count += 1
            return theta_new
        else:
            return theta_old

    def leapfrog_calibration(self, num_step_calib = 5e2, acc_rate_upper = 0.8, acc_rate_lower = 0.6, acc_rate_ref = 0.7, lfl_adj = int(1), num_calib = int(10), lfl_lower = int(1), lfl_upper = int(50)):
        # Calibrate Leapfrog parameters (lfl, lfeps) to control the acceptance rate within a suitable range
        
        '''
        # Parameters:
        
        # num_step_calib: number of trajectories of each calibration iteration
        # acc_rate_upper: upper bound of acceptance rate in calibration
        # acc_rate_lower: lower bound of acceptance rate in calibration
        # acc_rate_ref: reference value of acceptance rate
        # lfl_adj: stepsize of adjusting lfl in every calbration iteration
        # num_calib
        '''
        
        for i in range(num_calib):
            # save current values of Leapfrog parameters
            lfl_cur = sim_paras['lfl']
            lfeps_cur = 1 / lfl_cur
            
            # run the simulation for num_step_calib trajectories to obtain the acceptance rate under current parameters
            # initialization
            spin_config_calib = self.spin_config
            
            self.accepted_count = 0
            self.attempts_count = 0
            
            # evolute the system
            for j in range(num_step_calib):
                spin_config_calib = self.leapfrog(spin_config_calib, lfl = lfl_cur)
            
            # compute current acceptance rate
            self.acc_rate = self.accepted_count / self.attempts_count
            
            self.logger.info(f"calibrating iteration {i + 1} (max:{num_calib}): lfl = {sim_paras['lfl']}, lfeps = {1 / sim_paras['lfl']:.4f}, reference acc_rate = {self.acc_rate:.4f}")

            # implement the calibration procedure if current acceptance rate is out of given range
            if self.acc_rate > acc_rate_upper or self.acc_rate < acc_rate_lower:                
                # adaptively update Leapfrog parameter lfl
                # compute the deviation from the reference acceptance rate
                d_acc_rate = acc_rate_ref - self.acc_rate
                
                # update lfl depending on direction of deviation
                sim_paras['lfl'] += int(np.sign(d_acc_rate) * lfl_adj)
                
                # if lfl is beyond the upper and lower limits, the calibration failed, and return 'true' to stop_flag to terminate the simulation
                if sim_paras['lfl'] < lfl_lower or sim_paras['lfl'] > lfl_upper:
                    self.logger.info('\nCalibration Failed: attempt to increase lfl value')
                    return True
            else:
                self.logger.info(f'\nCalibration Completed: lfl = {lfl_cur}, lfeps = {lfeps_cur:.4f}, reference acc_rate = {self.acc_rate:.4f}')
                return False
        
        # if all iterations are completed but acceptence rate is still out of range, return 'true' to stop_flag to terminate the simulation
        self.logger.info('\nCalibration Failed: all iterations are completed but acceptence rate is still out of range')
        return True
        
    def measure(self, spin_config):
        # compute and save raw data of energy and magnetization (per spin)
        energy = self.compute_energy(spin_config) / self.N
        mx = np.sum(np.cos(spin_config)) / self.N
        my = np.sum(np.sin(spin_config)) / self.N
        m = np.sqrt(mx**2 + my**2) 

        self.data['energy'].append(energy)
        self.data['mx'].append(mx)
        self.data['my'].append(my)
        self.data['magnetization'].append(m)    
        
    def run_simulation(self, T = None, num_traj = int(1e4), lfl=10, lf_calib = True, write_times = 10, log_freq = int(100), max_sep_t = 5, vor_thld = 0.01, folder_temp = None):
        # Run the simulation under temperature T to generate sample points/HMC trajectories, and measure physical quantities under the equilibrium.
        
        '''
        Parameters:
        
        T: temperature (if none, the default value in initialialization will be used)
        num_traj: number of steps/trajectories
        lf_calib: if True, activate the calibration of leapfrog parameters
        vor_thld: threshold of angle differences on each plaquette to identify vortices and anti-vortices
        
        Outputs: saved in self.data
        
        '''
        
        # Initialization
        spin_config = self.spin_config
        self.data['T'] = T
        raw_data_path = os.path.join(folder_temp, f"raw_data_T{T:.2f}.txt")
        
        self.data['avg_gm'] = np.zeros(max_sep_t)
        self.data['var_gm'] = np.zeros(max_sep_t)
        
        self.logger.info(f"Lattice Size L^2 = {self.L}^2 \nSimulation Temperature T = {T:.2f}")
        
        # ------------------------------------------------------------------------------------------------

        # Calibrate Leapfrog parameters (lfl, lfeps)
        
        if lf_calib:
            self.logger.info('-' * 100)
            self.logger.info('Calibration of Leapforg Parameters Activated')
            self.logger.info('Calibrating Leapfrog parameters..')
            self.stop_flag = self.leapfrog_calibration(**calibration_paras)
        else:
            self.logger.info('-' * 100)
            self.logger.info('Calibration of Leapforg Parameters Not Activated')
            self.stop_flag = False
        
        # terminate the simulation if the caliibration fails
        if self.stop_flag:
            self.logger.info(f"Simulation at T ={T: .2f} Terminated")
            self.logger.info('=' * 100)
            return spin_config

        # ------------------------------------------------------------------------------------------------
        
        # Burn-in stage
        
        # reset counters
        self.accepted_count = 0
        self.attempts_count = 0
        
        self.logger.info('-' * 100)
        
        # evolute the system to the equilibrium
        equi_steps = int(0.1 * num_traj) # number of trajectories for the burn-in step
        
        with tqdm(total=equi_steps, desc="Equilibrating", ncols=100) as pbar:
            for n in range(equi_steps):
                spin_config = self.leapfrog(spin_config, sim_paras['lfl'])
                if (n + 1) % log_freq == 0:
                    pbar.update(log_freq)
                    self.logger.realtime(f'Equilibrating: trajectories {n + 1} / {equi_steps}')           
                     
        self.logger.info(f"\nBurn-in Stage Completed (T = {T:.2f})")
        self.logger.info('-' * 100)
        
        # ------------------------------------------------------------------------------------------------

        # Samlping stage
        
        # reset counters
        self.accepted_count = 0
        self.attempts_count = 0
        write_count = 0
        
        # sampling under equilibrium
        with tqdm(total=num_traj, desc="Sampling", ncols=100, leave=False) as pbar:
            
            for t in range(num_traj): 
                # evolute the system
                spin_config = self.leapfrog(spin_config, lfl = sim_paras['lfl'])
                
                if (t + 1) % log_freq == 0:
                    pbar.update(log_freq)
                    self.logger.realtime(f'Sampling: {t + 1} / {num_traj}')
                
                # save current number of trajectories
                self.data['num_traj'].append(t+1)
                
                # save current values of delta_H
                self.data['delta_H'].append(self.delta_H)

                # measure and save raw data (energy and magnetization)
                self.measure(spin_config)

                # compute and save acceptance rate
                self.acc_rate = self.accepted_count / self.attempts_count
                self.data['acc_rate'].append(self.acc_rate)
                
                # compute and save current average gamma values and variances of both directions
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
                
                # compute and save the numbers of vorteices and anti-vorteices, and the vortex density
                vor_num, avor_num, vor_den = compute_vor(spin_config, vor_thld)
                
                self.data['vor_num'].append(int(vor_num))
                self.data['avor_num'].append(int(avor_num))
                self.data['vor_den'].append(vor_den)
                
                # compute and save the helicity modulus
                heli_mod = compute_heli_mod(spin_config, T)
                self.data['heli_mod'].append(heli_mod)

                # raw data write-out
                if (t + 1) % (int(num_traj / write_times)) == 0:
                    # save all the raw data
                    with open(raw_data_path, 'a') as f:
                        # create header if file is newly created
                        if os.stat(raw_data_path).st_size == 0:
                            f.write("# num_traj energy magnetization mx my acc_rate delta_H vor_num avor_num vor_den heli_mod avg_gm var_gm\n")
                        
                        # write data
                        for i in range(len(self.data['num_traj'])):
                            avg_gm_str = np.array2string(self.data['avg_gm_buffer'][i], precision = 16, separator = ',', suppress_small = True, max_line_width = np.inf)
                            var_gm_str = np.array2string(self.data['var_gm_buffer'][i], precision = 16, separator = ',', suppress_small = True, max_line_width = np.inf)
                            
                            line = f"{self.data['num_traj'][i]} {self.data['energy'][i]:.16f} {self.data['magnetization'][i]:.16f} {self.data['mx'][i]:.16f} {self.data['my'][i]:.16f} {self.data['acc_rate'][i+int(write_count * num_traj / write_times)]:.16f} {self.data['delta_H'][i]:.16f} {self.data['vor_num'][i]} {self.data['avor_num'][i]} {self.data['vor_den'][i]:.16f} {self.data['heli_mod'][i]:.16f} {avg_gm_str} {var_gm_str}\n"
                            f.write(line)
                        
                    # reset after write-out
                    self.data['num_traj'].clear()
                    self.data['energy'].clear()
                    self.data['magnetization'].clear()
                    self.data['mx'].clear()
                    self.data['my'].clear()
                    self.data['acc_rate'].clear()
                    self.data['delta_H'].clear()
                    self.data['vor_num'].clear()
                    self.data['avor_num'].clear()
                    self.data['vor_den'].clear()
                    self.data['heli_mod'].clear()
                    self.data['avg_gm_buffer'].clear()
                    self.data['var_gm_buffer'].clear()
                    
            self.logger.info(f"\n\nSampling Stage Completed (T = {T:.2f})")
            self.logger.info('=' * 100)
        
        return spin_config
