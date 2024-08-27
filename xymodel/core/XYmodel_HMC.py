#===============================================================================
# HMC Simulation of 2D XY Model (Core Code)
#===============================================================================

import os
import numpy as np
from tqdm import tqdm

from numba import njit, prange

from xymodel.parameters import *
from xymodel.analysis.correlations import *
from xymodel.observables.vortex import *
from xymodel.observables.helicity_modulus import *

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

        # Nearest neighbors (right, down, left, top)
        self.nbr = np.zeros((L, L, 4), dtype=np.int64)

        for i in range(L):
            for j in range(L):
                self.nbr[i, j, 0] = i * L + (j + 1) % L     # Right site
                self.nbr[i, j, 1] = ((i + 1) % L) * L + j   # Down site
                self.nbr[i, j, 2] = i * L + (j - 1) % L     # Left site
                self.nbr[i, j, 3] = ((i - 1) % L) * L + j   # Top site
        
        # Initialization
        self.spin_config = np.random.random((L, L)) * 2 * np.pi
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
   
    def Metropolis_choice(self, delta_H):
        """
        Make a Metropolis-Hastings acceptance decision (bool).
        """
        if delta_H < 0:
            return True
        else:
            return np.random.rand() < np.exp(-delta_H)
            
    def leapfrog(self, theta_ini, lfl = int(10)):
        '''
        Perform a leapfrog integration step, and return the updated configuration.
        
        Inputs:
            theta_ini: Initial configuration
            lfl: Number of Leapfrog steps
            lfeps: Leapfrog stepsize
            
            (lfl * lfeps = 1 is kept)
        
        Outputs:
            theta_fin: Updated configuration
        '''
        # Initialization
        T = self.data['T']
        L = self.data['L']
        lfeps = 1 / lfl
        self.attempts_count += 1
        
        # Sample momentum from Gaussian distribution
        p_ini = np.random.normal(0, 1, np.shape(theta_ini))
        
        # First half-step for momentum
        deriv = compute_deriv(theta_ini, self.nbr, T)
        
        d_p_half = -0.5 * lfeps * deriv
        p = p_ini.copy() + d_p_half
        
        # Full step for positions
        theta = theta_ini + lfeps * p
        
        # (lfl-1) full steps for momentum and positions
        for _ in range(lfl - 1):
            deriv = compute_deriv(theta, self.nbr, T)
            
            d_p = -lfeps * deriv
            p += d_p
            
            d_theta = lfeps * p
            theta += d_theta
            
        theta_fin = theta.copy()
        
        # Last half-step for momentum
        deriv = compute_deriv(theta_fin, self.nbr, T)
        
        d_p_half = -0.5 * lfeps * deriv
        p_fin = p.copy() + d_p_half

        # Compute the difference in Hamiltonian
        H_ini = compute_Hamiltonian(theta_ini, p_ini, self.nbr, T, L)
        H_fin = compute_Hamiltonian(theta_fin, p_fin, self.nbr, T, L)
        delta_H = H_fin - H_ini
        self.delta_H = delta_H
        #print("delta_H (per site)", delta_H / L ** 2)
        # Metropolis acceptance step
        if self.Metropolis_choice(delta_H):
            self.accepted_count += 1
            return theta_fin
        else:
            return theta_ini
        
    def leapfrog_FA(self, theta_ini, lfl=int(10), m_FA=0.1):
        """
        Perform a leapfrog integration step with Fourier Acceleration (FA) and return the updated configuration.
        
        Inputs:
            theta_ini: Initial configuration
            lfl: Number of Leapfrog steps
            lfeps: Leapfrog stepsize
            m_FA: Mass parameter of FA
            
            (lfl * lfeps = 1 is kept)
        
        Outputs:
            theta_fin: Updated configuration
        """
        #Initializaiton
        L = self.L
        T = self.data['T']
        lfeps = 1 / lfl
        self.attempts_count += 1
        #print(100 * "-")
        # Construct the inverse Fourier transformed kernel
        K_ft_inv = inv_ft_kernel(L, m_FA)
        #print("K_ft_inv", K_ft_inv)
        # Sample the real-valued object from Gaussian distribution
        sigma = np.sqrt(L**2 / K_ft_inv)
        #print("sigma", sigma)
        Pi_k = sigma * np.random.normal(0, 1, K_ft_inv.shape)
        #print("Pi_k", Pi_k)
        # Construct the auxiliary momentum in Fourier space from the real-valued object
        p_k_ini = gen_momentum(Pi_k)
        p_ini_raw = np.fft.ifft2(p_k_ini) # should be real by Hermitian symmetry
        p_ini = np.real(p_ini_raw)
        #print("p_k_ini", p_k_ini)
        #print("p_ini_raw (should be real cuz of Hermitian sym)", p_ini_raw)
        #print("p_ini", p_ini)
        #print("theta_ini", theta_ini)
        
        # First half-step for momentum
        deriv = compute_deriv(theta_ini, self.nbr, T)
        
        d_p_half = -0.5 * lfeps * deriv
        p = p_ini.copy() + d_p_half
        #print("p (after first half-step update)", p)
        # Full step for positions
        p_k = np.fft.fft2(p)
        #print("p_k (in full step for positions)", p_k)
        d_theta = lfeps * np.real(np.fft.ifft2(np.multiply(K_ft_inv, p_k)))
        theta = theta_ini + d_theta

        # (lfl-1) full steps for momentum and positions
        #d_p_cumu = 0
        for _ in range(lfl - 1):
            deriv = compute_deriv(theta, self.nbr, T)
            #print(f"deriv (iteration {_ + 1})", deriv)
            d_p = -lfeps * deriv
            p += d_p
            #d_p_cumu += d_p
            #print(f"d_p (iteration {_ + 1})", d_p)
            #print(f"p (iteration {_ + 1})", p)
            d_theta = lfeps * np.real(np.fft.ifft2(np.multiply(K_ft_inv, np.fft.fft2(p))))
            theta += d_theta
            #print(f"theta (iteration {_ + 1})", theta)

        #print("d_p_cumu", d_p_cumu)
        theta_fin = theta.copy()
        #print("theta_fin", theta_fin)
        # Last half-step for momentum
        deriv = compute_deriv(theta_fin, self.nbr, T)
        d_p_half = -0.5 * lfeps * deriv
        p_fin = p.copy() + d_p_half
        #print("p_fin", p_fin)
        #print("d_p_tot", p_fin - p_ini)
        # Compute the difference of Hamiltonian
        p_k_fin = np.fft.fft2(p_fin)
        #print("p_k_fin", p_k_fin)
        #print("d_p_k_tot", p_k_fin - p_k_ini)
        H_ini, ke_matrix_ini = compute_Hamiltonian_FA(theta_ini, p_k_ini, K_ft_inv, self.nbr, T, L)
        H_fin, ke_matrix_fin = compute_Hamiltonian_FA(theta_fin, p_k_fin, K_ft_inv, self.nbr, T, L)
        delta_H = H_fin - H_ini
        #delta_ke_matrix = ke_matrix_fin - ke_matrix_ini
        self.delta_H = delta_H
        #print("delta_H (per site)", delta_H / L ** 2)
        #print("delta_ke_matrix", delta_ke_matrix)
        # Metropolis acceptance step
        if self.Metropolis_choice(delta_H):
            self.accepted_count += 1
            #print("accepted")
            return theta_fin
        else:
            #print("rejected")
            return theta_ini

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
            
            if sim_paras['FA']:
                for _ in range(num_step_calib):
                    spin_config_calib = self.leapfrog_FA(spin_config_calib, lfl=lfl_cur, m_FA=sim_paras['m_FA'])
            else:
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
        self.logger.info('\nCalibration Failed: acceptance rate still went out of range')
        return True
        
    def measure(self, spin_config):
        '''
        Measure and save raw data of energy and magnetization.
        '''
        energy = compute_energy(spin_config, self.nbr) / self.N
        mx = np.mean(np.cos(spin_config))
        my = np.mean(np.sin(spin_config))
        m = np.sqrt(mx**2 + my**2) 

        self.data['energy'].append(energy)
        self.data['mx'].append(mx)
        self.data['my'].append(my)
        self.data['magnetization'].append(m)    
        
    def run_simulation(self, T=None, num_traj=int(1e4), lfl=10, lf_calib=True, max_sep_t=5, folder_temp=None, 
                       FA=False, m_FA=0.1, comp_vor=False, comp_heli_mod=False, vor_thld=0.01, 
                       write_times=10, log_freq=int(100)):
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
        # --------------------------------------------------------------------------------------------------------------------
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
        
        # --------------------------------------------------------------------------------------------------------------------
        # Burn-in stage
        
        self.accepted_count = self.attempts_count = 0  # Reset counters
        self.logger.info('-' * 100)
        
        equi_steps = int(0.1 * num_traj)  # Number of trajectories for burn-in（10%)
        
        with tqdm(total=equi_steps, desc="Equilibrating", ncols=100) as pbar:
            for n in range(equi_steps):
                if FA:
                    spin_config = self.leapfrog_FA(spin_config, sim_paras['lfl'], m_FA)
                else:
                    spin_config = self.leapfrog(spin_config, sim_paras['lfl'])
                    
                if (n + 1) % log_freq == 0:
                    pbar.update(log_freq)
                    self.logger.realtime(f'Equilibrating: trajectories {n + 1} / {equi_steps}')           

        self.logger.info(f"\nBurn-in Stage Completed (T = {T:.2f})")
        self.logger.info('-' * 100)
        
        # --------------------------------------------------------------------------------------------------------------------
        # Sampling stage
        
        self.accepted_count = self.attempts_count = 0  # Reset counters
        
        with tqdm(total=num_traj, desc="Sampling", ncols=100, leave=False) as pbar:
            for t in range(num_traj): 
                # Whether to implement FA
                if FA:
                    spin_config = self.leapfrog_FA(spin_config, sim_paras['lfl'], m_FA)
                else:
                    spin_config = self.leapfrog(spin_config, sim_paras['lfl'])
                
                if (t + 1) % log_freq == 0:
                    pbar.update(log_freq)
                    self.logger.realtime(f'Sampling: {t + 1} / {num_traj}')
                    
                # Save number of trajectories and difference of Hamiltonian
                self.data['num_traj'].append(t+1)
                self.data['delta_H'].append(self.delta_H)

                # Measure and save raw data (energy and magnetization)
                self.measure(spin_config)
                
                # Compute and save acceptance rate
                self.acc_rate = self.accepted_count / self.attempts_count  
                self.data['acc_rate'].append(self.acc_rate)
                
                # Compute and save current average gamma values and variances for correlations
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
                if comp_vor:
                    vor_num, avor_num, vor_den = compute_vor(spin_config, vor_thld)
                    self.data['vor_num'].append(int(vor_num))
                    self.data['avor_num'].append(int(avor_num))
                    self.data['vor_den'].append(vor_den)
                else:
                    self.data['vor_num'].append(np.nan)
                    self.data['avor_num'].append(np.nan)
                    self.data['vor_den'].append(np.nan)
                
                # Compute and save the helicity modulus
                if comp_heli_mod:
                    heli_mod = compute_heli_mod(spin_config, T)
                    self.data['heli_mod'].append(heli_mod)
                else:
                    self.data['heli_mod'].append(np.nan)

                # Raw data write-out
                if (t + 1) % (int(num_traj / write_times)) == 0:
                    data_write_out(self.data, raw_data_path)
            
            self.logger.info(f"\n\nSampling Stage Completed (T = {T:.2f})")
            self.logger.info('=' * 100)
        return spin_config

def data_write_out(data, raw_data_path):
    with open(raw_data_path, 'a') as f:
        if os.stat(raw_data_path).st_size == 0:  # Create header if file is newly created
            f.write("# num_traj energy magnetization mx my acc_rate delta_H vor_num avor_num vor_den heli_mod avg_gm var_gm\n")
        #print("lens", len(data['num_traj']), len(data['energy']), len(data['acc_rate']))
        for i in range(len(data['num_traj'])):  # Write data
            avg_gm_str = np.array2string(data['avg_gm_buffer'][i], precision=16, separator=',', suppress_small=True, max_line_width=np.inf)
            var_gm_str = np.array2string(data['var_gm_buffer'][i], precision=16, separator=',', suppress_small=True, max_line_width=np.inf)
            
            line = f"{data['num_traj'][i]} {data['energy'][i]:.16f} {data['magnetization'][i]:.16f} {data['mx'][i]:.16f} {data['my'][i]:.16f} {data['acc_rate'][i]:.16f} {data['delta_H'][i]:.16f} {data['vor_num'][i]} {data['avor_num'][i]} {data['vor_den'][i]:.16f} {data['heli_mod'][i]:.16f} {avg_gm_str} {var_gm_str}\n"
            
            f.write(line)
        
    # Reset data buffers after write-out
    for key in ['num_traj', 'energy', 'magnetization', 'mx', 'my', 'acc_rate', 'delta_H', 'vor_num', 'avor_num', 'vor_den', 'heli_mod', 'avg_gm_buffer', 'var_gm_buffer']:
        data[key].clear()

@njit(parallel=True, fastmath=True)  # Enable parallelization and fast math operations
def compute_energy(spin_config, nbr):
    """
    Compute the total energy of the system using Numba.
    """
    #print(spin_config)
    L = spin_config.shape[0]

    energy = 0.0
    
    for i in prange(L):
        for j in prange(L):
            for k in range(4):
                neighbor = nbr[i, j, k]
                ni, nj = neighbor // L, neighbor % L
                energy += - np.cos(spin_config[i, j] - spin_config[ni, nj])
    return energy / 2

@njit(parallel=True, fastmath=True)
def compute_action(spin_config, nbr, T):
    """
    Compute the action of the system using Numba.
    """
    L = spin_config.shape[0]
    action = 0.0
    inv_T = 1.0 / T
    for i in prange(L):
        for j in prange(L):
            for k in range(4):
                neighbor = nbr[i, j, k]
                ni, nj = neighbor // L, neighbor % L
                action -= np.cos(spin_config[i, j] - spin_config[ni, nj]) * inv_T
    return action / 2

@njit(parallel=True, fastmath=True)
def compute_deriv(spin_config, nbr, T):
    """
    Compute the derivative of the action using Numba.
    """
    L = spin_config.shape[0]
    deriv = np.zeros_like(spin_config)
    inv_T = 1.0 / T
    for i in prange(L):
        for j in prange(L):
            for k in range(4):
                neighbor = nbr[i, j, k]
                ni, nj = neighbor // L, neighbor % L
                spin_diff = spin_config[i, j] - spin_config[ni, nj]
                deriv[i, j] += np.sin(spin_diff) * inv_T
    return deriv

def compute_Hamiltonian(spin_config, momentum, nbr, T):
    """
    Compute the Hamiltonian of the system.
    """
    ke_matrix = 0.5 * momentum**2
    
    ke = np.sum(ke_matrix)
    pe = compute_action(spin_config, nbr, T)
    
    return ke + pe

def compute_Hamiltonian_FA(spin_config, momentum_k, K_tilde_inv, nbr, T, L):
    """
    Compute the Hamiltonian of the system in Fourier accelerated HMC.
    """
    # Compute the kinetic energy in Fourier space
    ke_matrix = 0.5 * np.conj(momentum_k) * K_tilde_inv * momentum_k / L ** 2
    #print("ke_matrix", ke_matrix)
    ke = np.sum(np.real(ke_matrix))
    
    # Compute the action in real space
    pe = compute_action(spin_config, nbr, T)
    
    #print("ke, pe (per site)", ke / L ** 2, pe / L ** 2)
    return ke + pe, ke_matrix

def inv_ft_kernel(L=10, m_FA=1.0):
    # Generate 1D frequency arrays (L/2 corresponding to Nyquist frequency for even L)
    kx = np.fft.fftfreq(L) * np.pi
    ky = np.fft.fftfreq(L) * np.pi

    # Create 2D meshgrid
    kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
    #print(kx, ky)
    # Construct the kernel
    K_ft = 4 * (np.sin(kx_2d)**2 + np.sin(ky_2d)**2) + m_FA**2
    K_ft_inv = 1 / K_ft
    #print("denominator with m_FA = 0", K_ft - m_FA**2, threshold=np.inf)
    return K_ft_inv

def gen_momentum(Pi_k):
    # Initialization
    L = Pi_k.shape[0]
    p_k = np.zeros((L, L), dtype=np.complex128)
    
    # Assign components
    # 1) four vertices
    p_k[0, 0] = Pi_k[0, 0]
    p_k[L//2, 0] = Pi_k[L//2, 0]
    p_k[0, L//2] = Pi_k[0, L//2]
    p_k[L//2, L//2] = Pi_k[L//2, L//2]
    
    # 2) four bars on the inner edge
    p_k[0, 1:L//2] = (Pi_k[0, 1:L//2] + 1j * Pi_k[0, -1:-L//2:-1]) / np.sqrt(2)
    p_k[L//2, 1:L//2] = (Pi_k[L//2, 1:L//2] + 1j * Pi_k[L//2, -1:-L//2:-1]) / np.sqrt(2)
    p_k[1:L//2, 0] = (Pi_k[1:L//2, 0] + 1j * Pi_k[-1:-L//2:-1, 0]) / np.sqrt(2)
    p_k[1:L//2, L//2] = (Pi_k[1:L//2, L//2] + 1j * Pi_k[-1:-L//2:-1, L//2]) / np.sqrt(2)
    
    # 3) upper left square (real upper left square + imaginary upper right square)
    p_k[1:L//2, 1:L//2] = (Pi_k[1:L//2, 1:L//2] + 1j * Pi_k[1:L//2, -1:-L//2:-1]) / np.sqrt(2)
    
    # 4) lower left square (real lower left square + imaginary lower right square)
    p_k[(L//2+1):, 1:L//2] = (Pi_k[(L//2+1):, 1:L//2] + 1j * Pi_k[(L//2+1):, -1:-L//2:-1]) / np.sqrt(2)
    
    # Implement Hermitian symmetry
    # 1) four bars on the outer edge (conjugated with four bars on the inner edge)
    p_k[0, -1:-L//2:-1] = np.conjugate(p_k[0, 1:L//2])
    p_k[L//2, -1:-L//2:-1] = np.conjugate(p_k[L//2, 1:L//2])
    p_k[-1:-L//2:-1, 0] = np.conjugate(p_k[1:L//2, 0])
    p_k[-1:-L//2:-1, L//2] = np.conjugate(p_k[1:L//2, L//2])
    
    # 2) lower right square (conjugated with upper left square)
    p_k[-1:-L//2:-1, -1:-L//2:-1] = np.conjugate(p_k[1:L//2, 1:L//2])
    
    # 3) upper right square (conjugated with lower left square)
    p_k[L//2-1:0:-1, L//2+1:] = np.conjugate(p_k[L//2+1:, L//2-1:0:-1])

    return p_k
