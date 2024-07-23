#===============================================================================
# Autocorrelation Analysis
#===============================================================================

import numpy as np
from scipy.optimize import curve_fit

def compute_autocorre_func(data_temp, max_sep_n):
    """
    Compute the autocorrelation function (normalized autocovariance function) for a set of samples.
    
    Args:
        data_temp (dict): Dictionary containing magnetization data.
        max_sep_n (int): Maximum separation to compute autocorrelation for.
    
    Returns:
        tuple: Autocorrelation values and their standard errors.
    """
    rho = []
    se_rho = []
    
    mean_mag = np.mean(data_temp['magnetization_all'])
    
    for j in range(1, max_sep_n + 1):
        # Compute lag-j autocovariance
        As = [(data_temp['magnetization_all'][i] - mean_mag) * 
              (data_temp['magnetization_all'][i+j] - mean_mag) 
              for i in range(len(data_temp['magnetization_all'])-j)]
        
        A_n = np.mean(As)  # lag-j autocovariance function
        se_A_n = np.std(As, ddof=1) / np.sqrt(len(As))
        
        # Variance at zero separation
        A_0 = np.var(data_temp['magnetization_all'], ddof=0)
        
        # Normalize autocovariance
        rho_n = A_n / A_0
        se_rho_n = se_A_n / A_0
        
        rho.append(rho_n)
        se_rho.append(se_rho_n)
    
    return rho, se_rho

def autocorre_func_fit_model(n, b, tau):
    """
    Define an exponential decay model function to fit the autocorrelation function.
    """
    return b * np.exp(-n / tau)

def fit_autocorre_func(rho, p0_tau):
    """
    Fit the autocorrelation function to the exponential decay model.
    
    Args:
        rho (array): Autocorrelation values.
        p0_tau (float): Initial guess for tau parameter.
    
    Returns:
        tuple: Optimal parameter values and their standard deviations.
    """
    sep_n = np.arange(1, len(rho) + 1)
    popt, pcov = curve_fit(autocorre_func_fit_model, sep_n, rho, 
                           p0=[np.max(rho), p0_tau])
    
    # Extract optimal values and standard deviations
    b, tau = popt
    b_sd, tau_sd = np.sqrt(np.diag(pcov))
    
    return b, tau, b_sd, tau_sd

def compute_int_autocorre_time(data_temp):
    """
    Compute the integrated autocorrelation time tau_int with a cut-off window M.
    
    This function implements the windowing procedure described in 
    arXiv:1912.10997v3 (page 56).
    """
    L = data_temp['L']
    N = L * L
    
    for M in range(1, len(data_temp['magnetization_all'])):
        tau_int = 0.5 + np.sum(data_temp['rho'][:M])
        
        # Check windowing condition
        if M >= 4 * tau_int + 1:
            se_tau_int = np.sqrt((4 * M + 2) / N) * tau_int
            
            # Store results
            data_temp['cutoff_win'] = int(M)
            data_temp['tau_int'] = tau_int
            data_temp['se_tau_int'] = se_tau_int
            break
