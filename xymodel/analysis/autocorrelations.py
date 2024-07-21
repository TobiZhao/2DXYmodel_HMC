# Autocorrelations

import numpy as np

from scipy.optimize import curve_fit

def compute_autocorre_func(data_temp, max_sep_n):
    # compute the autocorrelation function (normalized autocovariance function) for a set of samples (requiring the frequency of measurement equaling to 1), with the observable being the total magnetization of the system
    
    rho = []
    se_rho = []
    
    mean_mag = np.mean(data_temp['magnetization_all'])
    
    sep_n = np.arange(1, max_sep_n+1)
    
    for j in sep_n:
        As = [((data_temp['magnetization_all'][i] - mean_mag)) * ((data_temp['magnetization_all'][i+j] - mean_mag)) for i in range(len(data_temp['magnetization_all'])-j)]
    
        A_n = np.mean(As) # lag-j autocovariance function
        se_A_n = np.std(As, ddof = 1) / np.sqrt(len(As))
        
        A_0 = np.std(data_temp['magnetization_all'], ddof = 0) ** 2 # variance recovered at separation of n is equal to zero

        # normalization
        rho_n = A_n / A_0 
        se_rho_n = se_A_n / A_0 
        
        rho.append(rho_n)
        se_rho.append(se_rho_n)
        
    return rho, se_rho

def autocorre_func_fit_model(n, b, tau):
    # define a model function to fit the correlation function
    
    return b * np.exp(- n / tau)

def fit_autocorre_func(rho, p0_tau):
    # fit the autocorrelation function to the exponential decay model, and obtain decocorrelation time
        
    sep_n = np.arange(1, len(rho)+1)
    popt, pcov = curve_fit(lambda n, b, tau: autocorre_func_fit_model(n, b, tau), sep_n, rho, p0 = [np.max(rho), p0_tau])
    
    # optimal values and standard deviations of b and tau
    b, tau = popt
    
    b_sd = np.sqrt(pcov[0, 0])
    tau_sd = np.sqrt(pcov[1, 1])

    return b, tau, b_sd, tau_sd 

def compute_int_autocorre_time(data_temp):
    # compute the integrated autocorrelation time tau_int
    
    L = data_temp['L']
    N = L * L
    
    for M in range(1, len(data_temp['magnetization_all'])):
        tau_int = 0.5 + np.sum(data_temp['rho'][0:M])
        
        # conditions in arxiv:1912.10997v3 (page 56) are used here as reference
        if M == (4 * tau_int + 1) or M > (4 * tau_int + 1):
            se_tau_int = np.sqrt((4 * M + 2) / N) * tau_int
            
            data_temp['cutoff_win'] = int(M)
            data_temp['tau_int'] = tau_int
            data_temp['se_tau_int'] = se_tau_int
            break
