# Correlations

import numpy as np

from scipy.optimize import curve_fit

def compute_zmo(spin_config):
    # compute and save zero momentum operators of each row/column of an equilibrium spin configuration
    
    L = int(np.sqrt(len(spin_config)))
    
    spin_matrix = np.reshape(spin_config, (L, L))
    
    o_x_row = np.sum(np.cos(spin_matrix), axis = 1)
    o_y_row = np.sum(np.sin(spin_matrix), axis = 1)
    
    o_x_col = np.sum(np.cos(spin_matrix), axis = -1)
    o_y_col = np.sum(np.sin(spin_matrix), axis = -1)
    
    o_row = np.stack((o_x_row, o_y_row), axis = -1) # shaped (L, 2)
    o_col = np.stack((o_x_col, o_y_col), axis = -1)
    
    return o_row, o_col

def compute_corre_func(o, t):
    # compute the correlation function for a given space separation t
    
    L = int(o.shape[0])
    
    gms = [np.dot(o[i], o[(i+t) % L]) for i in range(L)] # considering periodic boundary condition
    gm_t = np.mean(gms) 
    
    return gm_t

def compute_eff_mass(gm, var_gm, L):
    # compute effective mass and its error
    
    max_sep_t = len(gm) 
    
    eff_m = np.zeros(max_sep_t)
    se_eff_m = np.zeros(max_sep_t)
    
    eff_m[0] = np.nan
    se_eff_m[0] = np.nan

    sd_gm = np.sqrt(var_gm)

    for k in range(1, max_sep_t):
        N_eff = L # the effective number of samples here is given by the harmonized sample sizes of two neighboring sample sets, and the result is equal to the length
        eff_m[k] = np.log(gm[k-1] / gm[k])
        se_eff_m[k] = np.sqrt((sd_gm[k-1] / gm[k-1])**2 + (sd_gm[k] / gm[k])**2) / N_eff # from the error propagation formular
    
    return eff_m, se_eff_m

def corre_func_fit_model(t, c, m, L):
    # define a model function to fit the correlation function
    
    return c * np.cosh(m * (L / 2 - t))

def fit_corre_func(gm, L):
    # fit the correlation function to the cosh model, and obtain effective mass 
    
    sep_t = np.arange(1, len(gm)+1)
    popt, pcov = curve_fit(lambda t, c, m: corre_func_fit_model(t, c, m, L), sep_t, gm, p0 = [np.max(gm), 0.03]) # use lambda method to create a temporary function in order to fix value of L
    
    # optimal values and standard deviations of c and m
    c, m = popt
    
    c_sd = np.sqrt(pcov[0, 0])
    m_sd = np.sqrt(pcov[1, 1])
    
    return c, abs(m), c_sd, m_sd

# =================================================================================================

def proc_corre(spin_config, max_sep_t):
    # compute zero momentum operators
    o_row, o_col = compute_zmo(spin_config)
    
    gm = []
    
    for sep_t in range(1, max_sep_t + 1):
        gm_t_row = compute_corre_func(o_row, sep_t)
        gm_t_col = compute_corre_func(o_col, sep_t)
        
        gm_t = (gm_t_row + gm_t_col) / 2
        gm.append(gm_t)
    
    return gm
