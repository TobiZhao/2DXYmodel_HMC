#===============================================================================
# Correlation Functions and Effective Mass Calculations for XY Model
#===============================================================================

import numpy as np
from scipy.optimize import curve_fit

def compute_zmo(spin_config):
    """
    Compute zero momentum operators for each row/column of an equilibrium spin configuration.

    Args:
        spin_config (np.array): 1D array of spin angles

    Returns:
        tuple: Zero momentum operators for rows and columns
    """
    L = int(np.sqrt(len(spin_config)))
    spin_matrix = np.reshape(spin_config, (L, L))

    o_x_row = np.sum(np.cos(spin_matrix), axis=1)
    o_y_row = np.sum(np.sin(spin_matrix), axis=1)
    o_x_col = np.sum(np.cos(spin_matrix), axis=0)
    o_y_col = np.sum(np.sin(spin_matrix), axis=0)

    o_row = np.stack((o_x_row, o_y_row), axis=-1)  # shaped (L, 2)
    o_col = np.stack((o_x_col, o_y_col), axis=-1)

    return o_row, o_col

def compute_corre_func(o, t):
    """
    Compute the correlation function for a given space separation t.

    Args:
        o (np.array): Zero momentum operators
        t (int): Space separation

    Returns:
        float: Correlation function value
    """
    L = int(o.shape[0])
    gms = [np.dot(o[i], o[(i+t) % L]) for i in range(L)]  # Periodic boundary condition
    return np.mean(gms)

def compute_eff_mass(gm, var_gm, L):
    """
    Compute effective mass and its error.

    Args:
        gm (np.array): Correlation function values
        var_gm (np.array): Variances of correlation function values
        L (int): System size

    Returns:
        tuple: Effective mass and its standard error
    """
    max_sep_t = len(gm)
    eff_m = np.zeros(max_sep_t)
    se_eff_m = np.zeros(max_sep_t)

    eff_m[0] = se_eff_m[0] = np.nan
    sd_gm = np.sqrt(var_gm)

    for k in range(1, max_sep_t):
        N_eff = L  # Effective number of samples
        eff_m[k] = np.log(gm[k-1] / gm[k])
        se_eff_m[k] = np.sqrt((sd_gm[k-1] / gm[k-1])**2 + (sd_gm[k] / gm[k])**2) / N_eff  # Error propagation

    return eff_m, se_eff_m

def corre_func_fit_model(t, c, m, L):
    """
    Model function to fit the correlation function.

    Args:
        t (np.array): Space separations
        c (float): Amplitude parameter
        m (float): Mass parameter
        L (int): System size

    Returns:
        np.array: Model predictions
    """
    return c * np.cosh(m * (L / 2 - t))

def fit_corre_func(gm, L):
    """
    Fit the correlation function to the cosh model and obtain effective mass.

    Args:
        gm (np.array): Correlation function values
        L (int): System size

    Returns:
        tuple: Optimal parameter values and their standard deviations
    """
    sep_t = np.arange(1, len(gm)+1)
    popt, pcov = curve_fit(lambda t, c, m: corre_func_fit_model(t, c, m, L), 
                           sep_t, gm, p0=[np.max(gm), 0.03])

    c, m = popt
    c_sd, m_sd = np.sqrt(np.diag(pcov))

    return c, abs(m), c_sd, m_sd

def proc_corre(spin_config, max_sep_t):
    """
    Process correlation functions for a given spin configuration.

    Args:
        spin_config (np.array): 1D array of spin angles
        max_sep_t (int): Maximum separation to compute correlations for

    Returns:
        list: Correlation function values
    """
    o_row, o_col = compute_zmo(spin_config)
    gm = []

    for sep_t in range(1, max_sep_t + 1):
        gm_t_row = compute_corre_func(o_row, sep_t)
        gm_t_col = compute_corre_func(o_col, sep_t)
        gm_t = (gm_t_row + gm_t_col) / 2
        gm.append(gm_t)

    return gm
