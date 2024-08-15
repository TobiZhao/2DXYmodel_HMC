#===============================================================================
# Helicity Modulus Calculation Functions
#===============================================================================

import numpy as np

def hm_func(spin1, spin2, T):
    """
    Calculate the helicity modulus function for a pair of spins.

    Args:
        spin1 (tuple): (cos(theta), sin(theta)) for the first spin
        spin2 (tuple): (cos(theta), sin(theta)) for the second spin
        T (float): Temperature

    Returns:
        float: Helicity modulus function value
    """
    # Compute dot product and cross product of spins
    d_prod = np.dot(spin1, spin2)
    cr_prod = np.cross(spin1, spin2)
    
    # Calculate helicity modulus function
    hm_func = (d_prod - cr_prod * cr_prod / T) / T
    return hm_func

def compute_heli_mod(spin_config, T=1.02):
    """
    Compute the helicity modulus for a given spin configuration.

    The helicity modulus is a measure of the system's response to a twist 
    in the boundary conditions, related to the superfluid density.

    Args:
        spin_config (np.array): 1D array of spin angles
        T (float): Temperature. Default is 1.02.

    Returns:
        float: Helicity modulus of the system
    """
    L = spin_config.shape[0]
    hm = 0.0
    
    for x in range(L):
        for y in range(L):
            # Current spin
            s_st = np.array([np.cos(spin_config[x, y]), np.sin(spin_config[x, y])])
            
            # Right neighbor (with periodic boundary)
            s_st_r = np.array([np.cos(spin_config[(x+1)%L, y]), np.sin(spin_config[(x+1)%L, y])])
            
            # Top neighbor (with periodic boundary)
            s_st_t = np.array([np.cos(spin_config[x, (y+1)%L]), np.sin(spin_config[x, (y+1)%L])])
            
            # Compute helicity modulus contributions
            # Right neighbor
            d_prod_r = np.dot(s_st, s_st_r)
            cr_prod_r = s_st[0]*s_st_r[1] - s_st[1]*s_st_r[0]
            hm += (d_prod_r - cr_prod_r * cr_prod_r / T) / T
            
            # Top neighbor
            d_prod_t = np.dot(s_st, s_st_t)
            cr_prod_t = s_st[0]*s_st_t[1] - s_st[1]*s_st_t[0]
            hm += (d_prod_t - cr_prod_t * cr_prod_t / T) / T
    
    # Normalize by system size
    heli_mod = hm / (L * L)
    return heli_mod
