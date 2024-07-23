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
    # System parameters
    N = len(spin_config)
    L = int(np.sqrt(N))  # Lattice size
    
    hm = 0  # Accumulator for helicity modulus
    
    # Iterate over all lattice sites
    for x in range(L):
        for y in range(L):
            # Calculate indices for current site and its neighbors
            st = x * L + y           # Current site
            st_r = ((x + 1) % L) * L + y  # Right neighbor (with periodic boundary)
            st_t = x * L + (y + 1) % L    # Top neighbor (with periodic boundary)
            
            # Convert spin angles to 2D unit vectors
            s_st = (np.cos(spin_config[st]), np.sin(spin_config[st]))
            s_st_r = (np.cos(spin_config[st_r]), np.sin(spin_config[st_r]))
            s_st_t = (np.cos(spin_config[st_t]), np.sin(spin_config[st_t]))
            
            # Accumulate helicity modulus contributions
            hm += hm_func(s_st, s_st_r, T) + hm_func(s_st, s_st_t, T)
    
    # Normalize by system size
    heli_mod = hm / N
    return heli_mod
