#===============================================================================
# Vortex Analysis
#===============================================================================

import numpy as np

def compute_vor(spin_config, vor_thld=0.01):
    """
    Compute the vortex density for a given spin configuration.

    Args:
        spin_config (np.array): 1D array representing the spin angles on a 2D lattice.
        vor_thld (float): Threshold for identifying vortices. Default is 0.01.

    Returns:
        tuple: Number of vortices, number of anti-vortices, and vortex density.
    """
    L = spin_config.shape[0]  # Lattice size
    
    vor_num = 0  # Vortex count
    avor_num = 0  # Anti-vortex count
    
    for x in range(L):
        for y in range(L):
            # Compute phase differences along the plaquette
            dtheta1 = (spin_config[(x+1)%L, y] - spin_config[x, y]) % (2 * np.pi)
            dtheta2 = (spin_config[(x+1)%L, (y+1)%L] - spin_config[(x+1)%L, y]) % (2 * np.pi)
            dtheta3 = (spin_config[x, (y+1)%L] - spin_config[(x+1)%L, (y+1)%L]) % (2 * np.pi)
            dtheta4 = (spin_config[x, y] - spin_config[x, (y+1)%L]) % (2 * np.pi)
            
            # Map phase differences to [-π, π] interval
            dtheta1 = ((dtheta1 + np.pi) % (2 * np.pi)) - np.pi
            dtheta2 = ((dtheta2 + np.pi) % (2 * np.pi)) - np.pi
            dtheta3 = ((dtheta3 + np.pi) % (2 * np.pi)) - np.pi
            dtheta4 = ((dtheta4 + np.pi) % (2 * np.pi)) - np.pi
            
            # Compute vorticity (winding number)
            v = (dtheta1 + dtheta2 + dtheta3 + dtheta4) / (2 * np.pi)
            
            # Identify vortices and anti-vortices
            if abs(1 - abs(v)) < vor_thld:
                if v > 0:
                    vor_num += 1  # Vortex
                else:
                    avor_num += 1  # Anti-vortex
    
    vor_den = vor_num / (L * L)  # Vortex density
    
    return vor_num, avor_num, vor_den
