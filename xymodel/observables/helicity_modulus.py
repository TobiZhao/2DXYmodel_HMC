import numpy as np

def hm_func(spin1, spin2, T):
    d_prod = np.dot(spin1, spin2)
    cr_prod = np.cross(spin1, spin2)
    
    hm_func = (d_prod - cr_prod * cr_prod / T) / T
    return hm_func

def compute_heli_mod(spin_config, T = 1.02):
    # compute the helical modulus for the given configuration
    N = len(spin_config)
    L = int(np.sqrt(N))
    
    hm = 0
    
    for x in range(L):
        for y in range(L):
            # obtain the indices of the current site, its right, and its top
            st = x * L + y # current site
            st_r = ((x + 1) % L) * L + y # top
            st_t = x * L + (y + 1) % L # right

            s_st = (np.cos(spin_config[st]), np.sin(spin_config[st]))
            s_st_r = (np.cos(spin_config[st_r]), np.sin(spin_config[st_r]))
            s_st_t = (np.cos(spin_config[st_t]), np.sin(spin_config[st_t]))
            
            hm += hm_func(s_st, s_st_r, T) + hm_func(s_st, s_st_t, T)

    heli_mod = hm / N
    return heli_mod
