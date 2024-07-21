import numpy as np

def compute_vor(spin_config, vor_thld = 0.01):
    # compute the vortex density for the given configuration
    L = int(np.sqrt(len(spin_config)))
    
    vor_num = 0
    avor_num = 0
    
    for x in range(L):
        for y in range(L):
            # obtain the indices of four vertices of current plaquette
            st1 = x * L + y # beginning site
            st2 = ((x + 1) % L) * L + y # top
            st3 = ((x + 1) % L) * L + (y + 1) % L # diagonal
            st4 = x * L + (y + 1) % L # right
            
            dtheta1 = (spin_config[st2] - spin_config[st1]) % (2 * np.pi)
            dtheta2 = (spin_config[st3] - spin_config[st2]) % (2 * np.pi)
            dtheta3 = (spin_config[st4] - spin_config[st3]) % (2 * np.pi)
            dtheta4 = (spin_config[st1] - spin_config[st4]) % (2 * np.pi)
            
            dtheta1 = ((dtheta1 + np.pi) % (2 * np.pi)) - np.pi
            dtheta2 = ((dtheta2 + np.pi) % (2 * np.pi)) - np.pi
            dtheta3 = ((dtheta3 + np.pi) % (2 * np.pi)) - np.pi
            dtheta4 = ((dtheta4 + np.pi) % (2 * np.pi)) - np.pi
                        
            v = (dtheta1 + dtheta2 + dtheta3 + dtheta4) / (2 * np.pi)
            
            if abs(1 - abs(v)) < vor_thld:  # NB: the difference is around 1e-16 when L=25
                if v > 0:
                    vor_num += 1
                else:
                    avor_num += 1
            
    vor_den = vor_num / (L * L)
    return vor_num, avor_num, vor_den
