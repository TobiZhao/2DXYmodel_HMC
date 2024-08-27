import numpy as np

def inv_ft_kernel(L=10, m_FA=1.0):
    # Generate 1D frequency arrays (L/2 corresponding to Nyquist frequency for even L)
    kx = np.fft.fftfreq(L) * np.pi
    ky = np.fft.fftfreq(L) * np.pi

    # Create 2D meshgrid
    kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
    
    # Construct the kernel
    K_ft = 4 * (np.sin(kx_2d)**2 + np.sin(ky_2d)**2) + m_FA**2
    K_ft_inv = 1 / K_ft
    
    return K_ft, K_ft_inv

k, kinv = inv_ft_kernel(L=4, m_FA=10)

print("k", k)
print("kinv", kinv)
