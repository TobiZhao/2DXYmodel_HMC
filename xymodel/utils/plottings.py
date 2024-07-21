# Visualization

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from xymodel import *

#================================================================================================================================================================================================        

# plotting for the given spin configuration

def plot_spin_config(data, spin_config, path = None):  
    # plot the given configuration
    
    # set figure size
    fig, ax = plt.subplots(figsize=(16, 16))

    # reshape 1D list into 2D array
    spin_matrix = np.reshape(spin_config, (data['L'], data['L']))

    # construct the grid, each site corresponding to one spin
    # x, y are 2D arrays
    x, y = np.meshgrid(np.arange(data['L']), np.arange(data['L']))

    # compute components
    u = np.cos(spin_matrix)
    v = np.sin(spin_matrix)

    # compute the angle of each spin
    angles = np.arctan2(v, u)

    # map angles to colors varying by degrees using colormap (cm)
    # recommendations ('scale', 'L'): (60, 64) (130, 128) (300, 256)
    quiver = ax.quiver(x, y, u, v, angles, pivot = 'middle', scale = 300, 
                        cmap = plt.cm.hsv, norm = plt.Normalize(vmin = - np.pi, vmax = np.pi))

    # set axes and title
    ax.set_aspect('equal')

    ax.set_xlim(-0.5, data['L'] - 0.5)
    ax.set_ylim(-0.5, data['L'] - 0.5)
    
    ax.set_title(f"2D XY Model ($N = {data['L']}^2$ $T = {data['T']:.2f}$)")

    # create the colorbar
    cbar = fig.colorbar(quiver, ax = ax, boundaries = np.linspace(- np.pi, np.pi, 100), label = 'Spin angle')
    
    # save the figure    
    if path:
        plt.savefig(os.path.join(path, f"spin_config_T{data['T']:.2f}.png"), dpi = 500) 
        plt.close()
    else:
        plt.show()
        
    # save the spin configuration to a text file
    if path:
        np.savetxt(os.path.join(path, f"spin_config_T{data['T']:.2f}.txt"), spin_config)
    else:
        np.savetxt(f"spin_config_T{data['T']:.2f}.txt", spin_config)
        
#================================================================================================================================================================================================        

# plottings for data processing under each temperature

def plot_raw_data_basic(data_temp, path = None):
    # plot the raw data of energy, components of magnetization versus number of trajectory (with the given frequency of measurement)
    fig, axs = plt.subplots(3, figsize=(16, 12))
        
    scatter_size = 1.0

    axs[0].scatter(data_temp['num_traj'], data_temp['energy'], s = scatter_size)
    axs[0].set_xlabel('Number of Trajectories')
    axs[0].set_ylabel(r'$E$')
    axs[0].set_title(r'$E$ - Number of Trajectories')

    axs[1].scatter(data_temp['num_traj'], data_temp['mx'], s = scatter_size)
    axs[1].set_xlabel('Number of Trajectories')
    axs[1].set_ylabel(r'$M_x$')
    axs[1].set_title(r'$M_x$ - Number of Trajectories')
    
    axs[2].scatter(data_temp['num_traj'], data_temp['my'], s = scatter_size)
    axs[2].set_xlabel('Number of Trajectories')
    axs[2].set_ylabel(r'$M_y$')
    axs[2].set_title(r'$M_y$ - Number of Trajectories')

    plt.tight_layout()
    
    if path:
        plt.savefig(os.path.join(path, f"raw_data_basic_T{data_temp['T']:.2f}.png"), dpi = 500)
        plt.close()
        
    else:
        plt.show()
        
def plot_raw_data_vor_den(data_temp, path=None):
    # plot the curve of vortex density during sampling versus number of trajectory (with the given frequency of measurement)
    fig, ax = plt.subplots(figsize=(16, 16))
    
    scatter_size = 2.0

    ax.scatter(data_temp['num_traj'], data_temp['vor_den'], s = scatter_size)
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel(r'$\rho_{vor}$')
    ax.set_title(r'$\rho_{vor}$ - Number of Trajectories')

    plt.tight_layout()
    
    if path:
        plt.savefig(os.path.join(path, f"raw_data_vor_den_T{data_temp['T']:.2f}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_raw_data_heli_mod(data_temp, path=None):
    # plot the curve of helicity modulus versus number of trajectory (with the given frequency of measurement)
    fig, ax = plt.subplots(figsize=(16, 16))
    
    scatter_size = 2.0

    ax.scatter(data_temp['num_traj'], data_temp['heli_mod'], s = scatter_size)
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('Helicity Modulus')
    ax.set_title('Helicity Modulus - Number of Trajectories')

    plt.tight_layout()
    
    if path:
        plt.savefig(os.path.join(path, f"raw_data_heli_mod_T{data_temp['T']:.2f}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_ensemble_averages(data_temp, path = None):
    # plot the curves of ensemble average values of energy, magnetization components, and avg_exp_delta_H during sampling
    fig, axs = plt.subplots(3, figsize=(16, 12))

    scatter_size = 0.25
        
    axs[0].scatter(data_temp['num_traj'], data_temp['avg_exp_delta_H_cur'], s = scatter_size)
    axs[0].set_xlabel('Number of Trajectories')
    axs[0].set_ylabel(r'$\langle e^{- \Delta H} \rangle$')
    axs[0].set_title(r'$\langle e^{- \Delta H} \rangle$ - Number of Trajectories')

    axs[1].scatter(data_temp['num_traj'], data_temp['avg_mx_cur'], s = scatter_size)
    axs[1].set_xlabel('Number of Trajectories')
    axs[1].set_ylabel(r'$\langle M_x \rangle$')
    axs[1].set_title(r'$\langle M_x \rangle$ - Number of Trajectories')
    
    axs[2].scatter(data_temp['num_traj'], data_temp['avg_my_cur'], s = scatter_size)
    axs[2].set_xlabel('Number of Trajectories')
    axs[2].set_ylabel(r'$\langle M_y \rangle$')
    axs[2].set_title(r'$\langle M_y \rangle$ - Number of Trajectories')
    
    plt.tight_layout()
    
    if path:
        plt.savefig(os.path.join(path, f"ensemble_averages_T{data_temp['T']:.2f}.png"), dpi = 500)
        plt.close()
        
    else:
        plt.show()

def plot_correlations(data_temp, path = None):
    fig, axs = plt.subplots(2, figsize=(16, 12))

    scatter_size = 0.5

    sep_t = np.arange(1, len(data_temp['avg_gm'])+1)
    gm_fit = [corre_func_fit_model(t, data_temp['corre_fit_c'], data_temp['corre_fit_m'], data_temp['L']) for t in sep_t]

    axs[0].errorbar(sep_t, data_temp['eff_m'], yerr = data_temp['se_eff_m'], fmt = 'o', capsize = 3, markersize = scatter_size, ecolor = 'lightgray')
    axs[0].set_xlabel(r'Separation $t$')
    axs[0].set_ylabel(r'Effective Mass $m_{eff}$')
    axs[0].set_title(r'$m_{eff}$ - $t$')
    axs[0].xaxis.set_major_locator(ticker.MaxNLocator(integer = True))
    
    axs[1].errorbar(sep_t, data_temp['avg_gm'], yerr = data_temp['se_gm'], fmt = 'o', capsize = 3, markersize = scatter_size, ecolor = 'lightgray', label = 'Original Gamma')
    axs[1].plot(sep_t, gm_fit, color = 'red', label = f"Fit: $c = {data_temp['corre_fit_c']:.5f}, m = {data_temp['corre_fit_m']:.5f}, \\xi = {data_temp['corre_len']}$")
    axs[1].set_xlabel(r'Space Separation $t$')
    axs[1].set_ylabel(r'Correlation Function $\Gamma(t)$')
    axs[1].set_title(r'Fit of $\Gamma(t)$ - $t$')
    axs[1].legend()
    axs[1].xaxis.set_major_locator(ticker.MaxNLocator(integer = True))
        
    plt.tight_layout()

    if path:
        plt.savefig(os.path.join(path, f"correlations_T{data_temp['T']:.2f}.png"), dpi = 500)
        plt.close()
    else:
        plt.show()

def plot_autocorrelations(data_temp, path = None):
    fig, ax = plt.subplots(figsize=(16, 16))

    scatter_size = 0.5
    
    sep_n = np.arange(1, len(data_temp['rho'])+1)
    rho_fit = [autocorre_func_fit_model(n, data_temp['autocorre_fit_b'], data_temp['autocorre_fit_tau']) for n in sep_n]
    
    ax.errorbar(sep_n, data_temp['rho'], yerr = data_temp['se_rho'], fmt = 'o', capsize = 3, markersize = scatter_size, ecolor = 'lightgray', label = 'Original Gamma')
    ax.plot(sep_n, rho_fit, color = 'red', label = f"Fit: $b = {data_temp['autocorre_fit_b']:.5f}, tau = {data_temp['autocorre_fit_tau']:.5f}$")
    ax.set_xlabel(r'MC Time Separation $n$')
    ax.set_ylabel(r'Autocorrelation Function $\rho(n)$')
    ax.set_title(r'Fit of $\rho(n)$ - $n$')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))   
     
    plt.tight_layout()

    if path:
        plt.savefig(os.path.join(path, f"autocorrelations_T{data_temp['T']:.2f}.png"), dpi = 500)
        plt.close()
    else:
        plt.show()

#================================================================================================================================================================================================

# plottings for data processing under a range of temperatures

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# scaling parameters

ps_scaling = 5
data_point_size = 1 * ps_scaling
scatter_point_size = 10 * ps_scaling
errorbar_capsize = 3.0
errorbar_elinewidth = 1.5
errorbar_capthick = 0.6
scatter_linewidth = 1.0
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_energy(data_range, ref_data, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['energy'], yerr=data_range['se_energy'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.scatter(ref_data['temp'], ref_data['energy'], marker='s', s=scatter_point_size, color='r', 
               facecolors='None', linewidths=scatter_linewidth, label='Gupta 1992')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Average Energy')
    ax.set_title(f"L={data_range['L']} Average Energy vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_energy_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_spec_heat(data_range, ref_data, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['sh'], yerr=data_range['se_sh'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.scatter(ref_data['temp'], ref_data['sh'], marker='s', s=scatter_point_size, color='r', 
               facecolors='None', linewidths=scatter_linewidth, label='Gupta 1992')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Specific Heat')
    ax.set_title(f"L={data_range['L']} Specific Heat vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_sh_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_susceptibility(data_range, ref_data, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['sus'], yerr=data_range['se_sus'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.scatter(ref_data['temp'], ref_data['sus'], marker='s', s=scatter_point_size, color='r', 
               facecolors='None', linewidths=scatter_linewidth, label='Gupta 1992')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Susceptibility')
    ax.set_title(f"L={data_range['L']} Susceptibility vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_sus_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_vor_den(data_range, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['vor_den'], yerr=data_range['se_vor_den'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('10^3 Vortex Density')
    ax.set_title(f"L={data_range['L']} Vortex Density vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_vor_den_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_heli_mod(data_range, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['heli_mod'], yerr=data_range['se_heli_mod'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Helicity Modulus')
    ax.set_title(f"L={data_range['L']} Helicity Modulus vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_heli_mod_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()
        
def plot_bdc(data_range, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['bdc'], yerr=data_range['se_bdc'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Binder Cumulant')
    ax.set_title(f"L={data_range['L']} Binder Cumulant vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_bdc_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_corre_fit_c(data_range, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['corre_fit_c'], yerr=data_range['sd_corre_fit_c'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Correlation Function Fit C')
    ax.set_title(f"L={data_range['L']} Correlation Function Fit C vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_corre_fit_c_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_corre_fit_m(data_range, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['corre_fit_m'], yerr=data_range['sd_corre_fit_m'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Correlation Function Fit m')
    ax.set_title(f"L={data_range['L']} Correlation Function Fit m vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_corre_fit_m_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_corre_len(data_range, ref_data, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['corre_len'], yerr=data_range['sd_corre_len'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.scatter(ref_data['temp'], ref_data['corre_len'], marker='s', s=scatter_point_size, color='r', 
               facecolors='None', linewidths=scatter_linewidth, label='Gupta 1992')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Correlation Length')
    ax.set_title(f"L={data_range['L']} Correlation Length vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_corre_len_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_autocorre_fit_b(data_range, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['autocorre_fit_b'], yerr=data_range['sd_autocorre_fit_b'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Autocorrelation Function Fit B')
    ax.set_title(f"L={data_range['L']} Autocorrelation Function Fit B vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_autocorre_fit_b_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_autocorre_fit_tau(data_range, ref_data, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['autocorre_fit_tau'], yerr=data_range['sd_autocorre_fit_tau'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.scatter(ref_data['temp'], ref_data['autocorre_fit_tau'], marker='s', s=scatter_point_size, color='r', 
               facecolors='None', linewidths=scatter_linewidth, label='Gupta 1992')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Autocorrelation Function Fit Tau')
    ax.set_title(f"L={data_range['L']} Autocorrelation Function Fit Tau vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_autocorre_fit_tau_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_cutoff_win(data_range, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.scatter(data_range['temp'], data_range['cutoff_win'], c='gray', s=data_point_size, label='HMC')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Cutoff Window')
    ax.set_title(f"L={data_range['L']} Cutoff Window vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_cutoff_win_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_tau_int(data_range, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.errorbar(data_range['temp'], data_range['tau_int'], yerr=data_range['sd_tau_int'], fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Integrated Autocorrelation Time')
    ax.set_title(f"L={data_range['L']} Integrated Autocorrelation Time vs Temperature")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_tau_int_vs_temp_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()

def plot_critical_exponent(data_range, path=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    
    corre_len = np.array(data_range['corre_len'])
    autocorre_fit_tau = np.array(data_range['autocorre_fit_tau'])
    sd_corre_len = np.array(data_range['sd_corre_len'])
    sd_autocorre_fit_tau = np.array(data_range['sd_autocorre_fit_tau'])

    # filter invalid values
    valid_indices = (corre_len > 0) & (autocorre_fit_tau > 0)
    corre_len = corre_len[valid_indices]
    autocorre_fit_tau = autocorre_fit_tau[valid_indices]
    sd_corre_len = sd_corre_len[valid_indices]
    sd_autocorre_fit_tau = sd_autocorre_fit_tau[valid_indices]

    ln_corre_len = np.log(corre_len)
    ln_corre_fit_tau = np.log(autocorre_fit_tau)
    
    ax.errorbar(ln_corre_len, ln_corre_fit_tau, xerr=sd_corre_len/corre_len, 
                yerr=sd_autocorre_fit_tau/autocorre_fit_tau, fmt='o', ecolor='gray', 
                ms=data_point_size, capsize=errorbar_capsize, elinewidth=errorbar_elinewidth, 
                capthick=errorbar_capthick, label='HMC')
    
    ax.set_xlabel('ln(Correlation Length)')
    ax.set_ylabel('ln(Correlation Fit Tau)')
    ax.set_title(f"L={data_range['L']} ln(Correlation Length) vs ln(Correlation Fit Tau)")
    ax.legend(loc='lower right')
    
    if path:
        plt.savefig(os.path.join(path, f"L{data_range['L']}_critical_exponent_mf{data_range['meas_freq']}.png"), dpi=500)
        plt.close()
    else:
        plt.show()
