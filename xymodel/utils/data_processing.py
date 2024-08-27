#===============================================================================
# Data Processing Functions for 2D XY Model Simulation Results
#===============================================================================

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from xymodel import *

#===============================================================================
# Loading Reference Data
#===============================================================================

def load_ref_data(ref_data_file):
    """
    Load reference data from a file and organize it into a dictionary.

    Parameters:
        ref_data_file (str): Path to the reference data file.

    Returns:
        dict: A dictionary containing various physical quantities as reference.
    """
    # Initialize dictionary to store different physical quantities
    ref_data = {
        'temp': [],                # Temperature
        'autocorre_fit_tau': [],   # Autocorrelation fit tau
        'energy': [],              # Energy (note: stored as negative of input)
        'sh': [],                  # Specific heat
        'sus': [],                 # Susceptibility
        'corre_len': [],           # Correlation length
        'corre_len_av': []         # Average correlation length
    }

    with open(ref_data_file, 'r') as f:
        # Skip the header line
        next(f)
       
        for line in f:
            # Split each line into values
            values = line.split()
                       
            # Populate the ref_data dictionary
            ref_data['temp'].append(float(values[0]))
            ref_data['autocorre_fit_tau'].append(float(values[1]))
            ref_data['energy'].append(-float(values[2]))  # Note the negation
            ref_data['sh'].append(float(values[3]))
            ref_data['sus'].append(float(values[4]))
            
            # Handle potential '-' values for correlation lengths
            ref_data['corre_len'].append(float(values[5]) if values[5] != '-' else float('nan'))
            ref_data['corre_len_av'].append(float(values[6]) if values[6] != '-' else float('nan'))

    return ref_data

#===============================================================================
# Temperature Point Processing
#===============================================================================

def process_data_temp_point(para_data, data_folder):
    """
    Process data for a specific temperature point from a batch of jobs with raw data.
    
    This function processes data from multiple subfolders, and computes various physical quantities and statistics in each.

    Parameters:
        para_data (dict): Parameters for data processing.
        data_folder (str): Path to the folder containing job data.
    
    Returns:
        A json file containing values of processed data.
    """
    # Iterate through subfolders in the data_folder
    for subfolder in os.listdir(data_folder):
        folder_temp = os.path.join(data_folder, subfolder)
        
        print('Processing folder:', folder_temp)
        
        # Initialize dictionary to store processed data
        data_temp = {
            'T': np.nan,                     # Temperature
            'L': np.nan,                     # System size
            'meas_freq': para_data['meas_freq'],
            'max_sep_t': para_data['max_sep_t'],
            'max_sep_n': para_data['max_sep_n'],
            'num_traj': [],                  # Number of trajectories
            'energy': [],                    # Energy values
            'magnetization': [],             # Magnetization values
            'magnetization_all': [],         # All magnetization data for autocorrelations
            'mx': [],                        # x-component of magnetization
            'my': [],                        # y-component of magnetization
            'delta_H': [],                   # Change in Hamiltonian
            'vor_den': [],                   # Vortex density
            'heli_mod': [],                  # Helicity modulus
            'avg_mx_cur': [],                # Running average of mx
            'avg_my_cur': [],                # Running average of my
            'avg_exp_delta_H_cur': [],       # Running average of exp(-delta_H)
            'acc_rate_tot': np.nan,          # Total acceptance rate
            'avg_energy': np.nan,            # Average energy
            'se_energy': np.nan,             # Standard error of energy
            'sus': np.nan,                   # Susceptibility
            'se_sus': np.nan,                # Standard error of susceptibility
            'avg_vor_den': np.nan,           # Average vortex density
            'se_vor_den': np.nan,            # Standard error of vortex density
            'avg_heli_mod': np.nan,          # Average helicity modulus
            'se_heli_mod': np.nan,           # Standard error of helicity modulus
            'bdc': np.nan,                   # Binder cumulant
            'se_bdc': np.nan,                # Standard error of Binder cumulant
            'avg_gm': [],                    # Average correlation function
            'se_gm': [],                     # Standard error of correlation function
            'eff_m': [],                     # Effective mass
            'se_eff_m': [],                  # Standard error of effective mass
            'corre_fit_c': np.nan,           # Correlation fit parameter c
            'sd_corre_fit_c': np.nan,        # Standard deviation of c
            'corre_fit_m': np.nan,           # Correlation fit parameter m
            'sd_corre_fit_m': np.nan,        # Standard deviation of m
            'corre_len': np.nan,             # Correlation length
            'sd_corre_len': np.nan,          # Standard deviation of correlation length
            'rho': [],                       # Autocorrelation function
            'se_rho': [],                    # Standard error of autocorrelation function
            'autocorre_fit_b': np.nan,       # Autocorrelation fit parameter b
            'sd_autocorre_fit_b': np.nan,    # Standard deviation of b
            'autocorre_fit_tau': np.nan,     # Autocorrelation fit parameter tau
            'sd_autocorre_fit_tau': np.nan,  # Standard deviation of tau
            'cutoff_win': np.nan,            # Cutoff window for integrated autocorrelation time
            'tau_int': np.nan,               # Integrated autocorrelation time
            'se_tau_int': np.nan             # Standard error of integrated autocorrelation time
        }
        
        # Load configuration settings
        with open(os.path.join(folder_temp, 'config.json'), 'r') as json_file:
            settings = json.load(json_file)
            data_temp['T'] = settings.get('sim_paras', {}).get('T', 0.892)
            data_temp['L'] = settings.get('sys_paras', {}).get('L', int(10))
            L = data_temp['L']
            N = L ** 2

        # Process raw data
        with open(os.path.join(folder_temp, f'raw_data_T{data_temp["T"]:.2f}.txt'), 'r') as raw_data:
            # Skip header line and store last line separately
            lines = raw_data.readlines()
            data_lines = lines[1:-1]
            last_line = lines[-1]
            
            # Extract data from each line
            for count, line in enumerate(data_lines):
                parts = line.split('[') 
                values = parts[0].split()
                
                # Store all magnetization data if processing autocorrelations
                if para_data['proc_autocorre']:
                    data_temp['magnetization_all'].append(float(values[2]))
                
                # Extract data by measurement frequency
                if (count + 1) % para_data['meas_freq'] == 0:
                    # Extract scalar data
                    data_temp['num_traj'].append(int(values[0]))
                    data_temp['energy'].append(float(values[1]))
                    data_temp['magnetization'].append(float(values[2]))
                    data_temp['mx'].append(float(values[3]))
                    data_temp['my'].append(float(values[4]))
                    data_temp['delta_H'].append(float(values[6]))
                    
                    if para_data['proc_vor_den']:
                        data_temp['vor_den'].append(float(values[9]))
                        
                    if para_data['proc_heli_mod']:
                        data_temp['heli_mod'].append(float(values[10]))

                    # Compute running averages
                    data_temp['avg_mx_cur'].append(np.mean(data_temp['mx']))
                    data_temp['avg_my_cur'].append(np.mean(data_temp['my']))
                    data_temp['avg_exp_delta_H_cur'].append(np.mean(np.exp(-np.array(data_temp['delta_H']))))
            
            # Process last line for total values
            parts_last = last_line.split('[') 
            values_last = parts_last[0].split()
            data_temp['acc_rate_tot'] = float(values_last[5])
            
            # Process correlations if requested
            if para_data['proc_corre']:
                avg_gm_all = np.fromstring(parts_last[1].split(']')[0], sep = ',')
                var_gm_all = np.fromstring(parts_last[2].split(']')[0], sep = ',')
                
                avg_gm = avg_gm_all[:para_data['max_sep_t']]
                var_gm = var_gm_all[:para_data['max_sep_t']]
                
                se_gm = np.sqrt(var_gm) / L
                
                eff_m, se_eff_m = compute_eff_mass(avg_gm, var_gm, L)
                c, m, c_sd, m_sd = fit_corre_func(avg_gm, L)
                
                data_temp['avg_gm'] = avg_gm
                data_temp['se_gm'] = se_gm
                data_temp['eff_m'] = eff_m
                data_temp['se_eff_m'] = se_eff_m
                data_temp['corre_fit_c'] = c
                data_temp['sd_corre_fit_c'] = c_sd
                data_temp['corre_fit_m'] = m
                data_temp['sd_corre_fit_m'] = m_sd
                data_temp['corre_len'] = 1 / m
                data_temp['sd_corre_len'] = m_sd / (m * m) # Error propagation formula
                
                plot_correlations(data_temp, path = folder_temp)
        
        # Process autocorrelations if requested
        if para_data['proc_autocorre']:
            rho, se_rho = compute_autocorre_func(data_temp, para_data['max_sep_n'])

            b, tau, b_sd, tau_sd = fit_autocorre_func(rho, para_data['p0_tau'])
            
            data_temp['rho'] = rho
            data_temp['se_rho'] = se_rho
            data_temp['autocorre_fit_b'] = b
            data_temp['sd_autocorre_fit_b'] = b_sd 
            data_temp['autocorre_fit_tau'] = tau
            data_temp['sd_autocorre_fit_tau'] = tau_sd
            
            compute_int_autocorre_time(data_temp)
            
            plot_autocorrelations(data_temp, path = folder_temp)
        
        # Compute Binder cumulant if requested
        if para_data['proc_bdc']:
            mag = np.array(data_temp['magnetization'])
            m2 = mag * mag
            m4 = mag ** 4
            
            avg_m2 = np.mean(m2)
            sd_m2 = np.std(m2, ddof=1)
            
            avg_m4 = np.mean(m4)
            sd_m4 = np.std(m4, ddof=1)
            
            bdc = 1 - (avg_m4 / (3 * avg_m2 * avg_m2))
            sd_bdc = np.sqrt((2 * avg_m4 / (3 * avg_m2 ** 3) * sd_m2) ** 2 + (sd_m4 / (3 * avg_m2 ** 2) ) ** 2)
            se_bdc = sd_bdc / np.sqrt(len(mag))
            
            data_temp['bdc'] = bdc
            data_temp['se_bdc'] = se_bdc
            
        # Compute ensemble averages
        if para_data['proc_energy']:
            data_temp['avg_energy'] = np.mean(data_temp['energy'])
            data_temp['se_energy'] = np.std(data_temp['energy'], ddof = 1) / L
        
        if para_data['proc_vor_den']:
            data_temp['avg_vor_den'] = np.mean(data_temp['vor_den'])
            data_temp['se_vor_den'] = np.std(data_temp['vor_den']) / L
        if para_data['proc_heli_mod']:
            data_temp['avg_heli_mod'] = np.mean(data_temp['heli_mod'])
            data_temp['se_heli_mod'] = np.std(data_temp['heli_mod']) / L
            
        if para_data['proc_sus']: # Susceptibility
            sus_list = np.array(data_temp['magnetization']) * np.array(data_temp['magnetization']) * N
            data_temp['sus'] = np.mean(sus_list)
            data_temp['se_sus'] = np.std(sus_list, ddof = 1) / L
        
        # Plot raw data and processed results
        plot_raw_data_basic(data_temp, path = folder_temp)
        
        if para_data['proc_vor_den']:
            plot_raw_data_vor_den(data_temp, path = folder_temp)
        if para_data['proc_heli_mod']:
            plot_raw_data_heli_mod(data_temp, path = folder_temp)
        
        plot_ensemble_averages(data_temp, path = folder_temp)

        # Export total values to JSON file
        output_data = {
            'acc_rate': data_temp['acc_rate_tot'],
            'energy': data_temp['avg_energy'],
            'se_energy': data_temp['se_energy'],
            'sus': data_temp['sus'],
            'se_sus': data_temp['se_sus'],
            'vor_den': data_temp['avg_vor_den'],
            'se_vor_den': data_temp['se_vor_den'],
            'heli_mod': data_temp['avg_heli_mod'],
            'se_heli_mod': data_temp['se_heli_mod'],
            'bdc': data_temp['bdc'],
            'se_bdc': data_temp['se_bdc'],
            'corre_fit_c': data_temp['corre_fit_c'],
            'sd_corre_fit_c': data_temp['sd_corre_fit_c'],
            'corre_fit_m': data_temp['corre_fit_m'],
            'sd_corre_fit_m': data_temp['sd_corre_fit_m'],
            'corre_len': data_temp['corre_len'],
            'sd_corre_len': data_temp['sd_corre_len'],
            'autocorre_fit_b': data_temp['autocorre_fit_b'],
            'sd_autocorre_fit_b': data_temp['sd_autocorre_fit_b'],
            'autocorre_fit_tau': data_temp['autocorre_fit_tau'],
            'sd_autocorre_fit_tau': data_temp['sd_autocorre_fit_tau'],
            'cutoff_win': data_temp['cutoff_win'],
            'tau_int': data_temp['tau_int'],
            'se_tau_int': data_temp['se_tau_int']
        }

        output_file = os.path.join(folder_temp, f'values_T{data_temp["T"]:.2f}.json')
        with open(output_file, 'w') as outfile:
            json.dump(output_data, outfile, indent=4)

#===============================================================================
# Temperature Range Processing
#===============================================================================

def process_data_temp_range(para_data, data_folder, ref_data=None):
    """
    Process data from a batch of jobs generated under a range of temperatures.
    
    This function aggregates and processes data across a temperature range, computes various physical quantities, and generates plots for analysis.
    """
    print("=" * 100)
    print("Processing data under the range of temperature...")

    # Initialize dictionary to store data across temperature range
    data_range = {
        'temp': [],
        'L': np.nan,
        'meas_freq': para_data['meas_freq'],
        'num_traj': [],
        'num_bin': para_data['num_bin'],
        'energy': [],
        'se_energy': [],
        'sh': [],
        'se_sh': [],
        'sus': [],
        'se_sus': [],
        'vor_den': [],
        'se_vor_den': [],
        'heli_mod': [],
        'se_heli_mod': [],
        'bdc': [],
        'se_bdc': [],
        'corre_fit_c': [],
        'sd_corre_fit_c': [],
        'corre_fit_m': [],
        'sd_corre_fit_m': [],
        'corre_len': [],
        'sd_corre_len': [],
        'autocorre_fit_b': [],
        'sd_autocorre_fit_b': [],
        'autocorre_fit_tau': [],
        'sd_autocorre_fit_tau': [],
        'cutoff_win': [],
        'tau_int': [],
        'sd_tau_int': []
    }
        
    # Process each subfolder (each temperature point)
    for subfolder in os.listdir(data_folder):
        folder_temp = os.path.join(data_folder, subfolder)
        
        # Load configuration settings
        with open(os.path.join(folder_temp, 'config.json'), 'r') as json_file:
            settings = json.load(json_file)
            data_range['L'] = int(settings.get('sys_paras', {}).get('L'))
            num_traj = settings.get('sim_paras', {}).get('num_traj')
            temp = settings.get('sim_paras', {}).get('T')
            
            data_range['temp'].append(temp)
            data_range['num_traj'].append(num_traj)
            L = data_range['L']
            N = L ** 2

        # Load and process values for each temperature point
        with open(os.path.join(folder_temp, f'values_T{temp:.2f}.json'), 'r') as values_file:
            data_values = json.load(values_file)
            
            # Append data to the data_range lists
            for key in data_values:
                if key in data_range:
                    data_range[key].append(data_values[key])
        
        # Compute specific heat if requested
        if para_data['proc_sh']:
            with open(os.path.join(folder_temp, f'raw_data_T{temp:.2f}.txt'), 'r') as raw_data:
                energy_raw = []
                
                # Skip the header line
                lines = raw_data.readlines()
                data_lines = lines[1:-1]
                
                # Extract energy data by measurement frequency
                for count, line in enumerate(data_lines):
                    parts = line.split('[') 
                    values = parts[0].split()
                    
                    if (count + 1) % para_data['meas_freq'] == 0:
                        energy_raw.append(float(values[1]))
                
                # Compute specific heat
                sh = np.var(energy_raw, ddof = 1) * N / temp ** 2
                data_range['sh'].append(sh)
                
                # Use binning method to compute errors
                bin_size = int(len(energy_raw) / para_data['num_bin'])
                sh_bins = []
                
                for i in range(0, len(energy_raw), bin_size):
                    energies_bin = energy_raw[i:i + bin_size]
                    sh_bin = np.var(energies_bin, ddof = 1) * (L / temp) ** 2
                    sh_bins.append(sh_bin)
                
                se_sh = np.std(sh_bins, ddof = 1) / np.sqrt(len(sh_bins))
                data_range['se_sh'].append(se_sh)
        else:
            data_range['sh'].append(np.nan)
            data_range['se_sh'].append(np.nan)
    
    # Create output folder for temperature range data
    data_folder_range = os.path.join(data_folder, "data_range")
    os.makedirs(data_folder_range, exist_ok = True)
    
    # Write data to table
    save_as_table(data_range, data_folder_range)

    # Generate plots based on processed parameters
    if para_data['proc_energy']:
        plot_energy(data_range, data_folder_range, ref_data)
        
    if para_data['proc_sh']:
        plot_spec_heat(data_range, ref_data, path=data_folder_range)
        
    if para_data['proc_sus']:
        plot_susceptibility(data_range, ref_data, path=data_folder_range)

    if para_data['proc_vor_den']:
        plot_vor_den(data_range, path=data_folder_range)

    if para_data['proc_heli_mod']:
        plot_heli_mod(data_range, path=data_folder_range)
        
    if para_data['proc_bdc']:
        plot_bdc(data_range, path=data_folder_range)
        
    if para_data['proc_corre']:
        plot_corre_fit_c(data_range, path=data_folder_range)
        plot_corre_fit_m(data_range, path=data_folder_range)
        plot_corre_len(data_range, ref_data, path=data_folder_range)

    if para_data['proc_autocorre']:
        plot_autocorre_fit_b(data_range, path=data_folder_range)
        plot_autocorre_fit_tau(data_range, ref_data, path=data_folder_range)
        plot_cutoff_win(data_range, path=data_folder_range)
        plot_tau_int(data_range, path=data_folder_range)
    
    if para_data['proc_corre'] and para_data['proc_autocorre']:
        plot_critical_exponent(data_range, path=data_folder_range)
        
#===============================================================================
# Data Saving and Formatting
#===============================================================================

def save_as_table(data_range, data_folder):
    """
    Save processed data in a format convenient for creating tables in LaTeX.

    This function creates a txt file with LaTeX-formatted entries for various physical quantities.
    The data is organized in columns, with error values in parentheses.
    """
    
    # Define the output file path
    table_file = os.path.join(data_folder, f"data_table_L{data_range['L']}.txt")
    
    with open(table_file, mode='w', newline='', encoding='utf-8') as file:
        # Initialize CSV writer with '&' as delimiter for LaTeX table compatibility
        writer = csv.writer(file, delimiter='&', quoting=csv.QUOTE_MINIMAL)
        
        # Write header row with LaTeX-formatted column names
        writer.writerow([
            "$T$",                      # Temperature
            "$n_{\\text{traj}}$",       # Number of trajectories
            "$E$",                      # Energy
            "$C_v$",                    # Specific heat
            "$\\chi$",                  # Susceptibility
            "$10^{{3}} \\rho_{\\text{vor}}$",  # Vortex density (scaled by 10^3)
            "$\\Upsilon$",              # Helicity modulus
            "$U_4$",                    # Binder cumulant
            "$c_{\\text{cor}}$",        # Correlation fit parameter c
            "$m_{\\text{cor}}$",        # Correlation fit parameter m
            "$\\xi$",                   # Correlation length
            "$b_{\\text{atcor}}$",      # Autocorrelation fit parameter b
            "$\\tau_{\\text{atcor}}$",  # Autocorrelation fit parameter tau
            "$M_{\\text{win}}$",        # Cutoff window
            "$\\tau_{\\text{int}}$"     # Integrated autocorrelation time
        ])
        
        # Process and write data for each temperature point
        for i in range(len(data_range['temp'])):
            # Format temperature
            temp = f"{data_range['temp'][i]:.2f}" 
            
            # Format number of trajectories (in thousands)
            num_traj = f"{int(data_range['num_traj'][i] / 1000)}K"
            
            # Format other quantities with error values in parentheses
            # Use '-' for nan values
            energy = f"{data_range['energy'][i]:.4f}({int(data_range['se_energy'][i] * 10000)})" if not np.isnan(data_range['energy'][i]) and not np.isnan(data_range['se_energy'][i]) else '-'
            
            sh = f"{data_range['sh'][i]:.3f}({int(data_range['se_sh'][i] * 1000)})" if not np.isnan(data_range['sh'][i]) and not np.isnan(data_range['se_sh'][i]) else '-'

            sus = f"{data_range['sus'][i]:.2f}({int(data_range['se_sus'][i] * 100)})" if not np.isnan(data_range['sus'][i]) and not np.isnan(data_range['se_sus'][i]) else '-'
            
            vor_den = f"{data_range['vor_den'][i] * 1000:.4f}({int(data_range['se_vor_den'][i] * 10000000)})" if not np.isnan(data_range['vor_den'][i]) and not np.isnan(data_range['se_vor_den'][i]) else '-'
            
            heli_mod = f"{data_range['heli_mod'][i]:.4f}({int(data_range['se_heli_mod'][i] * 10000)})" if not np.isnan(data_range['heli_mod'][i]) and not np.isnan(data_range['se_heli_mod'][i]) else '-'
            
            bdc = f"{data_range['bdc'][i]:.4f}({int(data_range['se_bdc'][i] * 10000)})" if not np.isnan(data_range['bdc'][i]) and not np.isnan(data_range['se_bdc'][i]) else '-'
            
            corre_fit_c = f"{data_range['corre_fit_c'][i]:.2f}({int(data_range['sd_corre_fit_c'][i] * 100)})" if not np.isnan(data_range['corre_fit_c'][i]) and not np.isnan(data_range['sd_corre_fit_c'][i]) else '-'
            
            corre_fit_m = f"{data_range['corre_fit_m'][i]:.4f}({int(data_range['sd_corre_fit_m'][i] * 10000)})" if not np.isnan(data_range['corre_fit_m'][i]) and not np.isnan(data_range['sd_corre_fit_m'][i]) else '-'
            
            corre_len = f"{data_range['corre_len'][i]:.2f}({int(data_range['sd_corre_len'][i] * 100)})" if not np.isnan(data_range['corre_len'][i]) and not np.isnan(data_range['sd_corre_len'][i]) else '-'
            
            autocorre_fit_b = f"{data_range['autocorre_fit_b'][i]:.2f}({int(data_range['sd_autocorre_fit_b'][i] * 10000)})" if not np.isnan(data_range['autocorre_fit_b'][i]) and not np.isnan(data_range['sd_autocorre_fit_b'][i]) else '-'
            
            autocorre_fit_tau = f"{data_range['autocorre_fit_tau'][i]:.3f}({int(data_range['sd_autocorre_fit_tau'][i] * 10000)})" if not np.isnan(data_range['autocorre_fit_tau'][i]) and not np.isnan(data_range['sd_autocorre_fit_tau'][i]) else '-'
            
            cutoff_win = int(data_range['cutoff_win'][i]) if not np.isnan(data_range['cutoff_win'][i]) else '-'
            
            tau_int = f"{data_range['tau_int'][i]:.3f}({int(data_range['sd_tau_int'][i] * 1000)})" if not np.isnan(data_range['tau_int'][i]) and not np.isnan(data_range['sd_tau_int'][i]) else '-'
            
            # Write formatted row
            writer.writerow([
                f"${temp}$", 
                f"${num_traj}$", 
                f"${energy}$", 
                f"${sh}$", 
                f"${sus}$", 
                f"${vor_den}$", 
                f"${heli_mod}$", 
                f"${bdc}$", 
                f"${corre_fit_c}$", 
                f"${corre_fit_m}$", 
                f"${corre_len}$", 
                f"${autocorre_fit_b}$", 
                f"${autocorre_fit_tau}$", 
                f"${cutoff_win}$", 
                f"${tau_int}$ \\\\" 
            ])
