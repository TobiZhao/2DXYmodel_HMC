import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

from xymodel import *

# ================================================================================================
def load_ref_data(ref_data_file):
    ref_data = {
        'temp': [],
        'autocorre_fit_tau': [],
        'energy': [],
        'sh': [],
        'sus': [],
        'corre_len': [],
        'corre_len_av': []
    }

    with open(ref_data_file, 'r') as f:
        # skip the header line
        next(f)
        
        for line in f:
            values = line.split()
                        
            ref_data['temp'].append(float(values[0]))
            ref_data['autocorre_fit_tau'].append(float(values[1]))
            ref_data['energy'].append(-float(values[2]))
            ref_data['sh'].append(float(values[3]))
            ref_data['sus'].append(float(values[4]))
            ref_data['corre_len'].append(float(values[5]) if values[5] != '-' else float('nan'))
            ref_data['corre_len_av'].append(float(values[6]) if values[6] != '-' else float('nan'))
    return ref_data

def process_data_temp_point(para_data, data_folder):
    # requiring the data_folder containing jobs from the same batch
    
    for subfolder in os.listdir(data_folder):
        folder_temp = os.path.join(data_folder, subfolder)
        
        print('Processing folder:', folder_temp)
        
        data_temp = {
            'T': np.nan,
            'L': np.nan,
            'meas_freq': para_data['meas_freq'],
            'max_sep_t': para_data['max_sep_t'],
            'max_sep_n': para_data['max_sep_n'],
            'num_traj': [],
            'energy': [],
            'magnetization': [],
            'magnetization_all': [], # for computing autocorrelations
            'mx': [],
            'my': [],
            'delta_H': [],
            'vor_den': [],
            'heli_mod': [],
            'avg_mx_cur': [],
            'avg_my_cur': [],
            'avg_exp_delta_H_cur': [],
            'acc_rate_tot': np.nan,
            'avg_energy': np.nan,
            'se_energy': np.nan,
            'sus': np.nan,
            'se_sus': np.nan,
            'avg_vor_den': np.nan,
            'se_vor_den': np.nan,
            'avg_heli_mod': np.nan,
            'se_heli_mod': np.nan,
            'bdc': np.nan,
            'se_bdc': np.nan,
            'avg_gm': [],
            'se_gm': [],
            'eff_m': [],
            'se_eff_m': [],
            'corre_fit_c': np.nan,
            'sd_corre_fit_c': np.nan,
            'corre_fit_m': np.nan,
            'sd_corre_fit_m': np.nan,
            'corre_len': np.nan,
            'sd_corre_len': np.nan,
            'rho': [],
            'se_rho': [],
            'autocorre_fit_b': np.nan,
            'sd_autocorre_fit_b': np.nan,
            'autocorre_fit_tau': np.nan,
            'sd_autocorre_fit_tau': np.nan,
            'cutoff_win': np.nan,
            'tau_int': np.nan,
            'se_tau_int': np.nan
        }
        
        with open(os.path.join(folder_temp, 'config.json'), 'r') as json_file:
            settings = json.load(json_file)
            data_temp['T'] = settings.get('sim_paras', {}).get('T', 0.892)
            data_temp['L'] = settings.get('sys_paras', {}).get('L', int(10))
            L = data_temp['L']
            N = L ** 2

        with open(os.path.join(folder_temp, f'raw_data_T{data_temp["T"]:.2f}.txt'), 'r') as raw_data:
            # skip the header line, and store the last line separately
            lines = raw_data.readlines()
            data_lines = lines[1:-1]
            last_line = lines[-1]
            
            # extract the data
            for count, line in enumerate(data_lines):
                # split the line
                parts = line.split('[') 
                values = parts[0].split()
                
                # store all the magnetization data if processing autocorrelations
                if para_data['proc_autocorre']:
                    data_temp['magnetization_all'].append(float(values[2]))
                
                # extract data by the frequency of measurement
                if (count + 1) % para_data['meas_freq'] == 0:
                    # extract scalar data
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

                    # compute ensemble averages till current number of trajectory
                    data_temp['avg_mx_cur'].append(np.mean(data_temp['mx']))
                    data_temp['avg_my_cur'].append(np.mean(data_temp['my']))
                    data_temp['avg_exp_delta_H_cur'].append(np.mean(np.exp(-np.array(data_temp['delta_H']))))
            
            # save the total values from the last line
            parts_last = last_line.split('[') 
            values_last = parts_last[0].split()

            data_temp['acc_rate_tot'] = float(values_last[5])
            
            # process correlations
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
                data_temp['sd_corre_len'] = m_sd / (m * m) # from the error propagation formular, btw m * m is faster that m**2
                
                plot_correlations(data_temp, path = folder_temp)
        
        # process autocorrelations
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
        
        # compute the Binder cumulant
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
            
        # compute ensemble averages
        if para_data['proc_energy']:
            data_temp['avg_energy'] = np.mean(data_temp['energy'])
            data_temp['se_energy'] = np.std(data_temp['energy'], ddof = 1) / L
        
        if para_data['proc_vor_den']:
            data_temp['avg_vor_den'] = np.mean(data_temp['vor_den'])
            data_temp['se_vor_den'] = np.std(data_temp['vor_den']) / L
        if para_data['proc_heli_mod']:
            data_temp['avg_heli_mod'] = np.mean(data_temp['heli_mod'])
            data_temp['se_heli_mod'] = np.std(data_temp['heli_mod']) / L
            
        if para_data['proc_sus']: # susceptibility
            sus_list = np.array(data_temp['magnetization']) * np.array(data_temp['magnetization']) * N
            data_temp['sus'] = np.mean(sus_list)
            data_temp['se_sus'] = np.std(sus_list, ddof = 1) / L
        
        # plottings of raw data
        plot_raw_data_basic(data_temp, path = folder_temp)
        
        if para_data['proc_vor_den']:
            plot_raw_data_vor_den(data_temp, path = folder_temp)
        if para_data['proc_heli_mod']:
            plot_raw_data_heli_mod(data_temp, path = folder_temp)
        
        # plotting of average values of energy, and components of magnitization
        plot_ensemble_averages(data_temp, path = folder_temp)

        # export total values
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

#=======================================================================================================================================================

def process_data_temp_range(para_data, ref_data, data_folder):
    # process data from the same batch of jobs (generated under a range of temperatures, with other parameters unchanged)
    print("=" * 100)
    print("Processing data under the range of temperature...")

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
        
    for subfolder in os.listdir(data_folder):
        folder_temp = os.path.join(data_folder, subfolder)
        
        with open(os.path.join(folder_temp, 'config.json'), 'r') as json_file:
            settings = json.load(json_file)
            data_range['L'] = int(settings.get('sys_paras', {}).get('L'))
            num_traj = settings.get('sim_paras', {}).get('num_traj')
            temp = settings.get('sim_paras', {}).get('T')
            
            data_range['temp'].append(temp)
            data_range['num_traj'].append(num_traj)
            L = data_range['L']
            N = L ** 2

        with open(os.path.join(folder_temp, f'values_T{temp:.2f}.json'), 'r') as values_file:
            # load the values as a dictionary
            data_values = json.load(values_file)
            
            # append data to the data_range lists
            data_range['energy'].append(data_values['energy'])
            data_range['se_energy'].append(data_values['se_energy'])
            data_range['sus'].append(data_values['sus'])
            data_range['se_sus'].append(data_values['se_sus'])
            data_range['vor_den'].append(data_values['vor_den'])
            data_range['se_vor_den'].append(data_values['se_vor_den'])
            data_range['heli_mod'].append(data_values['heli_mod'])
            data_range['se_heli_mod'].append(data_values['se_heli_mod'])
            data_range['bdc'].append(data_values['bdc'])
            data_range['se_bdc'].append(data_values['se_bdc'])
            data_range['corre_fit_c'].append(data_values['corre_fit_c'])
            data_range['sd_corre_fit_c'].append(data_values['sd_corre_fit_c'])
            data_range['corre_fit_m'].append(data_values['corre_fit_m'])
            data_range['sd_corre_fit_m'].append(data_values['sd_corre_fit_m'])
            data_range['corre_len'].append(data_values['corre_len'])
            data_range['sd_corre_len'].append(data_values['sd_corre_len'])
            data_range['autocorre_fit_b'].append(data_values['autocorre_fit_b'])
            data_range['sd_autocorre_fit_b'].append(data_values['sd_autocorre_fit_b'])
            data_range['autocorre_fit_tau'].append(data_values['autocorre_fit_tau'])
            data_range['sd_autocorre_fit_tau'].append(data_values['sd_autocorre_fit_tau'])
            data_range['cutoff_win'].append(data_values['cutoff_win'])
            data_range['tau_int'].append(data_values['tau_int'])
            data_range['sd_tau_int'].append(data_values['se_tau_int'])
        
        if para_data['proc_sh']:
            with open(os.path.join(folder_temp, f'raw_data_T{temp:.2f}.txt'), 'r') as raw_data:
                #print('Specific heat:', f'T = {temp:.2f}', ' num_traj =', num_traj, ' meas_freq =', para_data['meas_freq'], ' num_measured_samples =', int(num_traj / para_data['meas_freq']), ' num_bin =', para_data['num_bin'], ' bin_size =', int(num_traj / para_data['meas_freq'] / para_data['num_bin']))
                
                energy_raw = []
                
                # skip the header line
                lines = raw_data.readlines()
                data_lines = lines[1:-1]
                
                # extract the data
                for count, line in enumerate(data_lines):
                    # split the line
                    parts = line.split('[') 
                    values = parts[0].split()
                    
                    # extract energy data by the frequency of measurement
                    if (count + 1) % para_data['meas_freq'] == 0:
                        energy_raw.append(float(values[1]))
                
                # compute the specific heat
                sh = np.var(energy_raw, ddof = 1) * N / temp ** 2
                
                data_range['sh'].append(sh)
                
                bin_size = int(len(energy_raw) / para_data['num_bin'])
                bin_count = 0
                sh_bins = []
                
                for i in range(0, len(energy_raw), bin_size): # use the binning method to compute errors by increasing the number of bins until the variances converge
                    bin_count += 1
                    energies_bin = energy_raw[i:i + bin_size]
                    
                    sh_bin = np.var(energies_bin, ddof = 1) * (L / temp) * (L / temp)
                    
                    sh_bins.append(sh_bin)
                
                se_sh = np.std(sh_bins, ddof = 1) / np.sqrt(len(sh_bins))
                
                data_range['se_sh'].append(se_sh)
        else:
            data_range['sh'].append(np.nan)
            data_range['se_sh'].append(np.nan)
    
    data_folder_range = os.path.join(data_folder, "data_range")
    os.makedirs(data_folder_range, exist_ok = True)
    
    # data write-out to a table
    save_as_table(data_range, data_folder_range)

    # plottings
    if para_data['proc_energy']:
        plot_energy(data_range, ref_data, path=data_folder_range)
        
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
    
def save_as_table(data_range, data_folder):
    # save the processed data in form convenient for making tables in LaTeX
    table_file = os.path.join(data_folder, f"data_table_L{data_range['L']}.txt")
    with open(table_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='&', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            "$T$", 
            "$n_{\\text{traj}}$", 
            "$E$", 
            "$C_v$", 
            "$\\chi$", 
            "$10^{{3}} \\rho_{\\text{vor}}$", 
            "$\\Upsilon$",
            "$U_4$",
            "$c_{\\text{cor}}$",
            "$m_{\\text{cor}}$",
            "$\\xi$", 
            "$b_{\\text{atcor}}$",
            "$\\tau_{\\text{atcor}}$",
            "$M_{\\text{win}}$",
            "$\\tau_{\\text{int}}$"
        ])
        
        for i in range(len(data_range['temp'])):
            temp = f"{data_range['temp'][i]:.2f}" 
            
            num_traj = f"{int(data_range['num_traj'][i] / 1000)}K"
            
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
