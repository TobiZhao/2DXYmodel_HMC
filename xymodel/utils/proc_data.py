import argparse

from xymodel import *

# ================================================================================================
# Command line settings

parser = argparse.ArgumentParser(description = "Process simulation data with custom parameters.")

parser.add_argument("--L_ref", type = int, default = 64, help = "length of lattices denoting reference data, only affecting data processing under the temperature range")
parser.add_argument("--meas_freq", type = int, default = 100, help = "frequency of measurements")
parser.add_argument("--max_sep_t", type = int, default = 4, help = "maximum of space separation t when computing correlation function")
parser.add_argument("--max_sep_n", type = int, default = 100, help = "maximum of Monte Carlo time separation n when computing autocorrelation function")
parser.add_argument("--num_bin", type = int, default = 20, help = "number of bins in binning method")
parser.add_argument("--p0_tau", type = float, default = 500.0, help = "initial estimation of parameter tau in the fit of autocorrelation function")

parser.add_argument("--no_energy", action='store_false', help="deactivating the processing of energy data")
parser.add_argument("--no_sh", action='store_false', help="deactivating the processing of specific heat data")
parser.add_argument("--no_sus", action='store_false', help="deactivating the processing of susceptibility data")
parser.add_argument("--no_vor_den", action='store_false', help="deactivating the processing of vortex density data")
parser.add_argument("--no_heli_mod", action='store_false', help="deactivating the processing of helicity modulus data")
parser.add_argument("--no_bdc", action='store_false', help="deactivating the processing of Binder cumulant data")
parser.add_argument("--no_corre", action='store_false', help="deactivating the processing of correlation data")
parser.add_argument("--no_autocorre", action='store_false', help="deactivating the processing of autocorrelation data")

args = parser.parse_args()

# ===================================================================================================================================================================================
# Initialization
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# set folder of data to process

data_folder = r"Z:\Desktop\Research\Fourier_Accelerated_Lattice_Field_Theory\codes\XYmodel_HMC\output\output_server\L256\L256_T0.90_1.07_0.01_nt100k_cor\output"

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

para_data = {
    'L_ref': args.L_ref,
    'meas_freq': args.meas_freq,
    'max_sep_t': args.max_sep_t,
    'max_sep_n': args.max_sep_n,
    'num_bin': args.num_bin,
    'p0_tau': args.p0_tau,
    'proc_energy': args.no_energy,
    'proc_sh': args.no_sh,
    'proc_sus': args.no_sus,
    'proc_vor_den': args.no_vor_den,
    'proc_heli_mod': args.no_heli_mod,
    'proc_bdc': args.no_bdc,
    'proc_corre': args.no_corre,
    'proc_autocorre': args.no_autocorre
}

# ===================================================================================================================================================================================
# Data proccessing
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# for each single temperature

process_data_temp_point(para_data, data_folder)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# for the range of temperatures

ref_data_file = f"Z:\Desktop\Research\Fourier_Accelerated_Lattice_Field_Theory\codes\XYmodel_HMC\data_Gupta_1992\L{para_data['L_ref']}.dat"

ref_data = load_ref_data(ref_data_file)

process_data_temp_range(para_data, ref_data, data_folder)
