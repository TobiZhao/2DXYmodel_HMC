#===============================================================================
# Execution of Data Processing
#===============================================================================

import argparse

from xymodel import *

def parse_arg_data_proc():
    """
    Parse command-line arguments for setting data processing parameters.
    
    Returns:
        Parsed argument object
    """
    parser = argparse.ArgumentParser(description="Process simulation data with custom parameters.")
   
    # Set various command-line arguments
    parser.add_argument("--L_ref", type=int, default=64, help="Length of lattices denoting reference data")
    parser.add_argument("--meas_freq", type=int, default=100, help="Frequency of measurements")
    parser.add_argument("--max_sep_t", type=int, default=4, help="Maximum space separation t for correlation function")
    parser.add_argument("--max_sep_n", type=int, default=100, help="Maximum Monte Carlo time separation n for autocorrelation function")
    parser.add_argument("--num_bin", type=int, default=20, help="Number of bins in binning method")
    parser.add_argument("--p0_tau", type=float, default=500.0, help="Initial estimation of parameter tau for autocorrelation function fit")
    parser.add_argument("--use_ref_data", type=bool, action='store_true', help="Use reference data as comparation")
    
    # Set data processing options
    processing_options = [
        "energy", "sh", "sus", "vor_den", "heli_mod", "bdc", "corre", "autocorre"
    ]
    for option in processing_options:
        parser.add_argument(f"--no_{option}", action='store_false', help=f"Deactivate processing of {option} data")
    
    return parser.parse_args()

def gen_para_dict(args):
    """
    Generate parameter dictionary.
    
    Args:
        args: Parsed argument object
    
    Returns:
        Dictionary of parameters controlling data processing
    """
    param_dict = vars(args)
    for key in list(param_dict.keys()):  # Use list() to avoid modifying dict during iteration
        if key.startswith("no_"):
            new_key = f"proc_{key[3:]}"
            param_dict[new_key] = param_dict.pop(key)
    return param_dict

def process_data(use_ref_data=False):
    """
    Execute data processing.
    """
    args = parse_arg_data_proc()
    para_data = gen_para_dict(args)
   
    # Set the data folder path
    data_folder = r"Z:\Desktop\Research\FALFT\codes\XYmodel_HMC\output\output_test"
    
    # Process data for each single temperature point
    process_data_temp_point(para_data, data_folder)
    
    # Process data for the range of temperatures
    if use_ref_data:
        ref_data_file = f"Z:\Desktop\Research\Fourier_Accelerated_Lattice_Field_Theory\codes\XYmodel_HMC\data_Gupta_1992\L{para_data['L_ref']}.dat"
        ref_data = load_ref_data(ref_data_file)
        process_data_temp_range(para_data, ref_data, data_folder)
    else:
        process_data_temp_range(para_data, data_folder)

if __name__ == "__main__":
    process_data()
