#===============================================================================
# Execution of Simulation
#===============================================================================

# Execution of the simulation
import argparse
import os
import time
import json

from xymodel import *

def parse_args_main():
    """
    Parse command-line arguments for customizing simulation parameters.
    """
    parser = argparse.ArgumentParser(description="Run simulation with custom parameters.")

    # System parameters
    parser.add_argument("--seed", type=bool, default=sys_paras['seed'], help="Seed for random number generator")
    parser.add_argument("--L", type=int, default=sys_paras['L'], help="Lattice size")
    parser.add_argument("--a", type=float, default=sys_paras['a'], help="Lattice spacing")
    parser.add_argument("--write_times", type=int, default=sys_paras['write_times'], help="Number of times to write out data")

    # Simulation parameters
    parser.add_argument("--lfl", type=int, default=sim_paras['lfl'], help="Number of Leapfrog steps for each trajectory")
    parser.add_argument("--FA", action='store_true', default=sim_paras['FA'], help="Activate Fourier acceleration")
    parser.add_argument("--m_FA", type=float, default=sim_paras['m_FA'], help="Number of trajectories during sampling")
    parser.add_argument("--num_traj", type=int, default=sim_paras['num_traj'], help="Number of trajectories during sampling")
    parser.add_argument("--no_lf_calib", action='store_false', help="Deactivate calibration of Leapfrog parameters")
    parser.add_argument("--log_freq", type=int, default=sim_paras['log_freq'], help="Frequency of displaying progress during equilibration and sampling stages")
    parser.add_argument("--max_sep_t", type=int, default=sim_paras['max_sep_t'], help="Maximum space separation when calculating correlation length")

    # Calibration parameters
    parser.add_argument("--num_step_calib", type=int, default=calibration_paras['num_step_calib'], help="Number of trajectories for each calibration iteration")
    parser.add_argument("--num_calib", type=int, default=calibration_paras['num_calib'], help="Maximum number of calibration iterations")

    # Temperature parameter
    parser.add_argument("--T", type=float, default=sim_paras['T'], help="Simulation temperature")

    return parser.parse_args()

def update_paras():
    """
    Update simulation parameters based on command-line arguments.
    """
    args = parse_args_main()

    # Create the output folder
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    if args.FA:
        folder_temp = os.path.join(dir_path, "output", f"L{args.L}_T{args.T:.2f}_nt{args.num_traj}_FA")
    else:
        folder_temp = os.path.join(dir_path, "output", f"L{args.L}_T{args.T:.2f}_nt{args.num_traj}_noFA")
        
    os.makedirs(folder_temp, exist_ok=True)
    
    # Update the simulation parameters with command-line arguments
    sys_paras.update({
        'seed': args.seed,
        'L': args.L,
        'a': args.a,
        'write_times': args.write_times
    })
    
    sim_paras.update({
        'T': args.T,
        'FA': args.FA,
        'm_FA': args.m_FA,
        'lfl': args.lfl,
        'num_traj': args.num_traj,
        'lf_calib': args.no_lf_calib,
        'log_freq': args.log_freq,
        'max_sep_t': args.max_sep_t,
        'folder_temp': folder_temp
    })
    
    calibration_paras.update({
        'num_step_calib': args.num_step_calib,
        'num_calib': args.num_calib
    })

def save_parameters(parameters, path):
    """
    Save parameters to a JSON file.
    """
    with open(path, "w") as file:
        json.dump(parameters, file, indent=4)

def run_simulation_temp(T=0.892, logger=None):
    """
    Run the simulation at a specific temperature.
    """
    start_time_sim = time.time()
    
    # Initialize and run the simulation
    xy_system = XYSystem(**sys_paras, logger=logger)
    spin_config = xy_system.run_simulation(**sim_paras)  # Obtain the latest configuration for plotting
    
    end_time_sim = time.time()
    sim_time = end_time_sim - start_time_sim
    
    # Plot the final spin configuration
    plot_spin_config(xy_system.data, spin_config, sim_paras['folder_temp'])

    logger.info(f"Simulation Completed (L^2 = {xy_system.data['L']}^2, T = {T:.2f})\nTotal Simulation Time = {sim_time:.2f} seconds")
    
    # Save the parameters in a JSON file
    para_file_path = os.path.join(sim_paras['folder_temp'], "config.json")
    
    all_parameters = {
        "sys_paras": sys_paras,
        "sim_paras": sim_paras,
        "calibration_paras": calibration_paras
    }
    save_parameters(all_parameters, para_file_path)

def main():
    """
    Main function to run the simulation.
    """
    update_paras()
    
    log_file = os.path.join(sim_paras['folder_temp'], f"log_T{sim_paras['T']:.2f}.log")
    logger = setup_logging(log_file)
    
    logger.info(100 * "=")
    logger.info("HMC Simulation of 2D XY model (v2.2.0)")
    logger.info(100 * "=")
    logger.info("Simulation Started")
    logger.info(100 * "=")

    try:
        run_simulation_temp(sim_paras['T'], logger)
    except Exception as e:
        logger.exception(f"Error: {str(e)}")
    logger.info(100 * "=")

if __name__ == '__main__':
    main()
