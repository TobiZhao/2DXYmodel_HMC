'''XYmodel/__init__.py'''

# Core simulation module
from .core.XYmodel_HMC import XYSystem

# Observable calculation modules
from .observables.helicity_modulus import compute_heli_mod
from .observables.vortex import compute_vor

# Analysis modules
from .analysis.correlations import (
    compute_zmo,           # Zero momentum operator
    compute_corre_func,    # Correlation function
    compute_eff_mass,      # Effective mass
    corre_func_fit_model,  # Correlation function fitting model
    fit_corre_func,        # Fit correlation function
    proc_corre             # Process correlation data
)
from .analysis.autocorrelations import (
    compute_autocorre_func,      # Autocorrelation function
    autocorre_func_fit_model,    # Autocorrelation function fitting model
    fit_autocorre_func,          # Fit autocorrelation function
    compute_int_autocorre_time   # Integrated autocorrelation time
)

# Utility modules
from .utils.logging import setup_logging, get_logger
from .utils.visualization import (
    # Basic plotting functions
    plot_spin_config, plot_raw_data_basic, plot_raw_data_vor_den,
    plot_raw_data_heli_mod, plot_ensemble_averages,
    
    # Correlation and autocorrelation plots
    plot_correlations, plot_autocorrelations,
    
    # Thermodynamic observables
    plot_energy, plot_spec_heat, plot_susceptibility,
    
    # Topological observables
    plot_vor_den, plot_heli_mod, plot_bdc,
    
    # Fitting and analysis plots
    plot_corre_fit_c, plot_corre_fit_m, plot_corre_len,
    plot_autocorre_fit_b, plot_autocorre_fit_tau,
    plot_cutoff_win, plot_tau_int, plot_critical_exponent
)
from .utils.data_processing import (
    load_ref_data,             # Load reference data
    process_data_temp_point,   # Process data for a single temperature
    process_data_temp_range,   # Process data for a temperature range
    save_as_table              # Save processed data as a table
)

# Simulation parameters
from .parameters import (sys_paras, sim_paras, calibration_paras)

# Package version
__version__ = "2.1.0"
