# XYmodel/__init__.py

# Import from core module
from .core.XYmodel_HMC import XYSystem

# Import from observables module
from .observables.helicity_modulus import compute_heli_mod
from .observables.vortex_func import compute_vor

# Import from analysis module
from .analysis.correlations import (
    compute_zmo, compute_corre_func, compute_eff_mass,
    corre_func_fit_model, fit_corre_func, proc_corre
)
from .analysis.autocorrelations import (
    compute_autocorre_func, autocorre_func_fit_model,
    fit_autocorre_func, compute_int_autocorre_time
)

# Import from utils module
from .utils.logging import setup_logging, get_logger

from .utils.plottings import (
    plot_spin_config, plot_raw_data_basic, plot_raw_data_vor_den,
    plot_raw_data_heli_mod, plot_ensemble_averages, plot_correlations,
    plot_autocorrelations, plot_energy, plot_spec_heat, plot_susceptibility, 
    plot_vor_den, plot_heli_mod, plot_bdc, plot_corre_fit_c, plot_corre_fit_m,
    plot_corre_len, plot_autocorre_fit_b, plot_autocorre_fit_tau,
    plot_cutoff_win, plot_tau_int, plot_critical_exponent
)
from .utils.proc_data_func import (
    load_ref_data, process_data_temp_point, 
    process_data_temp_range, save_as_table
)

# Import parameters
from .parameters import (sys_paras, sim_paras, calibration_paras)

__version__ = "1.0"
