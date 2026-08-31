"""Numeric constants for ExtremeWeatherBench.

This module is a leaf: it must not import other package modules so that
Numba kernels in ``_cape`` can bind these names as local globals via
``from extremeweatherbench.constants import ...``.
"""

# Thermodynamic constants
KAPPA = 2.0 / 7.0  # Poisson constant R/Cp for dry air (dimensionless)
GRAVITY = 9.80665  # Gravitational acceleration (m/s^2)
# Rounded g historically used to convert geopotential thickness to meters.
GRAVITY_ROUNDED = 9.81
Rd = 287.058  # Gas constant for dry air (J/kg/K)
Cp = 1005.7  # Specific heat at constant pressure for dry air (J/kg/K)

# Water vapor constants.
# EPSILON is the unrounded Mw(H2O)/Mw(dry air) ratio for the package.
EPSILON = 0.6219569100577033
VIRTUAL_TEMP_COEFF = 0.61  # Virtual temperature mixing-ratio coefficient

# Latent heat constants
L_V_0 = 2.501e6  # Latent heat of vaporization at 0°C (J/kg)
L_V_TEMP_COEFF = 2370.0  # Temperature dependence of latent heat (J/kg/K)

# Reference values
P_REF = 1000.0  # Reference pressure for potential temperature (hPa)
KELVIN_TO_CELSIUS = 273.15  # Kelvin minus Celsius offset (K)
EARTH_RADIUS_KM = 6371.0  # Mean Earth radius used in haversine (km)

# Bolton (1980) formula constants for saturation vapor pressure
# e_s = E0 * exp(A * T_c / (T_c + B)) with T_c in Celsius
E0_BOLTON = 6.112  # Reference vapor pressure (hPa)
A_BOLTON = 17.67  # Empirical constant (dimensionless)
B_BOLTON = 243.5  # Empirical constant (°C)

# Bolton (1980) LCL formula constants
LCL_OFFSET = 56.0  # Empirical constant for LCL calculation (K)
LCL_DENOM = 800.0  # Empirical constant for LCL calculation (K)

# Numerical integration parameters
MOIST_ASCENT_STEPS = 50  # Steps for moist adiabat integration from the LCL
# Steps per level gap when marching the adiabat up a profile. Marching
# never re-integrates the column already covered, so fewer steps per gap
# reach a smaller truncation error than restarting from the LCL with
# MOIST_ASCENT_STEPS. Against a converged integration of the same ODE
# over 100 ERA5 profiles, 16 substeps is 31 J/kg MAE vs 53 J/kg for
# restarting; 8 substeps was worse than restarting.
MOIST_ASCENT_SUBSTEPS = 16

# Mixing-ratio guard: cap vapor pressure at this fraction of total pressure.
MAX_VAPOR_PRESSURE_FRACTION = 0.9999

# Minimum environment virtual temperature (K) to avoid divide-by-zero
# in buoyancy; well below any realistic atmospheric temperature.
MIN_TV = 100.0

# Default radius for sample data extraction (degrees)
RADIUS_DEG = 2.0

# Nanoseconds in one hour, for datetime64[ns] gaps.
NS_PER_HOUR = 3_600_000_000_000
