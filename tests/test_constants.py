"""Tests for shared numeric constants."""

from extremeweatherbench import _cape, calc, constants


def test_cape_reexports_match_constants():
    """Numba kernels bind the same floats defined in constants.py."""
    assert _cape.EPSILON is constants.EPSILON
    assert _cape.GRAVITY is constants.GRAVITY
    assert _cape.KAPPA is constants.KAPPA
    assert _cape.KELVIN_TO_CELSIUS is constants.KELVIN_TO_CELSIUS
    assert _cape.Rd is constants.Rd
    assert _cape.E0_BOLTON is constants.E0_BOLTON
    assert _cape.MIN_TV is constants.MIN_TV


def test_calc_aliases_match_constants():
    """Public calc aliases stay equal to the canonical constants."""
    assert calc.epsilon is constants.EPSILON
    assert calc.g0 is constants.GRAVITY
    assert calc.sat_press_0c is constants.E0_BOLTON
    assert calc._ns_per_hour is constants.NS_PER_HOUR


def test_epsilon_value():
    """EPSILON is the unrounded Mw(H2O)/Mw(dry air) ratio."""
    assert constants.EPSILON == 0.6219569100577033
