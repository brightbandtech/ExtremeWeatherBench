# A Note on CAPE Accuracy

CAPE and CIN values produced by EWB changed slightly as part of an accuracy and
performance pass on the parcel-ascent code in `extremeweatherbench._cape`. This
page records what changed, how large the shift is, and why it was accepted.

**This is a deliberate accuracy improvement, not a regression.** The new values
are measurably closer to an exact solution of the same governing equations, and
closer to the MetPy reference implementation, than the values they replace.

## What changed

Two independent changes moved the numbers.

**1. Moist-adiabat integration scheme.** Computing the temperature of a parcel
lifted along a moist adiabat requires integrating

$$\frac{dT}{d\ln p} = \frac{R_d T + L_v w_s}{c_p + \frac{L_v^2 w_s \epsilon}{R_d T^2}}$$

down the sounding. The previous implementation restarted this integration from
the lifting condensation level for every pressure level in the profile, which
made the cost quadratic in the number of levels and left a comparatively coarse
effective step size at the top of the sounding. The integration now *marches*
level by level, carrying the parcel temperature forward from the level below and
taking a fixed 16 substeps across each gap (`_cape.MOIST_ASCENT_SUBSTEPS`).

**2. The molecular-weight ratio $\epsilon$.** The package previously carried two
values of the ratio of the molecular weights of water vapor and dry air: the
rounded `0.622` inside `_cape.py`, and the unrounded `0.6219569100577033` in
`calc.py` and the severe-convection module. These are now a single constant,
`_cape.EPSILON`, set to the unrounded value and imported everywhere else. It is
defined in `_cape.py` because that module imports nothing else from the package,
so any module can take it without an import cycle.

## Size of the shift

Measured over the 100 ERA5 profiles stored in `tests/data/era5_reference.npz`,
comparing the current code against the code as it stood before this work:

| Quantity | Mean absolute change | Max absolute change |
| --- | --- | --- |
| CAPE | 22.68 J/kg | 48.71 J/kg |
| CIN | 0.12 J/kg | 5.25 J/kg |

Mean CAPE across those profiles is 874.6 J/kg, so the typical shift is about
2.6% of the signal. Essentially all of it comes from the integration scheme; the
$\epsilon$ unification on its own accounts for a mean change of 0.85 J/kg and a
maximum of 1.47 J/kg.

## Why the new values are better

The integration scheme was checked against a converged solution of the same ODE,
obtained by running the same marching integrator with 400 substeps per level
instead of 16. That converged run is the exact answer this discretization is
approximating, so the distance from it is the truncation error of the scheme:

| Scheme | Mean absolute error vs converged solve | Max |
| --- | --- | --- |
| Previous (restart from LCL) | 53.13 J/kg | 101.47 J/kg |
| Current (march, 16 substeps) | 31.30 J/kg | 54.23 J/kg |

Truncation error is reduced by a factor of 1.70 while the parcel ascent also got
faster, because marching is linear rather than quadratic in the level count.

Agreement with the independent MetPy reference values stored alongside the
profiles improved in the same direction:

| Version | CAPE MAE vs MetPy |
| --- | --- |
| Before this work | 106.52 J/kg |
| Current | 101.99 J/kg |
| Converged solve (floor) | 100.42 J/kg |

The current code sits about 1.6 J/kg from the converged solve, so nearly all of
the remaining ~100 J/kg spread against MetPy is a difference in model
formulation rather than in numerics, and further refining the step count would
not close it.

## The reference dataset was regenerated

`data_prep/generate_cape_reference_data.py` derived dewpoint from specific
humidity with its own hardcoded `0.622`, so it now uses the shared
`_cape.EPSILON` as well. Because dewpoint is an *input* to the stored profiles,
`tests/data/era5_reference.npz` was regenerated to match. The reference CAPE and
CIN values in that file still come from `metpy.calc.mixed_layer_cape_cin`, so it
remains an independent oracle and the reference tests are not circular.

Regenerating moved the stored profiles slightly: dewpoint by 0.0007 K on average
and the MetPy reference CAPE by 0.49 J/kg, about 0.035%.

It also surfaced that the stored file was stale in one respect. Its CIN values
were all positive, whereas current MetPy returns CIN as negative energy, which
is both the conventional sign and the sign the script's own hand-written
pathological expectations already used. The regenerated CIN values are therefore
negated relative to the old ones. No test asserted on reference CIN, so nothing
depended on the old sign.

## What this means for you

If you have stored CAPE-based results from an earlier version of EWB, expect
differences of the order of tens of J/kg per profile when you recompute them.
Scores derived from CAPE will move correspondingly. The reference tests in
`tests/test_cape.py` continue to hold their published tolerances: a per-profile
bound of 10% or 50 J/kg, whichever is larger, and a mean absolute error against
MetPy below 150 J/kg.
