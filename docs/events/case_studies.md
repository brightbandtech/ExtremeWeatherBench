# Case Studies

EWB includes 337 curated cases spanning five extreme weather categories, covering
events worldwide from 2020 to 2024. Each category page describes how cases were
selected, the bounding box methodology, and the source for each individual event.

Cases are used as the atomic unit of evaluation: every metric is computed over a
single case's spatial domain and time window, then aggregated across the set.

## Active categories

| Category | Cases | Description |
|---|---|---|
| [Atmospheric Rivers](atmospheric_rivers.md) | 56 | IVT-corridor events from offshore origin to landfall, covering western North America, Europe, the Middle East, Australia, and Antarctica |
| [Severe Convection](severe_convection.md) | 115 | Multi-hazard convective outbreaks (tornadoes, hail, and damaging winds) across the US and globally |
| [Freezes](cold_snaps.md) | 14 | Large-scale cold-air outbreaks validated against ERA5 15th-percentile 2 m temperature thresholds |
| [Heat Waves](heat_waves.md) | 46 | Land-based heat waves validated against ERA5 85th-percentile 2 m temperature thresholds |
| [Tropical Cyclones](tropical_cyclones.md) | 106 | Landfalling storms from all ocean basins, tracks from IBTrACS genesis to dissipation |

## Planned categories

Additional event types have cases identified but metrics and benchmarks have not been defined.
See [Planned Event Types](planned_events.md) for the full list, which includes
derechos, flooding, hailstorms, tornado outbreaks, TC tornado outbreaks, and
winter weather.

## Selection criteria

All cases target the 2020–2024 period. The goal is at least 30 cases per category
for statistical robustness; freeze events are the exception at 14 cases due to the
rarity of qualifying outbreaks. Events are global where data permits.
