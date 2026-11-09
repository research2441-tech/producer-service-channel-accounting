# Exact channel accounting of producer-service empowerment and energy-related CO2

Code and data for the manuscript *Exact Channel Accounting of Producer-Service Empowerment in
Manufacturing and Energy-Related Emission Control across Chinese Provinces*.

## What is here (one flat folder)

| File | Role |
|---|---|
| `panel_raw.csv` | Assembled panel: 30 provinces x 2011-2021 (330 rows), all real published statistics |
| `contiguity_edges.csv` | First-order provincial contiguity (spatial weights) |
| `validation_va.csv` | NBS sectoral value added, 2019-2022 (external validation only) |
| `build_panel.py` | Builds `panel_raw.csv` from the raw sources (see below) |
| `estimators.py` | 2SLS/LIML, cluster-robust covariance, first-stage and effective F, Hansen J, Anderson-Rubin inversion, WRE and WCR wild bootstraps, sensitivity (robustness values) |
| `analysis.py` | Six-term identity, channel system, merged channels, ablations, cross-fitted ridge/boosting, SLX, heterogeneity, lags, robustness, permutation, leave-one-out, IV audit, validation |
| `placebo.py` | Placebo size experiment with synthetic null producer-service paths |
| `figures.py` | Grayscale figures 1-9 |
| `results.json`, `placebo.json` | Saved outputs of the run reported in the paper |
| `run_all.sh` | One-command reproduction |

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh
```

The full run takes about two minutes for `analysis.py` and roughly 15 minutes for the placebo
experiment on two CPU cores. Seeds are fixed, so the numbers match the manuscript.

## Rebuilding the panel from raw sources

`build_panel.py <data_root> <out_dir>` expects, under `data_root`:

* `statistical-modeling_CUC_2026_PYH/` with CEADs provincial energy inventories (`data_raw/energy/*.xlsx`),
  the processed NBS/CEADs/patent/digital-finance panel (`data_processed/panel_data.csv`),
  capital coordinates, contiguity and region groups;
* `China-Provincial-Panel-Dataset-on-Economic-Development-Energy-and-CO2-Emissions-2000-2022/data.xlsx`
  (technology-market turnover, trade);
* `pe/data/raw/nbs_exports/` with NBS provincial annual exports (validation series).

Energy is converted to standard coal equivalent with the China Energy Statistical Yearbook 2022
conversion factors (Appendix IV). The 2022 technology-market series in the source panel is internally
inconsistent, so the panel ends in 2021.

## Key definitions

* Identity: `ln E = ln Y + ln s + ln(M/(sY)) + ln(Q/M) + ln(C/Q) + ln(E/C)`
* Producer-service index: standardised mean of `ln(technology turnover / GDP)` and `ln(digital finance index)`,
  rescaled to unit standard deviation.
* Main estimator: two-way fixed effects with fiscal share, log population and year-specific coefficients on
  2011 log real output per capita and 2011 secondary share; inference clustered by province.

## Data sources

CEADs (Carbon Emission Accounts and Datasets) provincial energy inventories and apparent CO2 accounts;
National Bureau of Statistics of China; Peking University Digital Financial Inclusion Index.
Respect the terms of each original source when redistributing.
