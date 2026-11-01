# Aviation Delay Prediction: Temporal Generalisation, Market-Specific Training and Calibration

Reproducibility repository for an empirical study of US domestic flight-delay prediction using Bureau of Transportation Statistics (BTS) On-Time Performance records from **January 2022 to December 2025**.

The study investigates three questions:

1. **Temporal generalisation:** How stable are flight-delay prediction models when evaluated on later years and months?
2. **Market-specific training:** Does including historical observations from a target airline or origin airport improve predictions for that same market?
3. **Probability calibration:** How sensitive is out-of-time probability accuracy to the choice of historical calibration window?

The analysis compares **logistic regression** and **XGBoost**, with attention to AUROC, average precision (AUPRC), Brier score, calibration and paired uncertainty estimates. The target outcome is an arrival delay of **at least 15 minutes** among flights with an eligible observed arrival outcome.

> **Reproducibility status:** This repository documents the research workflow and archived aggregate outputs. A complete fresh run from downloaded BTS files should be performed and checked before describing the repository as end-to-end reproduced. See [Reproduction checklist](#reproduction-checklist).

## Data source and coverage

- **Provider:** United States Department of Transportation, Bureau of Transportation Statistics (BTS).
- **Dataset:** Airline On-Time Performance / Reporting Carrier On-Time Performance.
- **Portal:** https://www.transtats.bts.gov/
- **Period:** 2022-01 through 2025-12 (48 months).
- **Raw files:** 48 monthly ZIP archives downloaded from BTS.
- **Processed files:** 48 monthly Parquet files derived from the raw archives.
- **Research cohort:** Approximately 27.66 million scheduled flight records before outcome eligibility exclusions (see the data-audit outputs for exact definitions and counts).

The raw and processed flight-level datasets are **not committed to GitHub** because of their size. Researchers should obtain the original monthly files from BTS and use the preprocessing workflow to reconstruct the analysis files. Record the exact source query, columns, download date and file checksums in `data/README.md` and `data/manifests/`.

## Repository layout

The layout below is a **recommended target structure**. Existing script names and file paths should be retained until the documented commands have been tested.

```text
aviation-delay-generalisation/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── raw/                  # 48 monthly BTS ZIP files; local only
│   ├── processed/            # 48 monthly Parquet files; local only
│   └── manifests/            # source URLs, dates, hashes; tracked
├── outputs/                  # selected small CSV/JSON experiment outputs
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/
│   ├── phase6b_fixed/
│   └── phase7_inference/
├── figures/                  # publication figures and plotting code
├── docs/                     # data dictionary and experiment notes
└── *.py                      # analysis scripts; keep original working paths
```

### Data directories

`data/raw/` contains the **48 original monthly ZIP files** (2022-01 to 2025-12). `data/processed/` contains the corresponding **48 monthly Parquet files**. Files may be flat or arranged into year subfolders; what matters is that the scripts' input paths match the actual layout. Do not ZIP the Parquet files again for routine analysis.

`outputs/` contains *results* (small CSV/JSON metrics, configuration files and uncertainty summaries), not input data. Only commit results that are required to reconstruct the manuscript tables and figures. Large prediction-level Parquet outputs should be stored outside GitHub and documented separately.

## Installation

Recommended: Python 3.10+ and a virtual environment. The actual Python and dependency versions used for the submitted experiments should be recorded in a frozen environment file.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

Typical dependencies include `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `xgboost`, `scipy` and `matplotlib`. The checked-in `requirements.txt` must match the actual scripts; do not assume this list is exhaustive.

## Analysis workflow

The project was developed in sequential phases. Script names below reflect the research workflow; check the repository for the exact filename (some local versions use different numbering or have been superseded).

| Stage | Script / component | Output or purpose |
| --- | --- | --- |
| 1. Data audit and EDA | `01_audit_and_eda.py` | Eligibility, missingness, annual/monthly delay statistics |
| 2. Preliminary baseline | `02_temporal_baseline.py` | Initial logistic regression |
| 3. Chronological evaluation | `03_generalisation_experiments.py` | Logistic vs XGBoost, annual/monthly metrics |
| 4. Market holdouts | `04_cross_market_experiments.py` | Held-out airlines and origin airports |
| 5. Matched-market comparison | `05_matched_market.py` | Included vs excluded target-market training |
| 6. Calibration and shift | `06_calibration_robustness.py` | Raw, Platt and isotonic predictions; shift diagnostics |
| 7. Calibration windows | `07_calibration_window_sensitivity.py` | Q4 vs H2 2023 recalibration |
| 8. Robustness | `08_market_robustness_fixed.py` | Multiple training sizes and random seeds |
| 9. Aggregate audit | `09_phase7_audit.py` | Result integrity and summary checks |
| 10. Paired inference | `10_paired_prediction_inference.py` | Paired AUROC/Brier differences and clustered intervals |
| Figures | `make_figures.py` and any supplementary plotting scripts | Manuscript figures from archived CSV outputs |

**Do not run every historical version blindly.** Use the corrected scripts that produced the final manuscript numbers; earlier pilot scripts and the superseded `08_market_robustness.py` should be clearly marked as exploratory if retained.

### Running experiments

Most phase scripts accept command-line options such as `--input-dir` and `--output-dir`. For example, the final robustness experiment was run using the corrected script:

```powershell
python 08_market_robustness_fixed.py --input-dir data --output-dir outputs/phase6b_fixed
```

And the paired-inference stage used:

```powershell
python 10_paired_prediction_inference.py --input-dir data --output-dir outputs/phase7_inference
```

These are examples from the research workflow, **not a guarantee of a one-command clean run**. Run `python <script>.py --help`, confirm archive locations and dependencies, and document the tested commands and environment before release. Some scripts import helpers from other files in the repository root.

## Experimental design

### Chronological evaluation

The main temporal comparison trains on historical 2022–2023 observations, uses a chronologically later 2023 validation period, and evaluates separately on **2024 and 2025**. The precise split differs in the calibration-window sensitivity experiment, where the base model is trained only through June 2023 to avoid overlap with July–December calibration observations.

### Matched-market experiments

Six markets are evaluated: **Southwest (WN), Delta (DL), American (AA), Atlanta (ATL), Dallas/Fort Worth (DFW), and Denver (DEN)**. For each target market, two equally sized historical training samples are constructed: one containing target-market observations and another excluding them. The resulting models predict **identical target-market test flights** in 2024 and 2025. The robustness design uses a fixed **25% target-market quota** in the included condition, three training sizes (50,000, 100,000 and 160,000), and three seeds (42, 123 and 2026).

This comparison measures a **design-specific training-composition contrast**, not a causal effect of target-market data access. The 216 repeated configurations are correlated and must not be treated as 216 independent statistical tests.

### Paired uncertainty analysis

A primary configuration (training size 100,000; seed 42) evaluates both models across six markets and two years, yielding 24 paired comparisons. Departure-day cluster bootstrap intervals are computed on paired flight-level predictions. These intervals are conditional on fitted models and selected markets; they do not fully capture retraining or market-selection uncertainty.

### Calibration sensitivity

The analysis compares raw probabilities with Platt and isotonic recalibration using Q4 2023 versus H2 2023 calibration windows. Both are evaluated on later 2024 and 2025 flights, with non-overlapping base-model training data for the window-sensitivity experiment.

## Main archived findings

The following values are **descriptive summaries of the saved experiment outputs**, not independent confirmations from a fresh BTS-to-results run:

| Result | Value |
| --- | ---: |
| Completed matched-market robustness comparisons | 216 |
| Comparisons with positive AUROC change | 213 / 216 |
| Comparisons with lower Brier score | 185 / 216 |
| Mean included-minus-excluded AUROC | +0.00987 |
| Mean included-minus-excluded Brier score | -0.00270 |
| Primary paired comparisons with positive AUROC difference | 24 / 24 |
| Unadjusted 95% paired AUROC intervals above zero | 19 / 24 |

The calibration analyses indicate that recalibration using a short Q4 2023 window worsened future Brier scores in the examined settings; the longer H2 2023 window reduced this penalty but did not consistently beat raw predictions. These results should not be generalised to all recalibration procedures or markets.

## Reproducing manuscript tables and figures

1. Download the 48 original monthly BTS ZIP files and place them where the preprocessing script expects them.
2. Generate the 48 monthly Parquet files using the project's data-processing code.
3. Run the final (non-pilot) experiment scripts with their recorded configurations.
4. Verify the output CSV and JSON files against the archived result manifest/checksums.
5. Run the figure-generation script(s) on the archived result CSVs.
6. Confirm that each figure and table in the manuscript has a documented source file and generation command.

For the previously prepared nine-figure plotting package, the plotting command is:

```powershell
python make_figures.py --results-dir results --output-dir reproduced_figures
```

This command requires the expected `results/` inputs from that plotting package. If this repository uses `outputs/` instead, update the input mapping or provide the appropriate CSV copies. **The additional manuscript figures may require a second plotting script; include and document it before submission.**

## Reproduction checklist

- [ ] All 48 BTS raw monthly archives are documented with exact filenames and download details.
- [ ] Raw archive checksums and processing-code versions are recorded.
- [ ] All 48 processed monthly Parquet files can be regenerated.
- [ ] Every final experiment script runs in a clean environment.
- [ ] Output configurations and random seeds are archived.
- [ ] All manuscript tables and figures map to specific source files and scripts.
- [ ] All reported metrics match regenerated outputs within documented numerical tolerances.
- [ ] The public release has been archived with a permanent DOI (e.g., Zenodo).

## Data and code availability

The original flight records are publicly available through the [BTS TranStats portal](https://www.transtats.bts.gov/). Raw and processed flight-level files are excluded from this GitHub repository because of their size. The repository provides analysis code, experimental configurations, selected derived result tables and figure-generation instructions. A permanent archived release DOI should be added here after deposition.

## Citation

If you use this repository, please cite the associated article when published. The full article citation and repository DOI will be added after acceptance and archival release. A `CITATION.cff` file may be added once the author, title and DOI metadata have been confirmed.

## License

**To be selected by the author before public release.** Do not assume that the research code, manuscript text and third-party BTS data share the same license. A code license such as MIT can be considered for author-owned scripts; verify any third-party dependencies and materials separately.

## Contact

For questions about reproduction, use the GitHub repository's Issues tab or the corresponding-author contact listed in the published article.
