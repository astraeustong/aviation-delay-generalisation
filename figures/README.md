# Reproduce the 12 figures in the aviation manuscript

This package matches the **figure numbering and content in the uploaded 20-page `manuscript.docx`**. The earlier script produced nine figures with a different numbering sequence; do not use its old manifest for this manuscript.

## Run inside your aviation repository

Place `make_figures.py` at `figures/make_figures.py`. Ensure the seven required experiment CSVs are somewhere beneath `outputs/` (the script searches subfolders recursively). From the repository root:

```powershell
python -m pip install pandas numpy matplotlib
python figures/make_figures.py --results-dir outputs --output-dir figures/reproduced_figures
```

The script writes twelve numbered PNGs and `figures/reproduced_figures/figure_manifest.json` with input and output SHA-256 checksums. To smoke-test without searching your whole outputs tree, use the included `sample_results/` folder:

```powershell
python figures/make_figures.py --results-dir figures/sample_results --output-dir figures/reproduced_figures
```

## Required result files

- `monthly_audit.csv`: Fig. 1
- `annual_model_metrics.csv`: Fig. 2
- `monthly_model_metrics.csv`: Fig. 3
- `cross_market_weighted_monthly_summary.csv`: Fig. 4
- `market_robustness_metrics(1).csv`: Figs. 5, 6, 10, 11 (or `market_robustness_metrics.csv` if it is the **corrected 216-row version**)
- `paired_inference_summary.csv`: Fig. 7
- `calibration_window_annual.csv`: Figs. 8, 9, 12

If different files with the same expected name exist in different subfolders, the script refuses to choose between nonidentical copies. Curate the final result files into one folder if that happens. Keep the original corrected 216-row market-robustness output, not the earlier incomplete 156-row version.

## Figure mapping

| Manuscript | Generated file | Content |
|---|---|---|
| Fig. 1 | fig01_monthly_delay.png | Monthly observed delay prevalence |
| Fig. 2 | fig02_temporal_auroc.png | Chronological AUROC |
| Fig. 3 | fig03_monthly_auroc.png | Monthly AUROC in 2024-2025 |
| Fig. 4 | fig04_crossmarket.png | Descriptive unmatched cross-market differences |
| Fig. 5 | fig05_market_gains.png | Target-market AUROC gains |
| Fig. 6 | fig06_training_size.png | Training size by model |
| Fig. 7 | fig07_paired_ci.png | 24 paired AUROC confidence intervals |
| Fig. 8 | fig08_calibration_brier.png | Calibration Brier scores |
| Fig. 9 | fig09_calibration_bias.png | Calibration-in-the-large |
| Fig. 10 | fig10_robustness_distribution.png | Distribution of matched-market gains |
| Fig. 11 | fig11_market_size_interaction.png | Market by training-size interaction |
| Fig. 12 | fig12_calibration_year_interaction.png | Calibration method by future year |

**Scope:** Reproduces plots from the archived CSV results. It does not retrain models, generate those CSVs from the raw BTS data, or prove that the raw-data pipeline is reproducible. Pixel-identical output is not guaranteed across different Matplotlib versions and fonts; compare plotted values, labels, and grouping rather than image hashes across machines.
