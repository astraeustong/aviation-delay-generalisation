"""Reproduce Figures 1-12 in manuscript.docx from archived experiment CSVs.

Run from repository root:
    python figures/make_figures.py --results-dir outputs --output-dir figures/reproduced_figures

This regenerates figures from *saved experiment outputs*, not from raw BTS data.
The CSV filenames are searched recursively; duplicate filenames with different
contents trigger an error rather than silently choosing one.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--results-dir', type=Path, default=Path('outputs'))
PARSER.add_argument('--output-dir', type=Path, default=Path('figures/reproduced_figures'))
PARSER.add_argument('--manifest', type=Path, default=None, help='Output manifest path (default: <output-dir>/figure_manifest.json)')
ARGS = PARSER.parse_args()
OUT = ARGS.output_dir
OUT.mkdir(parents=True, exist_ok=True)
ROOT = ARGS.results_dir
if not ROOT.is_dir():
    PARSER.error(f'Cannot find results directory: {ROOT}. Run from repository root or pass --results-dir.')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

INPUTS = {}
def load(name, aliases=()):
    candidates = []
    candidates.extend(ROOT.rglob(name + '.csv'))
    if not candidates:
        for stem in aliases:
            candidates.extend(ROOT.rglob(stem + '.csv'))
    candidates = sorted(set(p.resolve() for p in candidates))
    if not candidates:
        raise FileNotFoundError(f'Missing {name}.csv below {ROOT}. Required to reproduce manuscript figures.')
    hashes = {digest(p) for p in candidates}
    if len(hashes) > 1:
        raise ValueError(f'Ambiguous input for {name}: multiple different CSVs: {candidates}. Keep only the final result or point to a curated results folder.')
    p = candidates[0]
    INPUTS[name] = {'path': str(p), 'sha256': digest(p)}
    return pd.read_csv(p)

monthly_audit = load('monthly_audit')
annual = load('annual_model_metrics')
monthly = load('monthly_model_metrics')
cross = load('cross_market_weighted_monthly_summary')
robust = load('market_robustness_metrics(1)', aliases=('market_robustness_metrics',))
paired = load('paired_inference_summary')
cal = load('calibration_window_annual')

if len(monthly_audit) != 48 or len(robust) != 216 or len(paired) != 24 or len(cal) != 60:
    raise ValueError(f'Unexpected experiment sizes: monthly={len(monthly_audit)}, robustness={len(robust)}, paired={len(paired)}, calibration={len(cal)}')
if robust[['seed','training_size','market','model','test_year']].duplicated().any():
    raise ValueError('Duplicate robustness configuration keys')
if paired[['market','model','test_year']].duplicated().any():
    raise ValueError('Duplicate primary inference configuration keys')

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 200})
FIGURES = {}
def save(num, name, caption):
    filename = f'fig{num:02d}_{name}.png'
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=200, bbox_inches='tight')
    plt.close()
    FIGURES[str(num)] = {'filename': filename, 'caption': caption,
                         'sha256': digest(OUT / filename)}

def plot_market_labels(ax):
    ax.axvline(0, color='0.4', lw=.8)

# Fig. 1: Monthly delay prevalence
fig, ax = plt.subplots(figsize=(8.2, 3.6))
for year, g in monthly_audit.groupby('year'):
    g = g.sort_values('month')
    ax.plot(g.month, g.delay_rate * 100, marker='o', ms=3, label=str(year))
ax.set(xticks=range(1, 13), xlabel='Calendar month', ylabel='Arrival delay >= 15 min (%)',
       title='Monthly observed delay prevalence, 2022-2025')
ax.legend(ncol=4)
save(1, 'monthly_delay', 'Monthly prevalence of arrival delays of at least 15 minutes among eligible arrivals.')

# Fig. 2: Chronological AUROC; explicit validation/test ordering
fig, ax = plt.subplots(figsize=(8.2, 3.6))
periods = [(2023, 'validation_oct_dec', 'Q4 2023'), (2024, 'test', '2024'), (2025, 'test', '2025')]
for model, g in annual.groupby('model'):
    ys = []
    for year, period, label in periods:
        sel = g[(g.year == year) & (g.period == period)]
        if len(sel) != 1:
            raise ValueError(f'Expected one annual row for {model} {year} {period}; found {len(sel)}')
        ys.append(float(sel.AUROC.iloc[0]))
    ax.plot([p[2] for p in periods], ys, marker='o', label=model)
ax.set(ylabel='AUROC', title='Chronological evaluation by model')
ax.legend()
save(2, 'temporal_auroc', 'Discrimination across chronological validation and two future test years.')

# Fig. 3: Monthly AUROC
fig, ax = plt.subplots(figsize=(8.2, 3.6))
for model, g in monthly[monthly.year.isin([2024, 2025])].groupby('model'):
    g = g.sort_values(['year', 'month'])
    if len(g) != 24:
        raise ValueError(f'Expected 24 months for {model}, got {len(g)}')
    ax.plot(np.arange(1, 25), g.AUROC, label=model)
ax.set(xlabel='Test month (Jan 2024-Dec 2025)', ylabel='AUROC',
       title='Monthly out-of-time discrimination', xticks=[1, 4, 7, 10, 13, 16, 19, 22, 24])
ax.legend()
save(3, 'monthly_auroc', 'Monthly AUROC in 2024 and 2025.')

# Fig. 4: Exploratory unmatched cross-market comparison
fig, ax = plt.subplots(figsize=(8.2, 3.6))
g = cross[(cross.model == 'xgboost') & (cross.year == 2025)]
pv = g.pivot_table(index=['experiment', 'holdout'], columns='segment', values='weighted_monthly_AUROC')
if not {'seen', 'unseen'}.issubset(pv.columns):
    raise ValueError('Cross-market file lacks seen/unseen segments')
d = (pv['unseen'] - pv['seen']).dropna().sort_values()
ax.barh([str(i[1]) for i in d.index], d.values)
plot_market_labels(ax)
ax.set(xlabel='Unseen minus seen weighted monthly AUROC',
       title='Descriptive cross-market differences (not matched)')
save(4, 'crossmarket', 'Descriptive Phase 3 unseen-minus-seen weighted monthly AUROC for XGBoost in 2025.')

# Fig. 5: Market-level robustness
fig, ax = plt.subplots(figsize=(8.2, 3.6))
g = robust.groupby('market').delta_auroc.mean().sort_values()
ax.barh(g.index, g.values)
ax.set(xlabel='Mean AUROC difference (included - excluded)',
       title='Target-market heterogeneity across 216 configurations')
save(5, 'market_gains', 'Mean market-specific AUROC gains across models, seeds, training sizes and test years.')

# Fig. 6: Training-size and model interaction
fig, ax = plt.subplots(figsize=(8.2, 3.6))
g = robust.groupby(['training_size', 'model']).delta_auroc.mean().unstack()
g.plot(kind='bar', ax=ax, rot=0)
ax.set(xlabel='Training observations', ylabel='Mean AUROC gain', title='Training-size sensitivity')
save(6, 'training_size', 'Mean included-minus-excluded AUROC by model and training-set size.')

# Fig. 7: Paired inference
fig, ax = plt.subplots(figsize=(8.2, 6.0))
g = paired.sort_values('delta_auroc')
labels = [f'{r.market} {"LR" if r.model == "logistic" else "XGB"} {r.test_year}' for r in g.itertuples()]
y = np.arange(len(g))
ax.errorbar(g.delta_auroc, y, xerr=[g.delta_auroc-g.auroc_ci_low,
                                     g.auroc_ci_high-g.delta_auroc], fmt='o', ms=3, capsize=2)
ax.axvline(0, color='0.4', lw=.8)
ax.set(yticks=y, yticklabels=labels,
       xlabel='AUROC difference and 95% day-cluster bootstrap interval',
       title='Primary paired inference (24 contrasts)')
save(7, 'paired_ci', 'Primary paired AUROC differences and unadjusted 95% departure-day cluster-bootstrap intervals.')

# Fig. 8: Calibration Brier scores
fig, ax = plt.subplots(figsize=(8.2, 3.6))
g = cal.groupby(['model', 'year', 'method']).brier.mean()
methods = ['raw', 'platt_q4', 'platt_h2', 'isotonic_q4', 'isotonic_h2']
order = [('logistic', 2024), ('logistic', 2025), ('xgboost', 2024), ('xgboost', 2025)]
for j, method in enumerate(methods):
    vals = [g.loc[(model, year, method)] for model, year in order]
    ax.bar(np.arange(4) + (j-2)*.16, vals, width=.155, label=method)
ax.set(xticks=np.arange(4), xticklabels=['LR 2024', 'LR 2025', 'XGB 2024', 'XGB 2025'],
       ylabel='Brier score (lower is better)', title='Calibration-window sensitivity: three-seed means')
ax.legend(ncol=3, fontsize=8)
save(8, 'calibration_brier', 'Brier scores averaged over three random seeds for raw and recalibrated predictions.')

# Fig. 9: Calibration-in-the-large, 2025
fig, ax = plt.subplots(figsize=(8.2, 3.6))
g = cal[cal.year == 2025].groupby(['model', 'method']).bias.mean()
for model in ['logistic', 'xgboost']:
    ax.plot(methods, [g.loc[(model, method)] * 100 for method in methods], marker='o', label=model)
ax.axhline(0, color='0.5', ls='--')
ax.set(ylabel='Calibration bias (percentage points)', title='2025 calibration-in-the-large by method')
ax.tick_params(axis='x', rotation=20)
ax.legend()
save(9, 'calibration_bias', 'Calibration-in-the-large in 2025; positive values indicate overprediction.')

# Fig. 10: Robustness distribution by training size and model
fig, ax = plt.subplots(figsize=(8.2, 3.7))
sizes = sorted(robust.training_size.unique())
models = ['logistic', 'xgboost']
positions, groups = [], []
for i, size in enumerate(sizes):
    for j, model in enumerate(models):
        vals = robust[(robust.training_size == size) & (robust.model == model)].delta_auroc.values
        groups.append(vals)
        positions.append(i*3 + j*.9)
ax.boxplot(groups, positions=positions, widths=.7, patch_artist=False, showfliers=True)
ax.axhline(0, color='0.6', lw=.8)
ax.set(xticks=[i*3+.45 for i in range(len(sizes))],
       xticklabels=[f'{s:,}' for s in sizes], xlabel='Training sample size (logistic then XGBoost)',
       ylabel='Included minus excluded AUROC', title='Distribution of matched-market gains by training size')
save(10, 'robustness_distribution', 'Distribution of included-minus-excluded AUROC differences by training size and model.')

# Fig. 11: Market-by-training-size sensitivity
fig, ax = plt.subplots(figsize=(8.2, 3.7))
g = robust.groupby(['market', 'training_size']).delta_auroc.mean().unstack()
for market, row in g.iterrows():
    ax.plot(sizes, [row[s] for s in sizes], marker='o', label=market)
ax.set(xlabel='Training size', ylabel='Mean AUROC gain', title='Market-level sensitivity to training size', xticks=sizes)
ax.legend(ncol=6, fontsize=8)
save(11, 'market_size_interaction', 'Market-specific mean AUROC gains across training sizes.')

# Fig. 12: Mean Brier score by future year and calibration method
fig, ax = plt.subplots(figsize=(8.2, 3.7))
g = cal.groupby(['year', 'method']).brier.mean()
for method in methods:
    ax.plot([2024, 2025], [g.loc[(year, method)] for year in [2024, 2025]],
            marker='o', label=method)
ax.set(xticks=[2024, 2025], xlabel='Test year', ylabel='Mean Brier score (three seeds, two models)',
       title='Calibration-window comparison across future years')
ax.legend(ncol=3, fontsize=8)
save(12, 'calibration_year_interaction', 'Mean future-year Brier scores across three seeds and both model families.')

manifest = {
    'manuscript': 'manuscript.docx, Figures 1-12 in appearance order',
    'scope': 'Archived experiment CSV to figure only; NOT raw BTS to fitted models',
    'generator': 'figures/make_figures.py',
    'inputs': INPUTS,
    'figures': FIGURES,
}
manifest_path = ARGS.manifest or (OUT / 'figure_manifest.json')
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(f'Generated {len(FIGURES)} manuscript figures in {OUT.resolve()}')
print(f'Wrote manifest: {manifest_path.resolve()}')
