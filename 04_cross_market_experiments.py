"""Phase 3: held-out carrier / origin-airport generalisation and temporal recalibration.
Reads 2022.zip ... 2025.zip, each containing 12 monthly Parquet files.
"""
import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
CATS = ['Reporting_Airline', 'Origin', 'Dest', 'DayOfWeek', 'Month']
NUMS = ['CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'Distance']
FEATURES = CATS + NUMS
COLS = FEATURES + ['ArrDelay', 'Cancelled', 'Diverted', 'FlightDate']

def read_months(root):
    for year in range(2022, 2026):
        archive = root / f'{year}.zip'
        if not archive.exists():
            raise FileNotFoundError(f'Missing archive: {archive}')
        with zipfile.ZipFile(archive) as z:
            found = {}
            for member in z.namelist():
                match = re.search(rf'_{year}_(\d{{1,2}})\.parquet$', member)
                if match:
                    month = int(match.group(1))
                    if month in found:
                        raise ValueError(f'Duplicate month: {year}-{month}')
                    found[month] = member
            if set(found) != set(range(1, 13)):
                raise ValueError(f'Expected 12 months in {archive}; got {sorted(found)}')
            for month in range(1, 13):
                with z.open(found[month]) as f:
                    d = pd.read_parquet(io.BytesIO(f.read()), columns=COLS)
                valid = (pd.to_numeric(d.Cancelled, errors='coerce').eq(0)
                         & pd.to_numeric(d.Diverted, errors='coerce').eq(0)
                         & pd.to_numeric(d.ArrDelay, errors='coerce').notna())
                d = d.loc[valid, FEATURES + ['ArrDelay', 'FlightDate']].copy()
                d['target'] = (pd.to_numeric(d.ArrDelay) >= 15).astype('int8')
                d.drop(columns='ArrDelay', inplace=True)
                for c in CATS:
                    d[c] = d[c].astype('string').fillna('__MISSING__').astype(str)
                for c in NUMS:
                    d[c] = pd.to_numeric(d[c], errors='coerce').astype('float32')
                dates = pd.to_datetime(d.FlightDate, errors='raise')
                if not ((dates.dt.year == year) & (dates.dt.month == month)).all():
                    raise ValueError(f'Date mismatch in {year}-{month}')
                d.drop(columns='FlightDate', inplace=True)
                yield year, month, d

def safe_metrics(y, p):
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    out = dict(n=len(y), prevalence=float(y.mean()),
               AUPRC=float(average_precision_score(y, p)),
               Brier=float(brier_score_loss(y, p)),
               log_loss=float(log_loss(y, p, labels=[0, 1])),
               mean_predicted=float(p.mean()),
               calibration_in_the_large=float(p.mean() - y.mean()))
    out['AUROC'] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    return out

def make_model(name):
    if name == 'logistic':
        prep = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=100), CATS),
            ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), NUMS)])
        clf = LogisticRegression(max_iter=500)
    else:
        prep = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=100), CATS),
            ('num', SimpleImputer(strategy='median'), NUMS)])
        clf = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.05,
                            subsample=0.85, colsample_bytree=0.85,
                            tree_method='hist', eval_metric='logloss',
                            n_jobs=4, random_state=SEED)
    return Pipeline([('prep', prep), ('model', clf)])

def sample_training(root, train_cap, excluded_col=None, excluded=None):
    rng = np.random.default_rng(SEED)
    # Reservoir-like per-month sampling; cap spread across 21 training months.
    pieces = []
    monthly_cap = max(1, int(np.ceil(train_cap / 21)))
    for year, month, d in read_months(root):
        if year == 2022 or (year == 2023 and month <= 9):
            if excluded_col:
                d = d[d[excluded_col] != excluded]
            if len(d):
                ix = rng.choice(len(d), min(len(d), monthly_cap), replace=False)
                pieces.append(d.iloc[ix][FEATURES + ['target']].copy())
    if not pieces:
        raise ValueError('No training records after exclusions')
    train = pd.concat(pieces, ignore_index=True)
    if len(train) > train_cap:
        train = train.sample(n=train_cap, random_state=SEED).reset_index(drop=True)
    return train

def choose_holdouts(root, group_col, count=3):
    # Deterministic selection: most frequent groups in training period.
    totals = {}
    for year, month, d in read_months(root):
        if year == 2022 or (year == 2023 and month <= 9):
            for group, n in d[group_col].value_counts().items():
                if group != '__MISSING__':
                    totals[group] = totals.get(group, 0) + int(n)
    return [g for g, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:count]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, default=Path('data'))
    ap.add_argument('--output-dir', type=Path, default=Path('outputs/phase3'))
    ap.add_argument('--train-cap', type=int, default=300000)
    ap.add_argument('--holdouts-per-type', type=int, default=3)
    ap.add_argument('--models', nargs='+', choices=['logistic', 'xgboost'], default=['logistic', 'xgboost'])
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    holdouts = {col: choose_holdouts(args.input_dir, col, args.holdouts_per_type)
                for col in ['Reporting_Airline', 'Origin']}
    (args.output_dir / 'holdout_groups.json').write_text(json.dumps(holdouts, indent=2))
    print('Held-out groups:', holdouts, flush=True)
    results = []
    # Separate model fitted for each holdout; 2024 and 2025 remain future test years.
    for col, groups in holdouts.items():
        for group in groups:
            print('Training exclusion:', col, group, flush=True)
            train = sample_training(args.input_dir, args.train_cap, col, group)
            for model_name in args.models:
                model = make_model(model_name)
                model.fit(train[FEATURES], train.target)
                for year, month, d in read_months(args.input_dir):
                    if year not in [2024, 2025]:
                        continue
                    for segment, mask in [('unseen', d[col].eq(group)),
                                          ('seen', d[col].ne(group))]:
                        sub = d.loc[mask]
                        if not len(sub):
                            continue
                        p = model.predict_proba(sub[FEATURES])[:, 1]
                        row = safe_metrics(sub.target.to_numpy(), p)
                        row.update(experiment=col, holdout=group, segment=segment,
                                   model=model_name, year=year, month=month)
                        results.append(row)
                print('Finished:', col, group, model_name, flush=True)
            del train
    df = pd.DataFrame(results)
    df.to_csv(args.output_dir / 'cross_market_monthly_metrics.csv', index=False)
    # Annual AUROC is not the mean of monthly AUROCs. These are descriptive weighted
    # monthly summaries, NOT pooled annual AUROC estimates.
    summaries = []
    keys = ['experiment', 'holdout', 'segment', 'model', 'year']
    for group_key, g in df.groupby(keys, dropna=False):
        valid = g.AUROC.notna()
        row = dict(zip(keys, group_key))
        row.update(n=int(g.n.sum()),
                   weighted_monthly_AUROC=(float(np.average(g.loc[valid, 'AUROC'], weights=g.loc[valid, 'n']))
                                           if valid.any() else np.nan),
                   weighted_monthly_Brier=float(np.average(g.Brier, weights=g.n)),
                   prevalence=float(np.average(g.prevalence, weights=g.n)),
                   months=int(len(g)))
        summaries.append(row)
    agg = pd.DataFrame(summaries)
    agg.to_csv(args.output_dir / 'cross_market_weighted_monthly_summary.csv', index=False)
    print('Saved outputs to', args.output_dir, flush=True)

if __name__ == '__main__':
    main()
