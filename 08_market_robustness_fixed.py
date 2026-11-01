"""Phase 6B: controlled market-specific historical data sensitivity study."""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

CATS = ['Reporting_Airline', 'Origin', 'Dest']
NUMS = ['Month', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'Distance']
COLS = ['FlightDate', 'ArrDelay', 'Cancelled', 'Diverted'] + CATS + NUMS
MARKETS = {'Reporting_Airline': ['WN', 'DL', 'AA'], 'Origin': ['ATL', 'DFW', 'DEN']}


def load_data(folder, per_month, seed, target_per_month=3000):
    """Retain independent per-market samples plus a general background sample.

    Sample without replacement within each monthly file. Overlap between market
    samples is deduplicated; this avoids underrepresenting small airports.
    """
    chunks = []
    for year in range(2022, 2026):
        path = folder / f'{year}.zip'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}')
        with zipfile.ZipFile(path) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith('.parquet'))
            if len(names) != 12:
                raise ValueError(f'{path.name}: expected 12 Parquet files; found {len(names)}')
            for month_index, name in enumerate(names):
                with z.open(name) as handle:
                    df = pd.read_parquet(io.BytesIO(handle.read()), columns=COLS)
                df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')
                df = df.loc[(df['Cancelled'] == 0) & (df['Diverted'] == 0) & df['ArrDelay'].notna() & df['FlightDate'].notna()].copy()
                rng_seed = seed + year * 100 + month_index
                selected = set(df.sample(n=min(per_month, len(df)), random_state=rng_seed).index)
                for market_col, values in MARKETS.items():
                    for j, value in enumerate(values):
                        pool = df.index[df[market_col] == value]
                        if len(pool):
                            sampled = pd.Series(pool).sample(n=min(target_per_month, len(pool)), random_state=rng_seed + 100 + j + (0 if market_col == 'Origin' else 10))
                            selected.update(sampled.to_list())
                part = df.loc[sorted(selected)].copy()
                part['target'] = (pd.to_numeric(part['ArrDelay']) >= 15).astype('int8')
                chunks.append(part[['FlightDate', 'target'] + CATS + NUMS])
                print(f'Loaded {year} {name}: {len(part):,}', flush=True)
    return pd.concat(chunks, ignore_index=True)


def make_model(name, seed):
    prep = ColumnTransformer([
        ('num', Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMS),
        ('cat', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATS)
    ])
    if name == 'logistic':
        estimator = LogisticRegression(max_iter=350, solver='lbfgs')
    else:
        estimator = XGBClassifier(n_estimators=180, max_depth=5, learning_rate=0.06,
                                  subsample=0.85, colsample_bytree=0.85,
                                  tree_method='hist', n_jobs=4, random_state=seed,
                                  eval_metric='logloss')
    return Pipeline([('prep', prep), ('model', estimator)])


def metric(y, p):
    return {'auroc': roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            'auprc': average_precision_score(y, p),
            'brier': brier_score_loss(y, p),
            'calibration_bias': float(np.mean(p) - np.mean(y)),
            'prevalence': float(np.mean(y)), 'n_test': len(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='data')
    ap.add_argument('--output-dir', default='outputs/phase6b')
    ap.add_argument('--per-month', type=int, default=25000)
    ap.add_argument('--target-per-month', type=int, default=3000)
    ap.add_argument('--train-sizes', default='50000,100000,160000')
    ap.add_argument('--seeds', default='42,123,2026')
    ap.add_argument('--test-per-year', type=int, default=10000)
    ap.add_argument('--target-share', type=float, default=0.25)
    ap.add_argument('--models', choices=['both','logistic','xgboost'], default='both')
    ap.add_argument('--pilot', action='store_true')
    args = ap.parse_args()
    sizes = [int(x) for x in args.train_sizes.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    if args.pilot:
        sizes, seeds = [10000], [42]
        args.per_month, args.test_per_year, args.target_per_month = 5000, 2000, 1000
    if not 0 < args.target_share < 1:
        raise ValueError('target-share must be strictly between zero and one')
    models = ['logistic', 'xgboost'] if args.models == 'both' else [args.models]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        data = load_data(Path(args.input_dir), args.per_month, seed, args.target_per_month)
        training = data.loc[data.FlightDate < '2023-10-01'].copy()
        tests = {y: data.loc[data.FlightDate.dt.year == y].copy() for y in (2024, 2025)}
        for market_col, values in MARKETS.items():
            for value in (values[:1] if args.pilot else values):
                pool_target = training.loc[training[market_col] == value]
                pool_other = training.loc[training[market_col] != value]
                target_tests = {}
                for year, frame in tests.items():
                    target = frame.loc[frame[market_col] == value]
                    if len(target) > args.test_per_year:
                        target = target.sample(n=args.test_per_year, random_state=seed + year)
                    target_tests[year] = target
                for size in sizes:
                    n_target = round(size * args.target_share)
                    if len(pool_target) < n_target or len(pool_other) < size:
                        print(f'SKIP {market_col}={value}, size={size}: insufficient sampled training pool', flush=True)
                        continue
                    # Fixed common other-market pool; inclusion swaps n_target other-market rows
                    common = pool_other.sample(n=size-n_target, random_state=seed+size)
                    remainder = pool_other.drop(index=common.index).sample(n=n_target, random_state=seed+size+1)
                    included = pd.concat([common, pool_target.sample(n=n_target, random_state=seed+size+2)])
                    excluded = pd.concat([common, remainder])
                    for model_name in models:
                        preds = {}
                        for condition, train_frame in [('included', included), ('excluded', excluded)]:
                            model = make_model(model_name, seed)
                            model.fit(train_frame[CATS+NUMS], train_frame.target)
                            for year, target in target_tests.items():
                                if len(target) < 50 or target.target.nunique() < 2:
                                    continue
                                preds[(condition,year)] = model.predict_proba(target[CATS+NUMS])[:,1]
                        for year, target in target_tests.items():
                            if ('included',year) not in preds or ('excluded',year) not in preds:
                                continue
                            a = metric(target.target, preds[('included',year)])
                            b = metric(target.target, preds[('excluded',year)])
                            results.append(dict(seed=seed, training_size=size, market_type=market_col,
                                market=value, model=model_name, test_year=year,
                                target_training_n=n_target, common_other_n=len(common),
                                **{f'included_{k}':v for k,v in a.items()},
                                **{f'excluded_{k}':v for k,v in b.items()},
                                delta_auroc=a['auroc']-b['auroc'],
                                delta_brier=a['brier']-b['brier']))
                            print(f'{market_col}={value} {model_name} size={size} seed={seed} year={year}: AUROC delta={a["auroc"]-b["auroc"]:.4f}', flush=True)
        del data, training, tests
    if not results:
        raise RuntimeError('No experiments completed. Increase --per-month or lower --train-sizes.')
    result = pd.DataFrame(results)
    expected = len(seeds) * len(sizes) * sum(len(v) for v in MARKETS.values()) * len(models) * 2
    if len(result) != expected:
        result.to_csv(out/'incomplete_market_robustness_metrics.csv', index=False)
        raise RuntimeError(f'Only {len(result)}/{expected} comparisons completed. See incomplete CSV; increase --target-per-month or --per-month.')
    result.to_csv(out/'market_robustness_metrics.csv', index=False)
    summary = result.groupby(['market_type','market','model','test_year'], as_index=False).agg(
        comparisons=('delta_auroc','size'), mean_delta_auroc=('delta_auroc','mean'),
        sd_delta_auroc=('delta_auroc','std'), positive_auroc_fraction=('delta_auroc',lambda s:(s>0).mean()),
        mean_delta_brier=('delta_brier','mean'), improved_brier_fraction=('delta_brier',lambda s:(s<0).mean()))
    summary.to_csv(out/'market_robustness_summary.csv', index=False)
    config = vars(args).copy(); config['train_sizes_parsed']=sizes; config['seeds_parsed']=seeds
    (out/'run_config.json').write_text(json.dumps(config, indent=2))
    print('Completed:', len(result), 'comparisons. Outputs:', out)

if __name__ == '__main__':
    main()
