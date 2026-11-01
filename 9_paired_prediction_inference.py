"""Phase 7: prediction-level paired inference on a PRE-SPECIFIED Phase 6B subset.

Primary analysis: seed 42, training size 100000, all six markets, both models,
2024 and 2025. This is not a re-analysis of 216 independent experiments.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from phase6b_models import load_data, make_model, CATS, NUMS, MARKETS


def deltas(y, pi, pe):
    return (float(roc_auc_score(y, pi)-roc_auc_score(y, pe)),
            float(brier_score_loss(y, pi)-brier_score_loss(y, pe)))


def paired_day_bootstrap(df, reps, seed):
    """Resample departure dates within each calendar month; retain paired predictions.

    CIs condition on the trained models and sampled test flights. They do not
    capture model-fit uncertainty, market-selection uncertainty, or all serial
    dependencies. Date clusters are not independent in every operational setting.
    """
    rng = np.random.default_rng(seed)
    date = pd.to_datetime(df['FlightDate'])
    month = date.dt.to_period('M')
    groups = []
    for m in sorted(month.unique()):
        mpos = np.flatnonzero((month == m).to_numpy())
        dates = date.iloc[mpos].dt.normalize().to_numpy()
        groups.append([mpos[dates == d] for d in np.unique(dates)])
    y = df['y_true'].to_numpy(dtype=np.int8)
    pi = df['p_included'].to_numpy(dtype=float)
    pe = df['p_excluded'].to_numpy(dtype=float)
    estimates = np.empty((reps, 2))
    for b in range(reps):
        parts = []
        for days in groups:
            draws = rng.integers(0, len(days), size=len(days))
            parts.extend(days[j] for j in draws)
        idx = np.concatenate(parts)
        if np.unique(y[idx]).size < 2:
            estimates[b] = np.nan
        else:
            estimates[b] = deltas(y[idx], pi[idx], pe[idx])
    return estimates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='data')
    ap.add_argument('--output-dir', default='outputs/phase7_inference')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--training-size', type=int, default=100000)
    ap.add_argument('--per-month', type=int, default=25000)
    ap.add_argument('--target-per-month', type=int, default=3000)
    ap.add_argument('--target-share', type=float, default=0.25)
    ap.add_argument('--test-per-year', type=int, default=10000)
    ap.add_argument('--bootstrap-reps', type=int, default=400)
    ap.add_argument('--models', choices=['both', 'logistic', 'xgboost'], default='both')
    ap.add_argument('--pilot', action='store_true')
    args = ap.parse_args()
    if args.pilot:
        args.training_size = 10000
        args.per_month = 5000
        args.target_per_month = 1000
        args.test_per_year = 2000
        args.bootstrap_reps = 30
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = ['logistic', 'xgboost'] if args.models == 'both' else [args.models]
    data = load_data(Path(args.input_dir), args.per_month, args.seed, args.target_per_month)
    training = data.loc[data.FlightDate < '2023-10-01'].copy()
    tests = {y: data.loc[data.FlightDate.dt.year == y].copy() for y in (2024, 2025)}
    summaries = []
    expected = sum(map(len, MARKETS.values())) * len(models) * 2
    for market_col, markets in MARKETS.items():
        for market in (markets[:1] if args.pilot else markets):
            target_pool = training.loc[training[market_col] == market]
            other_pool = training.loc[training[market_col] != market]
            n_target = round(args.training_size * args.target_share)
            if len(target_pool) < n_target or len(other_pool) < args.training_size:
                raise RuntimeError(f'Insufficient training pool: {market_col}={market}')
            common = other_pool.sample(n=args.training_size-n_target, random_state=args.seed+args.training_size)
            remainder = other_pool.drop(index=common.index).sample(n=n_target, random_state=args.seed+args.training_size+1)
            included = pd.concat([common, target_pool.sample(n=n_target, random_state=args.seed+args.training_size+2)])
            excluded = pd.concat([common, remainder])
            target_tests = {}
            for year, frame in tests.items():
                target = frame.loc[frame[market_col] == market]
                if len(target) > args.test_per_year:
                    target = target.sample(n=args.test_per_year, random_state=args.seed+year)
                if target.target.nunique() != 2:
                    raise RuntimeError(f'Only one outcome class for {market} {year}')
                target_tests[year] = target
            for model_name in models:
                preds = {}
                for condition, sample in [('included', included), ('excluded', excluded)]:
                    model = make_model(model_name, args.seed)
                    model.fit(sample[CATS+NUMS], sample.target)
                    for year, target in target_tests.items():
                        preds[(condition, year)] = model.predict_proba(target[CATS+NUMS])[:, 1]
                for year, target in target_tests.items():
                    frame = pd.DataFrame({
                        'FlightDate': target.FlightDate.to_numpy(),
                        'y_true': target.target.to_numpy(),
                        'p_included': preds[('included', year)],
                        'p_excluded': preds[('excluded', year)]})
                    # No stable flight ID was retained in the feature extraction.
                    # row_id is unique only within this output, not an external flight identifier.
                    frame.insert(0, 'row_id', np.arange(len(frame)))
                    frame.insert(1, 'market_type', market_col)
                    frame.insert(2, 'market', market)
                    frame.insert(3, 'model', model_name)
                    frame.insert(4, 'test_year', year)
                    filename = f'paired_{market_col}_{market}_{model_name}_{year}.parquet'
                    frame.to_parquet(out / filename, index=False, compression='zstd')
                    auroc, brier = deltas(frame.y_true, frame.p_included, frame.p_excluded)
                    draws = paired_day_bootstrap(frame, args.bootstrap_reps, args.seed+year)
                    valid = draws[np.isfinite(draws).all(axis=1)]
                    if len(valid) < 0.95 * args.bootstrap_reps:
                        raise RuntimeError('Too many bootstrap replicates had one class')
                    ci = np.quantile(valid, [0.025, 0.975], axis=0)
                    summaries.append(dict(market_type=market_col, market=market,
                        model=model_name, test_year=year, seed=args.seed,
                        training_size=args.training_size, n_test=len(frame),
                        n_dates=frame.FlightDate.nunique(),
                        delta_auroc=auroc, auroc_ci_low=ci[0,0], auroc_ci_high=ci[1,0],
                        delta_brier=brier, brier_ci_low=ci[0,1], brier_ci_high=ci[1,1],
                        bootstrap_reps=len(valid)))
                    pd.DataFrame(summaries).to_csv(out/'paired_inference_summary.csv', index=False)
                    print(f'{market_col}={market} {model_name} {year}: delta AUROC={auroc:+.4f}, delta Brier={brier:+.5f}', flush=True)
    if not args.pilot and len(summaries) != expected:
        raise RuntimeError(f'Incomplete: {len(summaries)} of {expected} configurations')
    (out/'run_config.json').write_text(json.dumps(vars(args), indent=2))
    print(f'Finished: {len(summaries)} paired analyses. Files in {out}')

if __name__ == '__main__':
    main()
