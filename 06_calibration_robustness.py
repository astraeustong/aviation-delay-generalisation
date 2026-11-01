"""Phase 5: strictly out-of-time probability calibration and monthly robustness.
Reads 2022.zip ... 2025.zip containing 12 Parquet files each.
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

CATS = ['Reporting_Airline', 'Origin', 'Dest', 'DayOfWeek', 'Month']
NUMS = ['CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'Distance']
FEATURES = CATS + NUMS
KEEP = FEATURES + ['FlightDate', 'ArrDelay', 'Cancelled', 'Diverted']


def frames(root):
    for year in range(2022, 2026):
        path = root / f'{year}.zip'
        if not path.is_file():
            raise FileNotFoundError(f'Expected {path}')
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if n.lower().endswith('.parquet')]
            by_month = {}
            for member in members:
                m = re.search(rf'_{year}_(\d{{1,2}})\.parquet$', member)
                if m is None:
                    raise ValueError(f'Unexpected member name: {member}')
                month = int(m.group(1))
                if month in by_month:
                    raise ValueError(f'Duplicate month {year}-{month}')
                by_month[month] = member
            if set(by_month) != set(range(1, 13)):
                raise ValueError(f'Missing months for {year}: {sorted(set(range(1,13))-set(by_month))}')
            for month in range(1, 13):
                with z.open(by_month[month]) as f:
                    d = pd.read_parquet(io.BytesIO(f.read()), columns=KEEP)
                dates = pd.to_datetime(d.FlightDate, errors='coerce')
                valid = (pd.to_numeric(d.Cancelled, errors='coerce').eq(0)
                         & pd.to_numeric(d.Diverted, errors='coerce').eq(0)
                         & pd.to_numeric(d.ArrDelay, errors='coerce').notna()
                         & dates.dt.year.eq(year) & dates.dt.month.eq(month))
                y = (pd.to_numeric(d.loc[valid, 'ArrDelay']) >= 15).astype('int8')
                d = d.loc[valid, FEATURES].copy()
                d['target'] = y.to_numpy()
                for col in CATS:
                    d[col] = d[col].astype('string').fillna('__MISSING__').astype(str)
                for col in NUMS:
                    d[col] = pd.to_numeric(d[col], errors='coerce').astype('float32')
                yield year, month, d


def make_model(name, seed):
    prep = ColumnTransformer([
        ('cat', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=100))]), CATS),
        ('num', Pipeline([('fill', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), NUMS),
    ])
    est = (LogisticRegression(max_iter=400) if name == 'logistic' else
           XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                         subsample=0.85, colsample_bytree=0.85,
                         tree_method='hist', n_jobs=4, random_state=seed,
                         eval_metric='logloss'))
    return Pipeline([('prep', prep), ('est', est)])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stats(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=len(y), prevalence=float(y.mean()), mean_pred=float(p.mean()),
                calibration_bias=float(p.mean()-y.mean()),
                auroc=float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else np.nan,
                auprc=float(average_precision_score(y,p)),
                brier=float(brier_score_loss(y,p)),
                log_loss=float(log_loss(y,p,labels=[0,1])))


def psi_numeric(train, test, bins=10):
    a = np.asarray(train, dtype=float); b = np.asarray(test, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 100 or len(b) < 100: return np.nan
    edges = np.unique(np.quantile(a, np.linspace(0,1,bins+1)))
    if len(edges)<3: return np.nan
    edges[0]=-np.inf; edges[-1]=np.inf
    pa = np.histogram(a,bins=edges)[0].astype(float)+0.5
    pb = np.histogram(b,bins=edges)[0].astype(float)+0.5
    pa/=pa.sum();pb/=pb.sum()
    return float(np.sum((pb-pa)*np.log(pb/pa)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,default=Path('data'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/phase5'))
    ap.add_argument('--train-cap',type=int,default=400000)
    ap.add_argument('--calibration-per-month',type=int,default=30000)
    ap.add_argument('--test-per-month',type=int,default=30000)
    ap.add_argument('--bootstrap-reps',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--models',choices=['both','logistic','xgboost'],default='both')
    args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    train_parts=[]; calib_parts=[]; test_parts=[]
    train_month_cap=max(1,int(np.ceil(args.train_cap/21)))
    for yr,mo,d in frames(args.input_dir):
        if yr==2022 or (yr==2023 and mo<=9):
            n=min(len(d),train_month_cap)
            train_parts.append(d.iloc[rng.choice(len(d),n,replace=False)].copy())
        elif yr==2023:
            n=min(len(d),args.calibration_per_month)
            calib_parts.append(d.iloc[rng.choice(len(d),n,replace=False)].copy())
        else:
            n=min(len(d),args.test_per_month)
            sampled=d.iloc[rng.choice(len(d),n,replace=False)].copy()
            sampled['year']=yr;sampled['month_num']=mo
            test_parts.append(sampled)
        print(f'Loaded {yr}-{mo:02d}',flush=True)
    train=pd.concat(train_parts,ignore_index=True)
    if len(train)>args.train_cap:
        train=train.sample(n=args.train_cap,random_state=args.seed)
    calib=pd.concat(calib_parts,ignore_index=True)
    tests=pd.concat(test_parts,ignore_index=True)
    print(f'Training {len(train):,}; calibration {len(calib):,}; testing {len(tests):,}',flush=True)
    summary=[]; monthly=[]; shift=[]; boot=[]; bins=[]
    for col in NUMS:
        for year in [2024,2025]:
            shift.append(dict(variable=col,year=year,psi=psi_numeric(train[col],tests.loc[tests.year==year,col])))
    names=['logistic','xgboost'] if args.models=='both' else [args.models]
    for name in names:
        model=make_model(name,args.seed)
        model.fit(train[FEATURES],train.target)
        pcal=model.predict_proba(calib[FEATURES])[:,1]
        platt=LogisticRegression(max_iter=300)
        platt.fit(logit(pcal).reshape(-1,1),calib.target.to_numpy())
        iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip')
        iso.fit(pcal,calib.target.to_numpy())
        raw=model.predict_proba(tests[FEATURES])[:,1]
        pred={'raw':raw,'platt':platt.predict_proba(logit(raw).reshape(-1,1))[:,1],
              'isotonic':iso.predict(raw)}
        y=tests.target.to_numpy()
        for method,p in pred.items():
            for year in [2024,2025]:
                mask=(tests.year==year).to_numpy()
                summary.append(dict(model=name,method=method,year=year,**stats(y[mask],p[mask])))
            for (yr,mo),idx in tests.groupby(['year','month_num']).indices.items():
                monthly.append(dict(model=name,method=method,year=int(yr),month=int(mo),**stats(y[idx],p[idx])))
            for year in [2024,2025]:
                mask=(tests.year==year).to_numpy()
                yy=y[mask];pp=p[mask]
                edges=np.linspace(0,1,11)
                ids=np.minimum(np.digitize(pp,edges,right=False)-1,9)
                for i in range(10):
                    m=ids==i
                    bins.append(dict(model=name,method=method,year=year,bin=i,
                                     n=int(m.sum()),mean_pred=float(pp[m].mean()) if m.any() else np.nan,
                                     observed=float(yy[m].mean()) if m.any() else np.nan))
        # Paired monthly-block bootstrap of yearly mean Brier difference (calibrated - raw).
        # Resample 12 months, keeping predictions on the same flights within each month.
        for year in [2024,2025]:
            ym=(tests.year==year).to_numpy()
            mm=tests.loc[ym,'month_num'].to_numpy()
            yy=y[ym]
            for method in ['platt','isotonic']:
                delta=(yy-pred[method][ym])**2-(yy-pred['raw'][ym])**2
                monthly_sums=np.array([delta[mm==m].sum() for m in range(1,13)])
                monthly_ns=np.array([(mm==m).sum() for m in range(1,13)])
                ix=rng.integers(0,12,size=(args.bootstrap_reps,12))
                draws=monthly_sums[ix].sum(axis=1)/monthly_ns[ix].sum(axis=1)
                boot.append(dict(model=name,year=year,method=method,
                                 brier_delta=float(delta.mean()),
                                 ci_low=float(np.quantile(draws,.025)),
                                 ci_high=float(np.quantile(draws,.975)),
                                 bootstrap_unit='calendar_month',reps=args.bootstrap_reps))
        print(f'Finished {name}',flush=True)
    pd.DataFrame(summary).to_csv(out/'calibration_annual_metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(out/'calibration_monthly_metrics.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'calibration_reliability_bins.csv',index=False)
    pd.DataFrame(shift).to_csv(out/'numeric_feature_psi.csv',index=False)
    pd.DataFrame(boot).to_csv(out/'calibration_brier_bootstrap.csv',index=False)
    (out/'run_config.json').write_text(json.dumps(vars(args),indent=2,default=str),encoding='utf-8')
    print('Done:',out)

if __name__=='__main__':
    main()
