"""Phase 6A: leakage-safe calibration-window sensitivity.
Run from the aviation project root. Four yearly ZIPs in --input-dir.
Train Jan 2022-Jun 2023; calibrate Jul-Dec 2023 or Oct-Dec 2023;
evaluate 2024-2025. All models use identical training and test samples per seed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from scipy.stats import spearmanr
from importlib.machinery import SourceFileLoader

# Reuse the Phase 5 input reader and model specification for comparability.

def load_phase5(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing Phase 5 script: {path}. Put it beside this script or specify --phase5-script.')
    return SourceFileLoader('aviation_phase5_shared', str(path.resolve())).load_module()

def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))

def metric(y,p):
    p=np.clip(np.asarray(p),1e-8,1-1e-8)
    return dict(n=int(len(y)), prevalence=float(np.mean(y)), mean_pred=float(np.mean(p)),
                bias=float(np.mean(p)-np.mean(y)), brier=float(brier_score_loss(y,p)),
                log_loss=float(log_loss(y,p,labels=[0,1])),
                auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
                auprc=float(average_precision_score(y,p)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,default=Path('data'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/phase6'))
    ap.add_argument('--phase5-script',type=Path,default=Path('06_calibration_robustness.py'))
    ap.add_argument('--train-cap',type=int,default=300000)
    ap.add_argument('--calibration-per-month',type=int,default=15000)
    ap.add_argument('--test-per-month',type=int,default=15000)
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,123,2026])
    ap.add_argument('--bootstrap-reps',type=int,default=1000)
    ap.add_argument('--models',choices=['both','logistic','xgboost'],default='both')
    a=ap.parse_args()
    if min(a.train_cap,a.calibration_per_month,a.test_per_month,a.bootstrap_reps)<1: ap.error('Sample sizes and bootstrap reps must be positive')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    phase5=load_phase5(a.phase5_script)
    # Keep each sampled month once in memory; different seeds resample each month.
    months={(year,month):df for year,month,df in phase5.frames(a.input_dir)}
    if len(months)!=48: raise ValueError(f'Expected 48 months, got {len(months)}')
    rows=[]; monthly=[]; uncertainty=[]; samples=[]
    names=['logistic','xgboost'] if a.models=='both' else [a.models]
    for seed in a.seeds:
        rng=np.random.default_rng(seed)
        train=[];cal=[];tests=[]
        # 18 training months. No calibration data are used to fit the predictive model.
        for (yr,mo),df in sorted(months.items()):
            if yr==2022 or (yr==2023 and mo<=6):
                cap=int(np.ceil(a.train_cap/18))
                n=min(len(df),cap)
                train.append(df.iloc[rng.choice(len(df),n,replace=False)].copy())
            elif yr==2023:
                n=min(len(df),a.calibration_per_month)
                d=df.iloc[rng.choice(len(df),n,replace=False)].copy()
                d['year']=yr;d['month_num']=mo
                cal.append(d)
            else:
                n=min(len(df),a.test_per_month)
                d=df.iloc[rng.choice(len(df),n,replace=False)].copy()
                d['year']=yr;d['month_num']=mo
                tests.append(d)
        train=pd.concat(train,ignore_index=True)
        if len(train)>a.train_cap: train=train.sample(a.train_cap,random_state=seed)
        cal=pd.concat(cal,ignore_index=True)
        tests=pd.concat(tests,ignore_index=True)
        y=tests.target.to_numpy(dtype=int)
        samples.append(dict(seed=seed,train_n=len(train),cal_h2_n=len(cal),cal_q4_n=int((cal.month_num>=10).sum()),test_n=len(tests)))
        for name in names:
            model=phase5.make_model(name,seed)
            model.fit(train[phase5.FEATURES],train.target)
            p_cal=model.predict_proba(cal[phase5.FEATURES])[:,1]
            p_test=model.predict_proba(tests[phase5.FEATURES])[:,1]
            predictions={'raw':p_test}
            for window,mask in [('q4',cal.month_num>=10),('h2',cal.month_num>=7)]:
                yy=cal.loc[mask,'target'].to_numpy(dtype=int)
                pp=p_cal[mask.to_numpy()]
                platt=LogisticRegression(max_iter=300)
                platt.fit(logit(pp).reshape(-1,1),yy)
                predictions[f'platt_{window}']=platt.predict_proba(logit(p_test).reshape(-1,1))[:,1]
                iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip')
                iso.fit(pp,yy)
                predictions[f'isotonic_{window}']=iso.predict(p_test)
            for year in (2024,2025):
                idx=np.flatnonzero(tests.year.to_numpy()==year)
                yy=y[idx]
                mm=tests.month_num.to_numpy()[idx]
                for method,p in predictions.items():
                    pp=p[idx]
                    rows.append(dict(seed=seed,model=name,year=year,method=method,**metric(yy,pp)))
                    for m in range(1,13):
                        mi=mm==m
                        monthly.append(dict(seed=seed,model=name,year=year,month=m,method=method,**metric(yy[mi],pp[mi])))
                    if method=='raw': continue
                    delta=(yy-pp)**2-(yy-predictions['raw'][idx])**2
                    sums=np.array([delta[mm==m].sum() for m in range(1,13)])
                    ns=np.array([(mm==m).sum() for m in range(1,13)])
                    # Month-block bootstrap, descriptive CI; 12 independent time blocks is a limitation.
                    draw=rng.integers(0,12,size=(a.bootstrap_reps,12))
                    vals=sums[draw].sum(axis=1)/ns[draw].sum(axis=1)
                    uncertainty.append(dict(seed=seed,model=name,year=year,method=method,
                                            delta_brier=float(delta.mean()),ci_low=float(np.quantile(vals,.025)),
                                            ci_high=float(np.quantile(vals,.975)),blocks=12,reps=a.bootstrap_reps))
            print(f'Completed seed={seed} model={name}',flush=True)
    pd.DataFrame(rows).to_csv(a.output_dir/'calibration_window_annual.csv',index=False)
    pd.DataFrame(monthly).to_csv(a.output_dir/'calibration_window_monthly.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(a.output_dir/'calibration_window_bootstrap.csv',index=False)
    pd.DataFrame(samples).to_csv(a.output_dir/'sample_counts.csv',index=False)
    (a.output_dir/'run_config.json').write_text(json.dumps({**vars(a),'input_dir':str(a.input_dir),
        'output_dir':str(a.output_dir),'phase5_script':str(a.phase5_script)},indent=2),encoding='utf-8')
    print('Finished:',a.output_dir,flush=True)
if __name__=='__main__': main()
