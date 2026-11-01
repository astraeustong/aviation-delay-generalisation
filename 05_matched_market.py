import argparse, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

CATS=['Reporting_Airline','Origin','Dest']
NUMS=['Month','DayOfWeek','CRSDepTime','CRSArrTime','CRSElapsedTime','Distance']
KEEP=['FlightDate','ArrDelay','Cancelled','Diverted']+CATS+NUMS
TARGETS={'Reporting_Airline':['WN','DL','AA'],'Origin':['ATL','DFW','DEN']}

def load_archives(root,seed,train_cap,test_cap):
    rng=np.random.default_rng(seed)
    chunks=[]
    for year in range(2022,2026):
        zp=root/f'{year}.zip'
        if not zp.exists(): raise FileNotFoundError(zp)
        with zipfile.ZipFile(zp) as z:
            members=sorted(n for n in z.namelist() if n.lower().endswith('.parquet'))
            if len(members)!=12: raise ValueError(f'{zp}: expected 12 parquet members; got {len(members)}')
            for member in members:
                with z.open(member) as f:
                    frame=pd.read_parquet(f,columns=KEEP)
                frame['FlightDate']=pd.to_datetime(frame['FlightDate'],errors='coerce')
                frame=frame.loc[(frame.Cancelled==0)&(frame.Diverted==0)&frame.ArrDelay.notna()&frame.FlightDate.notna()].copy()
                frame['y']=(frame.ArrDelay>=15).astype('int8')
                frame=frame.drop(columns=['ArrDelay','Cancelled','Diverted'])
                # Fixed-size random per-month cap; all market groups remain eligible.
                cap=train_cap if year<=2023 else test_cap
                if len(frame)>cap:
                    frame=frame.sample(n=cap,random_state=int(rng.integers(2**31-1)))
                chunks.append(frame)
                print(f'{year} {member}: retained {len(frame):,}',flush=True)
    return pd.concat(chunks,ignore_index=True)

def pipeline(model,seed):
    prep=ColumnTransformer([
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='UNKNOWN')),('encode',OneHotEncoder(handle_unknown='ignore',min_frequency=100))]),CATS),
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMS)
    ])
    if model=='logistic': estimator=LogisticRegression(max_iter=300,solver='lbfgs')
    else: estimator=XGBClassifier(n_estimators=200,max_depth=5,learning_rate=.05,subsample=.85,colsample_bytree=.85,tree_method='hist',n_jobs=4,random_state=seed,eval_metric='logloss')
    return Pipeline([('preprocess',prep),('model',estimator)])

def metrics(y,p):
    return {'n':len(y),'prevalence':float(np.mean(y)),'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,'auprc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'calibration_bias':float(np.mean(p)-np.mean(y))}

def paired_boot(y,a,b,reps,seed):
    rng=np.random.default_rng(seed); n=len(y); rows=[]
    for _ in range(reps):
        ix=rng.integers(n,size=n)
        yy=y[ix]
        if len(np.unique(yy))<2: continue
        rows.append([roc_auc_score(yy,a[ix])-roc_auc_score(yy,b[ix]),brier_score_loss(yy,a[ix])-brier_score_loss(yy,b[ix])])
    if not rows:return {}
    z=np.asarray(rows)
    return {'auroc_diff_ci_low':float(np.quantile(z[:,0],.025)),'auroc_diff_ci_high':float(np.quantile(z[:,0],.975)),'brier_diff_ci_low':float(np.quantile(z[:,1],.025)),'brier_diff_ci_high':float(np.quantile(z[:,1],.975))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',default='data');ap.add_argument('--output-dir',default='outputs/phase4')
    ap.add_argument('--train-per-month',type=int,default=15000);ap.add_argument('--test-per-month',type=int,default=12000)
    ap.add_argument('--train-size',type=int,default=160000);ap.add_argument('--test-size',type=int,default=30000)
    ap.add_argument('--bootstrap-reps',type=int,default=200);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--models',choices=['logistic','xgboost','both'],default='both')
    args=ap.parse_args(); out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=load_archives(Path(args.input_dir),args.seed,args.train_per_month,args.test_per_month)
    train=df[df.FlightDate<'2023-10-01']; valid=df[(df.FlightDate>='2023-10-01')&(df.FlightDate<'2024-01-01')]
    test=df[df.FlightDate>='2024-01-01']
    rng=np.random.default_rng(args.seed); results=[]; ci=[]
    models=['logistic','xgboost'] if args.models=='both' else [args.models]
    for field,values in TARGETS.items():
        for value in values:
            # Included/excluded models trained on EQUAL counts, both tested on identical target rows.
            tr_in=train.copy(); tr_ex=train[train[field]!=value]
            target=test[test[field]==value]
            if len(target)>args.test_size: target=target.sample(args.test_size,random_state=args.seed)
            if len(target)<100 or target.y.nunique()<2: print('Skipping',field,value,'insufficient test');continue
            n=min(args.train_size,len(tr_ex),len(tr_in))
            if n<1000: print('Skipping',field,value,'insufficient training');continue
            tr_in=tr_in.sample(n,random_state=args.seed)
            tr_ex=tr_ex.sample(n,random_state=args.seed)
            # Validation used only for reporting; no test-based tuning.
            val=valid[valid[field]==value]
            if len(val)>args.test_size:val=val.sample(args.test_size,random_state=args.seed)
            for model in models:
                predictions={}
                for condition,tr in [('included',tr_in),('excluded',tr_ex)]:
                    fitted=pipeline(model,args.seed);fitted.fit(tr[CATS+NUMS],tr.y)
                    predictions[condition]=fitted.predict_proba(target[CATS+NUMS])[:,1]
                    for yr in [2024,2025]:
                        mask=(target.FlightDate.dt.year==yr).to_numpy()
                        if mask.sum()<30:continue
                        mm=metrics(target.y.to_numpy()[mask],predictions[condition][mask]);results.append(dict(market_type=field,market=value,model=model,condition=condition,year=yr,train_n=n,**mm))
                yy=target.y.to_numpy(); a=predictions['included']; b=predictions['excluded']
                for yr in [2024,2025]:
                    mask=(target.FlightDate.dt.year==yr).to_numpy()
                    if mask.sum()<30:continue
                    stats=paired_boot(yy[mask],a[mask],b[mask],args.bootstrap_reps,args.seed)
                    ci.append(dict(market_type=field,market=value,model=model,year=yr,comparison='included_minus_excluded',**stats))
                print('Completed',field,value,model,flush=True)
    pd.DataFrame(results).to_csv(out/'matched_market_metrics.csv',index=False)
    pd.DataFrame(ci).to_csv(out/'matched_market_bootstrap.csv',index=False)
    (out/'run_config.json').write_text(json.dumps(vars(args),indent=2))
    print('Saved outputs to',out)
if __name__=='__main__':main()
