"""Chronological logistic/XGBoost study from yearly ZIPs of monthly Parquet files.
Run: python 03_generalisation_experiments.py --input-dir data --output-dir outputs/phase2
"""
import argparse
import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, precision_recall_curve

CATS = ['Reporting_Airline','Origin','Dest','DayOfWeek','Month']
NUMS = ['CRSDepTime','CRSArrTime','CRSElapsedTime','Distance']
FEATURES = CATS + NUMS
NEEDED = FEATURES + ['ArrDelay','Cancelled','Diverted','FlightDate']
SEED = 42

def monthly_frames(input_dir):
    for year in range(2022,2026):
        path = input_dir / f'{year}.zip'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}')
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith('.parquet')]
            months = {}
            for name in names:
                match = re.search(rf'_{year}_(\d{{1,2}})\.parquet$',name)
                if not match: raise ValueError(f'Unexpected monthly filename: {name}')
                month = int(match.group(1))
                if month in months: raise ValueError(f'Duplicate month {year}-{month}')
                months[month] = name
            if set(months) != set(range(1,13)):
                raise ValueError(f'Expected 12 months in {path}, found {sorted(months)}')
            for month in range(1,13):
                with z.open(months[month]) as stream:
                    frame = pd.read_parquet(io.BytesIO(stream.read()),columns=NEEDED)
                valid = (pd.to_numeric(frame['Cancelled'],errors='coerce').eq(0) &
                         pd.to_numeric(frame['Diverted'],errors='coerce').eq(0) &
                         pd.to_numeric(frame['ArrDelay'],errors='coerce').notna())
                d = frame.loc[valid,FEATURES+['ArrDelay','FlightDate']].copy()
                d['target'] = (pd.to_numeric(d['ArrDelay']) >= 15).astype('int8')
                d.drop(columns='ArrDelay',inplace=True)
                for c in CATS: d[c] = d[c].astype('string').fillna('__MISSING__').astype(str)
                for c in NUMS: d[c] = pd.to_numeric(d[c],errors='coerce').astype('float32')
                d['FlightDate'] = pd.to_datetime(d['FlightDate'],errors='raise')
                if not d['FlightDate'].dt.year.eq(year).all() or not d['FlightDate'].dt.month.eq(month).all():
                    raise ValueError(f'Date mismatch in {year}-{month}')
                yield year,month,d

def metrics(y,p):
    return {'n':len(y),'prevalence':float(y.mean()),'AUROC':float(roc_auc_score(y,p)),
            'AUPRC':float(average_precision_score(y,p)),
            'Brier':float(brier_score_loss(y,p)),
            'log_loss':float(log_loss(y,p,labels=[0,1]))}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,default=Path('data'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/phase2'))
    parser.add_argument('--train-cap',type=int,default=600000,
                        help='Maximum training rows; sampled proportionally per training month')
    parser.add_argument('--bootstrap-reps',type=int,default=200)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    # Two passes: exact training-month sizes, then stratified-by-month random samples.
    counts={}
    for yr,mo,d in monthly_frames(args.input_dir):
        if (yr==2022) or (yr==2023 and mo<=9): counts[(yr,mo)] = len(d)
        del d
    total=sum(counts.values())
    train_parts=[]
    for yr,mo,d in monthly_frames(args.input_dir):
        if (yr,mo) in counts:
            take=min(len(d),max(1,round(args.train_cap*len(d)/total)))
            selected=rng.choice(len(d),size=take,replace=False)
            train_parts.append(d.iloc[selected][FEATURES+['target']].copy())
        del d
    train=pd.concat(train_parts,ignore_index=True)
    del train_parts
    print('Training rows:',len(train),'training prevalence:',round(train.target.mean(),4),flush=True)
    prep=ColumnTransformer([
        ('categorical',OneHotEncoder(handle_unknown='ignore',min_frequency=100),CATS),
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),
                             ('scaler',StandardScaler())]),NUMS)
    ])
    logistic=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=500))])
    logistic.fit(train[FEATURES],train.target)
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError('Install xgboost: python -m pip install xgboost') from exc
    # Use same encoded feature space; fit preprocessing only on training data.
    prep_xgb=ColumnTransformer([
        ('categorical',OneHotEncoder(handle_unknown='ignore',min_frequency=100),CATS),
        ('numeric',SimpleImputer(strategy='median'),NUMS)
    ])
    xgb=Pipeline([('prep',prep_xgb),('model',XGBClassifier(
        n_estimators=350,max_depth=5,learning_rate=0.05,subsample=0.85,
        colsample_bytree=0.85,tree_method='hist',eval_metric='logloss',
        n_jobs=4,random_state=SEED))])
    xgb.fit(train[FEATURES],train.target)
    del train
    models={'logistic':logistic,'xgboost':xgb}
    monthly=[]
    annual_predictions={m:{2024:[],2025:[]} for m in models}
    validation={m:[] for m in models}
    for yr,mo,d in monthly_frames(args.input_dir):
        if yr==2022 or (yr==2023 and mo<=9):
            continue
        y=d.target.to_numpy(dtype=np.int8)
        for name,model in models.items():
            p=model.predict_proba(d[FEATURES])[:,1].astype(np.float32)
            result=metrics(y,p)
            result.update(model=name,year=yr,month=mo,period=('validation' if yr==2023 else 'test'))
            monthly.append(result)
            if yr==2023: validation[name].append((y,p))
            else: annual_predictions[name][yr].append((y,p))
        print(f'Finished {yr}-{mo:02d}: {len(d):,} eligible flights',flush=True)
        del d
    pd.DataFrame(monthly).to_csv(args.output_dir/'monthly_model_metrics.csv',index=False)
    annual=[]
    calibration=[]
    curves={}
    for name in models:
        for yr,blocks in [(2023,validation[name]),(2024,annual_predictions[name][2024]),
                          (2025,annual_predictions[name][2025])]:
            y=np.concatenate([a for a,b in blocks]); p=np.concatenate([b for a,b in blocks])
            row=metrics(y,p)
            row.update(model=name,year=yr,period=('validation_oct_dec' if yr==2023 else 'test'))
            annual.append(row)
            if yr==2023:
                precision,recall,thresholds=precision_recall_curve(y,p)
                f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
                idx=int(np.argmax(f1))
                curves[name]=float(thresholds[idx])
            # Equal-width calibration bins, aggregated from all predictions.
            edges=np.linspace(0,1,11)
            bins=np.clip(np.digitize(p,edges,right=False)-1,0,9)
            for b in range(10):
                mask=bins==b
                if mask.any(): calibration.append(dict(model=name,year=yr,bin=b,
                    n=int(mask.sum()),mean_prediction=float(p[mask].mean()),
                    observed_rate=float(y[mask].mean())))
            del y,p
    annual_df=pd.DataFrame(annual)
    annual_df.to_csv(args.output_dir/'annual_model_metrics.csv',index=False)
    pd.DataFrame(calibration).to_csv(args.output_dir/'calibration_bins.csv',index=False)
    with open(args.output_dir/'validation_thresholds.json','w') as f: json.dump(curves,f,indent=2)
    # Paired month-block bootstrap: compare annual AUROC, resampling whole months.
    boot=[]
    for name in models:
        # Cap each monthly block for feasible bootstrap runtime on laptops.
        def subsample_blocks(blocks, cap=25000):
            result=[]
            for yy,pp in blocks:
                idx=rng.choice(len(yy),size=min(cap,len(yy)),replace=False)
                result.append((yy[idx],pp[idx]))
            return result
        blocks24=subsample_blocks(annual_predictions[name][2024])
        blocks25=subsample_blocks(annual_predictions[name][2025])
        for rep in range(args.bootstrap_reps):
            i24=rng.integers(0,12,12); i25=rng.integers(0,12,12)
            y24=np.concatenate([blocks24[i][0] for i in i24]); p24=np.concatenate([blocks24[i][1] for i in i24])
            y25=np.concatenate([blocks25[i][0] for i in i25]); p25=np.concatenate([blocks25[i][1] for i in i25])
            boot.append(dict(model=name,rep=rep,auroc_2025_minus_2024=
                             roc_auc_score(y25,p25)-roc_auc_score(y24,p24)))
    boot_df=pd.DataFrame(boot)
    ci=boot_df.groupby('model')['auroc_2025_minus_2024'].agg(
        estimate_bootstrap_mean='mean',ci_lower=lambda s:s.quantile(.025),
        ci_upper=lambda s:s.quantile(.975)).reset_index()
    ci.to_csv(args.output_dir/'auroc_change_bootstrap_ci.csv',index=False)
    # Figures
    monthly_df=pd.DataFrame(monthly)
    for name in models:
        sub=monthly_df[(monthly_df.model==name)&(monthly_df.year>=2024)].copy()
        sub['date']=pd.to_datetime(dict(year=sub.year,month=sub.month,day=1))
        plt.figure(figsize=(10,4))
        plt.plot(sub.date,sub.AUROC,marker='o')
        plt.xlabel('Evaluation month');plt.ylabel('AUROC')
        plt.title(f'Monthly out-of-time discrimination: {name}')
        plt.grid(alpha=.25);plt.tight_layout()
        plt.savefig(args.output_dir/f'monthly_auroc_{name}.png',dpi=180);plt.close()
        fig,ax=plt.subplots(figsize=(5,5))
        for yr in [2024,2025]:
            s=pd.DataFrame(calibration)
            s=s[(s.model==name)&(s.year==yr)]
            ax.plot(s.mean_prediction,s.observed_rate,marker='o',label=str(yr))
        ax.plot([0,1],[0,1],linestyle='--',color='gray')
        ax.set(xlabel='Mean predicted probability',ylabel='Observed delay frequency',
               title=f'Calibration: {name}',xlim=(0,1),ylim=(0,1))
        ax.legend();fig.tight_layout()
        fig.savefig(args.output_dir/f'calibration_{name}.png',dpi=180);plt.close(fig)
    print('\nAnnual metrics:\n',annual_df.to_string(index=False))
    print('\nAUROC change bootstrap intervals:\n',ci.to_string(index=False))
    print('Outputs saved in:',args.output_dir.resolve())

if __name__=='__main__': main()
