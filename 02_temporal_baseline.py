"""Leakage-safe temporal logistic baseline with chronological train/validation/test.
Requires: pandas numpy pyarrow scikit-learn matplotlib
Run: python 02_temporal_baseline.py --input-dir data --output-dir outputs
Train 2022-23, validate 2024, test 2025. Model is fit on 2022-23 only.
Uses a stratified capped training sample to fit on ordinary laptops; evaluation is full year.
"""
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
from sklearn.metrics import (average_precision_score,roc_auc_score,brier_score_loss,
                             log_loss,precision_score,recall_score,f1_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

CATS=['Reporting_Airline','Origin','Dest','DayOfWeek','Month']
NUM=['CRSDepTime','CRSArrTime','CRSElapsedTime','Distance']
NEEDED=CATS+NUM+['ArrDelay','Cancelled','Diverted','FlightDate']

def load_year(path):
    chunks=[]
    with zipfile.ZipFile(path) as z:
        names=sorted(n for n in z.namelist() if n.endswith('.parquet'))
        for name in names:
            with z.open(name) as f:
                d=pd.read_parquet(io.BytesIO(f.read()),columns=NEEDED)
            eligible=(pd.to_numeric(d.Cancelled,errors='coerce').eq(0)&
                      pd.to_numeric(d.Diverted,errors='coerce').eq(0)&
                      pd.to_numeric(d.ArrDelay,errors='coerce').notna())
            d=d.loc[eligible,CATS+NUM+['ArrDelay']].copy()
            d['target']=(pd.to_numeric(d.ArrDelay)>=15).astype('int8')
            d.drop(columns='ArrDelay',inplace=True)
            for col in CATS:d[col]=d[col].astype('string').fillna('__MISSING__').astype(str)
            for col in NUM:d[col]=pd.to_numeric(d[col],errors='coerce').astype('float32')
            chunks.append(d)
    return pd.concat(chunks,ignore_index=True)

def evaluate(model,d,name):
    y=d.target.to_numpy();p=model.predict_proba(d[CATS+NUM])[:,1]
    pred=p>=.5
    return {'period':name,'n':len(d),'prevalence':float(y.mean()),
            'AUROC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),
            'Brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p)),
            'precision_05':float(precision_score(y,pred,zero_division=0)),
            'recall_05':float(recall_score(y,pred,zero_division=0)),
            'F1_05':float(f1_score(y,pred,zero_division=0))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,default=Path('data'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs'))
    ap.add_argument('--train-cap',type=int,default=400000)
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    from sklearn.model_selection import train_test_split
    train=pd.concat([load_year(args.input_dir/f'{y}.zip') for y in (2022,2023)],ignore_index=True)
    if len(train)>args.train_cap:
        train,_=train_test_split(train,train_size=args.train_cap,stratify=train.target,random_state=42)
    # Explicitly fit preprocessing on training data only. No actual departure or postarrival fields.
    preprocess=ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=100),CATS),
        ('num',Pipeline([('impute',SimpleImputer(strategy='median')),
                         ('scale',StandardScaler())]),NUM)])
    model=Pipeline([('prep',preprocess),('clf',LogisticRegression(max_iter=500,solver='lbfgs'))])
    model.fit(train[CATS+NUM],train.target)
    results=[evaluate(model,train,'training_sample_2022_2023')]
    del train
    for year in (2024,2025):
        d=load_year(args.input_dir/f'{year}.zip')
        results.append(evaluate(model,d,str(year)))
        del d
    result=pd.DataFrame(results)
    result.to_csv(args.output_dir/'temporal_logistic_baseline.csv',index=False)
    print(result.to_string(index=False))
    print('WARNING: 2024 is an evaluation year, not used for model selection in this script.')
if __name__=='__main__':main()
