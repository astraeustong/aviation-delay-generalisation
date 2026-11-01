"""BTS aviation 2022-2025: reproducible flight-level audit and EDA.
Run: python 01_audit_and_eda.py --input-dir PATH_TO_YEAR_ZIPS --output-dir outputs
Requires: pandas numpy pyarrow matplotlib scipy
"""
import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

FIELDS = ['FlightDate','Year','Month','DayofMonth','DayOfWeek','Reporting_Airline',
          'Flight_Number_Reporting_Airline','Tail_Number','Origin','Dest',
          'OriginAirportID','DestAirportID','CRSDepTime','CRSArrTime',
          'CRSElapsedTime','Distance','ArrDelay','ArrDelayMinutes','ArrDel15',
          'Cancelled','Diverted','DepTime','DepDelay','ActualElapsedTime',
          'CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay']
PREDEP = ['FlightDate','Reporting_Airline','Origin','Dest','CRSDepTime',
          'CRSArrTime','CRSElapsedTime','Distance','DayOfWeek']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,default=Path('data'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs'))
    args=ap.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    files=sorted(args.input_dir.glob('20??.zip'))
    if not files: raise FileNotFoundError(f'No year ZIP archives in {args.input_dir}')
    audits=[]; monthly=[]; airlines=[]; origins=[]; missing=[]; yearly=[]
    seen=set()
    for zpath in files:
        with zipfile.ZipFile(zpath) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.parquet')]
            print(zpath.name, len(names),'monthly files',flush=True)
            for name in sorted(names):
                with z.open(name) as f:
                    df=pd.read_parquet(io.BytesIO(f.read()))
                absent=sorted(set(FIELDS)-set(df.columns))
                if absent: raise ValueError(f'{name}: missing columns {absent}')
                dates=pd.to_datetime(df['FlightDate'],errors='coerce')
                year=int(zpath.stem)
                months=dates.dt.month.dropna().unique()
                if len(months)!=1 or not (dates.dt.year.dropna()==year).all():
                    raise ValueError(f'{name}: unexpected year/month in FlightDate')
                month=int(months[0]); key=(year,month)
                if key in seen: raise ValueError(f'Duplicate monthly partition {key}')
                seen.add(key)
                cancelled=pd.to_numeric(df['Cancelled'],errors='coerce').eq(1)
                diverted=pd.to_numeric(df['Diverted'],errors='coerce').eq(1)
                delay=pd.to_numeric(df['ArrDelay'],errors='coerce')
                eligible=(~cancelled)&(~diverted)&delay.notna()
                late=delay.ge(15)&eligible
                # Compare BTS ArrDel15 with derived target, when populated.
                label=pd.to_numeric(df['ArrDel15'],errors='coerce')
                disagree=int((eligible&label.notna()&(label.ne(delay.ge(15).astype(int)))).sum())
                # Exact duplicate detection within monthly partition only; do not delete automatically.
                dup=int(df.duplicated().sum())
                row={'year':year,'month':month,'rows':len(df),'eligible':int(eligible.sum()),
                     'late15':int(late.sum()),'cancelled':int(cancelled.sum()),
                     'diverted':int(diverted.sum()),'unknown_arrival':int((~cancelled&~diverted&delay.isna()).sum()),
                     'duplicates_full_row':dup,'target_disagreements':disagree,
                     'delay_rate':float(late.sum()/eligible.sum()) if eligible.sum() else np.nan,
                     'cancellation_rate':float(cancelled.mean())}
                audits.append(row); monthly.append(row)
                for col in FIELDS:
                    missing.append({'year':year,'month':month,'column':col,'missing':int(df[col].isna().sum()),'rows':len(df)})
                # Counts are aggregated within a month; no raw personal or tail-number records exported.
                temp=df.loc[eligible,['Reporting_Airline','Origin']].copy()
                temp['late15']=late.loc[eligible].astype('int8').values
                ag=temp.groupby('Reporting_Airline',dropna=False)['late15'].agg(['sum','count']).reset_index()
                ag['year']=year;ag['month']=month;airlines.append(ag)
                og=temp.groupby('Origin',dropna=False)['late15'].agg(['sum','count']).reset_index()
                og['year']=year;og['month']=month;origins.append(og)
                del df
    expected={(y,m) for y in range(2022,2026) for m in range(1,13)}
    absent=sorted(expected-seen)
    audit=pd.DataFrame(audits).sort_values(['year','month'])
    audit.to_csv(out/'monthly_audit.csv',index=False)
    pd.concat(airlines,ignore_index=True).to_csv(out/'airline_monthly_counts.csv',index=False)
    pd.concat(origins,ignore_index=True).to_csv(out/'airport_monthly_counts.csv',index=False)
    miss=pd.DataFrame(missing)
    miss.groupby('column',as_index=False)[['missing','rows']].sum().assign(
        missing_pct=lambda x:100*x['missing']/x['rows']).sort_values('missing_pct',ascending=False).to_csv(out/'missingness.csv',index=False)
    annual=audit.groupby('year',as_index=False)[['rows','eligible','late15','cancelled','diverted','unknown_arrival','duplicates_full_row','target_disagreements']].sum()
    annual['delay_rate']=annual['late15']/annual['eligible']
    annual['cancellation_rate']=annual['cancelled']/annual['rows']
    annual.to_csv(out/'annual_summary.csv',index=False)
    # Descriptive comparison only: huge N and clustering mean simple chi-square is not a causal test.
    contingency=np.column_stack([annual['late15'],annual['eligible']-annual['late15']])
    chi2,p,dof,_=chi2_contingency(contingency)
    (out/'statistical_notes.txt').write_text(
        f'Pearson chi-square of annual delay prevalence: chi2={chi2:.3f}, dof={dof}, p={p:.6g}.\n'
        'Descriptive association only. Flights are dependent within airport, airline and date; '
        'do not interpret as a causal test or independent-flight inference.\n'
        f'Missing year-month partitions: {absent}\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(pd.to_datetime(dict(year=audit.year,month=audit.month,day=1)),100*audit.delay_rate,marker='.',linewidth=1)
    ax.set(ylabel='Arrivals >=15 min late (%)',xlabel='Month',title='US reporting-carrier arrival delays (eligible flights)')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'figures'/'monthly_delay_rate.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.bar(annual.year.astype(str),100*annual.delay_rate)
    ax.set(ylabel='Arrivals >=15 min late (%)',xlabel='Year',title='Annual arrival delay prevalence')
    fig.tight_layout();fig.savefig(out/'figures'/'annual_delay_rate.png',dpi=180);plt.close(fig)
    print('\nANNUAL SUMMARY\n',annual.to_string(index=False))
    print('Missing partitions:',absent)
    print('Outputs:',out.resolve())

if __name__=='__main__': main()
