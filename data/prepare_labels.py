"""Create labels from unified metadata without inventing severity for unmasked cracked images."""
from pathlib import Path
import json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; META=ROOT/'results'/'metadata.csv'; OUT=ROOT/'results'/'labels.csv'; CONFIG=ROOT/'results'/'label_config.json'; CLASSES=['no-crack','minor','moderate','severe']
def main():
    df=pd.read_csv(META)
    if df.empty: raise ValueError('metadata.csv is empty')
    mask=(df['split']=='train')&(df['mask_available']==1)&(df['has_crack']==1)
    ratios=pd.to_numeric(df.loc[mask,'crack_pixel_ratio']).dropna(); cuts=np.quantile(ratios,[1/3,2/3]).tolist()
    def label(row):
        if row['source_dataset']=='sdnet2018' and row['has_crack']==1: return 'severity_pending'
        r=float(row['crack_pixel_ratio']) if pd.notna(row['crack_pixel_ratio']) else 0.0
        if int(row['has_crack'])==0 or r<=0: return 'no-crack'
        return 'minor' if r<=cuts[0] else ('moderate' if r<=cuts[1] else 'severe')
    df['severity_class']=df.apply(label,axis=1); df.to_csv(OUT,index=False); json.dump({'classes':CLASSES,'cutoffs':cuts,'unmasked_cracked_policy':'severity_pending'},open(CONFIG,'w'),indent=2)
    print(f'Created {OUT} with {len(df)} records'); print(pd.crosstab(df['source_dataset'],df['severity_class'],dropna=False).to_string()); print(f'cutoffs: minor <= {cuts[0]:.6f}; moderate <= {cuts[1]:.6f}; severe > {cuts[1]:.6f}')
if __name__=='__main__': main()
