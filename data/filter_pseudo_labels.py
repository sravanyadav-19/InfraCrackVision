"""Filter SDNET pseudo-mask ratios before using them as proxy severity labels.
Original pseudo masks are never deleted. Extreme predictions are flagged.
"""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results'/'sdnet_pseudo_labels.csv'; OUT=ROOT/'results'/'sdnet_pseudo_labels_filtered.csv'

def main():
    df=pd.read_csv(SRC)
    df['quality_flag']='usable'
    df.loc[df.crack_pixel_ratio<=0,'quality_flag']='empty_prediction'
    df.loc[df.crack_pixel_ratio>0.5,'quality_flag']='extreme_prediction'
    df.to_csv(OUT,index=False)
    print(f'Created {OUT} with {len(df)} records')
    print(df['quality_flag'].value_counts().to_string())
    print('Usable nonzero pseudo-masks:', int(((df.quality_flag=='usable')&(df.crack_pixel_ratio>0)).sum()))
    print('Original masks remain in:', ROOT/'datasets/sdnet2018/pseudo_masks')
if __name__=='__main__': main()
