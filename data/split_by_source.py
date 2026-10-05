"""Create a reproducible source-aware train/validation/test split.
DeepCrack's original test set remains test-only. SDNET records are split 70/15/15.
"""
from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]; CLASSES=['no-crack','minor','moderate','severe']
def split_three(df,seed):
    if len(df)<3: return df.assign(split='train')
    try: tr,tmp=train_test_split(df,test_size=.30,random_state=seed,stratify=df.severity_class)
    except ValueError: tr,tmp=train_test_split(df,test_size=.30,random_state=seed)
    try: va,te=train_test_split(tmp,test_size=.50,random_state=seed,stratify=tmp.severity_class)
    except ValueError: va,te=train_test_split(tmp,test_size=.50,random_state=seed)
    for x,s in [(tr,'train'),(va,'validation'),(te,'test')]: x['split']=s
    return pd.concat([tr,va,te])
def split_two(df,seed):
    if len(df)<2 or df.severity_class.nunique()<2: return df.assign(split='train')
    try: tr,va=train_test_split(df,test_size=.20,random_state=seed,stratify=df.severity_class)
    except ValueError: tr,va=train_test_split(df,test_size=.20,random_state=seed)
    tr['split']='train';va['split']='validation';return pd.concat([tr,va])
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);a=p.parse_args(); src=ROOT/'results'/'four_class_metadata.csv';out=ROOT/'results'/'splits.csv'
    if not src.exists(): raise FileNotFoundError('Run data/build_four_class_metadata.py first')
    df=pd.read_csv(src);df=df[df.severity_class.isin(CLASSES)].copy();pieces=[]
    dc_test=df[(df.source_dataset=='deepcrack')&(df.split=='test')].copy();dc_test['split']='test';pieces.append(dc_test)
    dc_train=df[(df.source_dataset=='deepcrack')&(df.split=='train')].copy();pieces.append(split_two(dc_train,a.seed))
    other=df[df.source_dataset!='deepcrack'].copy();pieces.append(split_three(other,a.seed))
    result=pd.concat(pieces,ignore_index=True);result.to_csv(out,index=False);print(f'Created {out} with {len(result)} records');print(pd.crosstab(result.split,result.severity_class).reindex(columns=CLASSES,fill_value=0));print(pd.crosstab(result.split,result.source_dataset));a=set(result[(result.source_dataset=='deepcrack')&(result.split=='test')].image_path);b=set(result[result.split.isin(['train','validation'])].image_path);print('DeepCrack test isolation:', 'PASSED' if not a&b else 'FAILED');
    if any((result[result.split==s].severity_class.nunique()<4) for s in ['train','validation','test']): print('WARNING: one split lacks a class')
if __name__=='__main__':main()
