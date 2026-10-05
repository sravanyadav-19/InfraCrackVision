"""Build the first four-class metadata table.

Cutoffs remain anchored to DeepCrack training ground-truth masks. SDNET
noncracked images become no-crack; only quality-filtered SDNET pseudo-masks
become proxy minor/moderate/severe records.
"""
from pathlib import Path
import json,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'; classes=['no-crack','minor','moderate','severe']
def main():
    meta=pd.read_csv(OUT/'metadata.csv'); pseudo=pd.read_csv(OUT/'sdnet_pseudo_labels_filtered.csv'); cfg=json.load(open(OUT/'label_config.json')); q1,q2=cfg['cutoffs']
    rows=[]
    for _,r in meta[meta.source_dataset=='deepcrack'].iterrows():
        ratio=float(r.crack_pixel_ratio); label='minor' if ratio<=q1 else ('moderate' if ratio<=q2 else 'severe'); r=r.to_dict();r['severity_class']=label;rows.append(r)
    for _,r in meta[(meta.source_dataset=='sdnet2018')&(meta.has_crack==0)].iterrows():
        r=r.to_dict();r['severity_class']='no-crack';rows.append(r)
    for _,r in pseudo[pseudo.quality_flag=='usable'].iterrows():
        ratio=float(r.crack_pixel_ratio); label='minor' if ratio<=q1 else ('moderate' if ratio<=q2 else 'severe'); rows.append({'filename':r.filename,'source_dataset':'sdnet2018_pseudo','image_path':f'datasets/sdnet2018/cracked/{r.filename}','mask_path':r.mask_path,'has_crack':1,'mask_available':1,'crack_pixel_ratio':ratio,'severity_class':label,'source_scene_id':f'sdnet_pseudo_{Path(r.filename).stem}','split':'unassigned'})
    df=pd.DataFrame(rows);df.to_csv(OUT/'four_class_metadata.csv',index=False);print(f'Created {OUT}/four_class_metadata.csv with {len(df)} records');print(df.severity_class.value_counts().reindex(classes,fill_value=0).to_string());print('cutoffs anchored to DeepCrack:',q1,q2)
if __name__=='__main__':main()
