"""Build unified metadata for DeepCrack and flat SDNET2018 folders."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'datasets'; OUT=ROOT/'results'/'metadata.csv'
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
def files(folder): return sorted([p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTS],key=lambda p:p.name)
def add_deepcrack(rows,split):
    idir=DATA/'deepcrack'/'images'/split; mdir=DATA/'deepcrack'/'masks'/split; masks={p.stem:p for p in files(mdir)}
    for p in files(idir):
        m=masks.get(p.stem)
        if m is None: raise FileNotFoundError(f'No mask for {p.name}')
        ratio=float((np.asarray(Image.open(m).convert('L'))>127).mean())
        rows.append({'filename':p.name,'source_dataset':'deepcrack','image_path':p.relative_to(ROOT).as_posix(),'mask_path':m.relative_to(ROOT).as_posix(),'has_crack':int(ratio>0),'mask_available':1,'crack_pixel_ratio':ratio,'severity_class':'','source_scene_id':f'deepcrack_{p.stem}','split':split})
def add_sdnet(rows):
    for label in ['noncracked','cracked']:
        for p in files(DATA/'sdnet2018'/label):
            rows.append({'filename':p.name,'source_dataset':'sdnet2018','image_path':p.relative_to(ROOT).as_posix(),'mask_path':'','has_crack':int(label=='cracked'),'mask_available':0,'crack_pixel_ratio':0.0 if label=='noncracked' else np.nan,'severity_class':'no-crack' if label=='noncracked' else 'severity_pending','source_scene_id':f'sdnet2018_{label}_{p.stem}','split':'unassigned'})
def main():
    rows=[]; add_deepcrack(rows,'train'); add_deepcrack(rows,'test'); add_sdnet(rows)
    cols=['filename','source_dataset','image_path','mask_path','has_crack','mask_available','crack_pixel_ratio','severity_class','source_scene_id','split']
    OUT.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows,columns=cols).to_csv(OUT,index=False)
    df=pd.DataFrame(rows); print(f'Created {OUT} with {len(df)} records'); print(df.groupby(['source_dataset','severity_class'],dropna=False).size().to_string())
if __name__=='__main__': main()
