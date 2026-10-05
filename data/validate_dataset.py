"""Validate InfraCrackVision datasets before model training.

Run from the repository root:
    python data/validate_dataset.py
"""
from pathlib import Path
from collections import Counter
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets'
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
errors=[]; warnings=[]
def images(folder): return sorted([p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTS], key=lambda p: p.name)
def check_readable(paths, label):
    bad=[]; modes=Counter(); sizes=Counter()
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                modes[im.mode]+=1; sizes[im.size]+=1
        except Exception as e: bad.append(f'{p}: {e}')
    if bad: errors.extend([f'{label} unreadable: {x}' for x in bad])
    return modes,sizes

def main():
    print('InfraCrackVision dataset validation')
    print('Root:', DATA)
    # DeepCrack fixed train/test pairing
    for split in ['train','test']:
        idir=DATA/'deepcrack'/'images'/split; mdir=DATA/'deepcrack'/'masks'/split
        if not idir.is_dir() or not mdir.is_dir(): errors.append(f'Missing DeepCrack {split} image/mask folder'); continue
        ims=images(idir); masks=images(mdir); imst={p.stem for p in ims}; mst={p.stem for p in masks}
        print(f'DeepCrack {split}: images={len(ims)}, masks={len(masks)}')
        if imst-mst: errors.append(f'DeepCrack {split} missing masks: {len(imst-mst)}')
        if mst-imst: errors.append(f'DeepCrack {split} missing images: {len(mst-imst)}')
        im_modes,im_sizes=check_readable(ims,f'DeepCrack {split} images'); mask_modes,mask_sizes=check_readable(masks,f'DeepCrack {split} masks')
        print('  image modes:',dict(im_modes),'sizes:',dict(im_sizes)); print('  mask modes:',dict(mask_modes),'sizes:',dict(mask_sizes))
        for p in masks[:min(20,len(masks))]:
            a=np.asarray(Image.open(p).convert('L')); vals=set(np.unique(a).tolist())
            if not vals.issubset({0,255}): warnings.append(f'{p} has non-binary values: {sorted(vals)[:10]}')
    # SDNET counts and readability; .gitkeep is ignored
    for cls in ['cracked','noncracked']:
        d=DATA/'sdnet2018'/cls
        if not d.is_dir(): warnings.append(f'Missing SDNET2018 folder: {cls}'); continue
        fs=images(d); print(f'SDNET2018 {cls}: images={len(fs)}'); check_readable(fs,f'SDNET2018 {cls}')
    print('\nSummary:')
    print('Errors:',len(errors),'Warnings:',len(warnings))
    for x in errors: print('ERROR:',x)
    for x in warnings[:20]: print('WARNING:',x)
    if errors: sys.exit(1)
    print('VALIDATION PASSED' if not warnings else 'VALIDATION PASSED WITH WARNINGS')
if __name__=='__main__': main()
