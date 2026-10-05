"""Use checkpoints/segmentor.pt to create masks and coverage labels for SDNET cracked images."""
from pathlib import Path
import sys
import argparse, cv2, torch, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.train_segmentor import SegNet
def main():
 p=argparse.ArgumentParser();p.add_argument('--threshold',type=float,default=.5);a=p.parse_args(); ck=torch.load(ROOT/'checkpoints/segmentor.pt',map_location='cpu');m=SegNet();m.load_state_dict(ck['model']);m.eval(); src=ROOT/'datasets/sdnet2018/cracked'; out=ROOT/'datasets/sdnet2018/pseudo_masks';out.mkdir(parents=True,exist_ok=True); rows=[]
 with torch.no_grad():
  for f in [x for x in src.rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png'}]:
   im=cv2.cvtColor(cv2.imread(str(f)),cv2.COLOR_BGR2RGB); im=cv2.resize(im,(128,128)); x=torch.tensor(im/127.5-1,dtype=torch.float32).permute(2,0,1).unsqueeze(0); mask=(torch.sigmoid(m(x))[0,0].numpy()>=a.threshold).astype(np.uint8)*255; cv2.imwrite(str(out/(f.stem+'.png')),mask); rows.append({'filename':f.name,'mask_path':str(out/(f.stem+'.png')),'crack_pixel_ratio':float((mask>0).mean())})
 pd.DataFrame(rows).to_csv(ROOT/'results/sdnet_pseudo_labels.csv',index=False);print(f'Created {len(rows)} pseudo masks')
if __name__=='__main__': main()
