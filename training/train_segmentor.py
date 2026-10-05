"""Train a DeepCrack-inspired segmentation CNN on datasets/deepcrack.
Run smoke test: python training/train_segmentor.py --epochs 2 --limit 32
"""
from pathlib import Path
import argparse, json, time
import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'datasets/deepcrack'; CKPT=ROOT/'checkpoints'; OUT=ROOT/'results'; CKPT.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
class CrackDS(Dataset):
 def __init__(self,split,limit=None):
  self.img=sorted([p for p in (DATA/'images'/split).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}]); self.mask=DATA/'masks'/split; self.img=self.img[:limit] if limit else self.img
 def __len__(self): return len(self.img)
 def __getitem__(self,i):
  p=self.img[i]; m=self.mask/(p.stem+'.png'); x=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB); y=cv2.imread(str(m),0); x=cv2.resize(x,(128,128)); y=cv2.resize(y,(128,128),interpolation=cv2.INTER_NEAREST); return torch.tensor(x/127.5-1,dtype=torch.float32).permute(2,0,1),torch.tensor(y>127,dtype=torch.float32).unsqueeze(0)
class SegNet(nn.Module):
 def __init__(self):
  super().__init__(); self.e=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU()); self.d=nn.Sequential(nn.ConvTranspose2d(128,64,2,2),nn.ReLU(),nn.ConvTranspose2d(64,32,2,2),nn.ReLU(),nn.Conv2d(32,1,1))
 def forward(self,x): return self.d(self.e(x))
def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=30);p.add_argument('--limit',type=int,default=None);a=p.parse_args(); ds=CrackDS('train',a.limit); dl=DataLoader(ds,batch_size=8,shuffle=True,num_workers=0); m=SegNet(); opt=torch.optim.Adam(m.parameters(),1e-3); lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.])); hist=[]; t=time.time()
 for e in range(a.epochs):
  m.train(); ls=[]
  for x,y in dl: opt.zero_grad(); z=m(x); loss=lossfn(z,y); loss.backward(); opt.step(); ls.append(loss.item())
  hist.append(float(np.mean(ls))); print(f'epoch {e+1}/{a.epochs} loss={hist[-1]:.4f}')
 torch.save({'model':m.state_dict(),'image_size':128},CKPT/'segmentor.pt'); json.dump({'loss':hist,'seconds':time.time()-t},open(OUT/'segmentor_history.json','w'),indent=2); print('saved',CKPT/'segmentor.pt')
if __name__=='__main__': main()
