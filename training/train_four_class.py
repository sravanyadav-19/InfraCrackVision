"""Train the controlled four-class CNN using the frozen source-safe split.

Smoke test:
  python training/train_four_class.py --epochs 2 --limit 64
Full prototype:
  python training/train_four_class.py --epochs 20
"""
from pathlib import Path
import argparse,json,random,time
import cv2,numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,classification_report
ROOT=Path(__file__).resolve().parents[1]; SPLIT=ROOT/'results'/'splits.csv'; CKPT=ROOT/'checkpoints'; OUT=ROOT/'results'; CKPT.mkdir(exist_ok=True)
CLASSES=['no-crack','minor','moderate','severe']
class DS(Dataset):
 def __init__(self,df): self.df=df.reset_index(drop=True)
 def __len__(self): return len(self.df)
 def __getitem__(self,i):
  r=self.df.iloc[i]; x=cv2.imread(str(ROOT/r.image_path));
  if x is None: raise FileNotFoundError(r.image_path)
  x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB); x=cv2.resize(x,(128,128)); return torch.tensor(x/127.5-1,dtype=torch.float32).permute(2,0,1),CLASSES.index(r.severity_class)
class Net(nn.Module):
 def __init__(self):
  super().__init__(); self.f=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(128,4)
 def forward(self,x): return self.head(self.f(x).flatten(1))
def cap(df,n,seed):
 if n is None: return df
 return pd.concat([g.sample(min(len(g),max(1,n//4)),random_state=seed) for _,g in df.groupby('severity_class')],ignore_index=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--limit',type=int,default=None);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 if not SPLIT.exists(): raise FileNotFoundError('Run data/split_by_source.py first')
 df=pd.read_csv(SPLIT); train=cap(df[df.split=='train'],a.limit,a.seed); val=df[df.split=='validation'].copy(); print('train distribution:\n',train.severity_class.value_counts().reindex(CLASSES,fill_value=0)); print('validation distribution:\n',val.severity_class.value_counts().reindex(CLASSES,fill_value=0))
 tr=DataLoader(DS(train),batch_size=16,shuffle=True,num_workers=0); va=DataLoader(DS(val),batch_size=32,num_workers=0); m=Net(); counts=train.severity_class.value_counts().reindex(CLASSES).fillna(1); weights=torch.tensor((len(train)/(4*counts)).to_numpy(),dtype=torch.float32); lossfn=nn.CrossEntropyLoss(weight=weights); opt=torch.optim.Adam(m.parameters(),1e-3); hist=[];best=-1;start=time.time()
 for e in range(a.epochs):
  m.train();losses=[];yp=[];yt=[]
  for x,y in tr: opt.zero_grad();z=m(x);loss=lossfn(z,y);loss.backward();opt.step();losses.append(loss.item());yp+=z.argmax(1).tolist();yt+=y.tolist()
  m.eval();vl=[];vp=[];vy=[]
  with torch.no_grad():
   for x,y in va:z=m(x);vl.append(lossfn(z,y).item());vp+=z.argmax(1).tolist();vy+=y.tolist()
  rec={'epoch':e+1,'train_loss':float(np.mean(losses)),'val_loss':float(np.mean(vl)),'train_acc':accuracy_score(yt,yp),'val_acc':accuracy_score(vy,vp),'val_macro_f1':classification_report(vy,vp,labels=range(4),output_dict=True,zero_division=0)['macro avg']['f1-score']};hist.append(rec);print(rec)
  if rec['val_macro_f1']>best: best=rec['val_macro_f1'];torch.save({'model':m.state_dict(),'classes':CLASSES,'image_size':128,'split_file':'results/splits.csv'},CKPT/'four_class_cnn_best.pt')
 json.dump({'history':hist,'seconds':time.time()-start,'best_val_macro_f1':best},open(OUT/'four_class_history.json','w'),indent=2);print('saved',CKPT/'four_class_cnn_best.pt')
if __name__=='__main__':main()
