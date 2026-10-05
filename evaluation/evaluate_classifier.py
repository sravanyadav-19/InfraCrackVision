"""Evaluate the saved four-class CNN once on the untouched test split.
Run: python evaluation/evaluate_classifier.py
"""
from pathlib import Path
import sys
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.train_four_class import Net, DS, CLASSES; SPLIT=ROOT/'results'/'splits.csv'; CKPT=ROOT/'checkpoints'/'four_class_cnn_best.pt'; OUT=ROOT/'results'
def main():
    if not SPLIT.exists(): raise FileNotFoundError('results/splits.csv not found')
    if not CKPT.exists(): raise FileNotFoundError('checkpoints/four_class_cnn_best.pt not found')
    df=pd.read_csv(SPLIT); test=df[df.split=='test'].copy(); loader=DataLoader(DS(test),batch_size=32,num_workers=0)
    payload=torch.load(CKPT,map_location='cpu'); model=Net(); model.load_state_dict(payload['model']); model.eval(); y=[];p=[]
    with torch.no_grad():
        for x,t in loader: p.extend(model(x).argmax(1).tolist()); y.extend(t.tolist())
    report=classification_report(y,p,labels=range(4),target_names=CLASSES,output_dict=True,zero_division=0); result={'test_records':len(test),'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'confusion_matrix':confusion_matrix(y,p,labels=range(4)).tolist(),'classification_report':report}
    OUT.mkdir(exist_ok=True); json.dump(result,open(OUT/'four_class_test_metrics.json','w'),indent=2); print(json.dumps({'test_records':len(test),'accuracy':result['accuracy'],'balanced_accuracy':result['balanced_accuracy'],'macro_f1':report['macro avg']['f1-score']},indent=2)); print(pd.DataFrame(result['confusion_matrix'],index=CLASSES,columns=CLASSES))
if __name__=='__main__':main()
