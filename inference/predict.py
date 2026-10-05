"""Run InfraCrackVision on one new RGB image.

Usage:
    python inference/predict.py --image path/to/image.jpg
"""
from pathlib import Path
import sys,argparse,json
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.train_four_class import Net,CLASSES
from training.train_segmentor import SegNet

def tensor_image(im):
    x=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(128,128));return torch.tensor(x/127.5-1,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--threshold',type=float,default=.5);a=p.parse_args(); src=Path(a.image)
    im=cv2.imread(str(src));
    if im is None: raise FileNotFoundError(src)
    out=ROOT/'results'/'inference';out.mkdir(parents=True,exist_ok=True);x=tensor_image(im)
    clf=Net(); payload=torch.load(ROOT/'checkpoints'/'four_class_cnn_best.pt',map_location='cpu');clf.load_state_dict(payload['model']);clf.eval()
    with torch.no_grad(): probs=torch.softmax(clf(x),1)[0]; idx=int(probs.argmax());
    result={'image':str(src),'predicted_class':CLASSES[idx],'confidence':float(probs[idx]),'class_probabilities':{c:float(probs[i]) for i,c in enumerate(CLASSES)}}
    seg_path=ROOT/'checkpoints'/'segmentor.pt'
    if seg_path.exists():
        seg=SegNet();seg.load_state_dict(torch.load(seg_path,map_location='cpu')['model']);seg.eval()
        with torch.no_grad(): mask=(torch.sigmoid(seg(x))[0,0].numpy()>=a.threshold).astype(np.uint8)*255
        result['predicted_crack_coverage']=float((mask>0).mean()); mask_path=out/(src.stem+'_mask.png'); overlay_path=out/(src.stem+'_overlay.png'); cv2.imwrite(str(mask_path),mask); base=cv2.resize(im,(128,128)); red=np.zeros_like(base);red[:,:,2]=mask; overlay=cv2.addWeighted(base,0.75,red,0.35,0);cv2.imwrite(str(overlay_path),overlay);result['mask_path']=str(mask_path);result['overlay_path']=str(overlay_path)
    else: result['segmentation_note']='segmentor.pt not found; classification only'
    json.dump(result,open(out/(src.stem+'_prediction.json'),'w'),indent=2);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
