#!/usr/bin/env python3
"""Select illustrative saved masks from existing demo clips; no mask generation or inference."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
HERE=Path(__file__).resolve().parent

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root',type=Path,required=True)
    ap.add_argument('--preview-only',action='store_true')
    args=ap.parse_args(); root=args.dataset_root
    demo=json.loads((HERE/'demo-data.json').read_text())
    candidates=[]
    for clip in demo['clips']:
        ranked=[]
        for row in clip['frames'][::12]:
            stem=f"{clip['id']}_{row['sourceFrame']:04d}"
            mask_path=root/'images2_blur7_valid_gt5'/clip['id']/(stem+'.png')
            with Image.open(mask_path) as im: mask=np.array(im.convert('L'))
            assert mask.shape==(1080,1920) and set(np.unique(mask)).issubset({0,255})
            x1,y1,x2,y2=row['gt'][0]
            xa,ya,xb,yb=math.floor(x1),math.floor(y1),math.ceil(x2),math.ceil(y2)
            coverage=float((mask[ya:yb,xa:xb]>0).mean())
            cx,cy,cw,ch=row['crop']; crop=mask[cy:cy+ch,cx:cx+cw]
            background=float((crop>0).mean())
            score=coverage-2*background
            ranked.append(dict(video=clip['id'],frame=row['sourceFrame'],stem=stem,crop=row['crop'],gt=row['gt'],coverage=coverage,background=background,score=score))
        chosen=[]
        for row in sorted(ranked,key=lambda x:x['score'],reverse=True):
            if all(abs(row['frame']-r['frame'])>=30 for r in chosen):chosen.append(row)
            if len(chosen)==3:break
        candidates.extend(chosen)
    qa=HERE/'.qa';qa.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(960,len(candidates)*210),'#faf8f1'); draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(candidates):
        x,y,w,h=row['crop']
        rgb=Image.open(root/'images'/row['video']/(row['stem']+'.jpg')).convert('RGB')
        mask=Image.open(root/'images2_blur7_valid_gt5'/row['video']/(row['stem']+'.png')).convert('L')
        sheet.paste(rgb.resize((320,180)),(0,i*210+25))
        sheet.paste(rgb.crop((x,y,x+w,y+h)).resize((180,180),Image.Resampling.NEAREST),(330,i*210+25))
        sheet.paste(mask.crop((x,y,x+w,y+h)).resize((180,180),Image.Resampling.NEAREST),(520,i*210+25))
        draw.text((5,i*210+5),f"{row['stem']}  GT foreground={row['coverage']:.2f} local foreground={row['background']:.3f}",fill='#182e37')
    sheet.save(qa/'motion-candidates.jpg',quality=90)
    (qa/'motion-candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
    if args.preview_only:
        print(json.dumps(candidates));return
    # Use the strongest screened response from each of three existing sequences.
    selected=[next(r for r in candidates if r['video']==v) for v in ('phantom09','phantom47','phantom46')]
    out=HERE/'assets'/'motion-examples'
    out.mkdir(exist_ok=False)
    examples=[]
    for index,row in enumerate(selected):
        folder=out/row['stem'];folder.mkdir()
        rgb_path=root/'images'/row['video']/(row['stem']+'.jpg')
        mask_path=root/'images2_blur7_valid_gt5'/row['video']/(row['stem']+'.png')
        before={ 'rgb':digest(rgb_path),'mask':digest(mask_path) }
        rgb=Image.open(rgb_path).convert('RGB'); mask=Image.open(mask_path).convert('L')
        x,y,w,h=row['crop']
        rgb.save(folder/'rgb.webp',lossless=True)
        mask.save(folder/'mask.png')
        rgb.crop((x,y,x+w,y+h)).save(folder/'rgb-crop.webp',lossless=True)
        mask.crop((x,y,x+w,y+h)).save(folder/'mask-crop.png')
        assert np.array_equal(np.array(Image.open(folder/'rgb.webp')),np.array(rgb))
        assert np.array_equal(np.array(Image.open(folder/'mask.png')),np.array(mask))
        assert np.array_equal(np.array(Image.open(folder/'rgb-crop.webp')),np.array(rgb.crop((x,y,x+w,y+h))))
        assert np.array_equal(np.array(Image.open(folder/'mask-crop.png')),np.array(mask.crop((x,y,x+w,y+h))))
        assert before=={'rgb':digest(rgb_path),'mask':digest(mask_path)}
        examples.append(dict(**row, id=f'example-{index+1}',paths={name:f"assets/motion-examples/{row['stem']}/{file}" for name,file in [('rgb','rgb.webp'),('mask','mask.png'),('rgbCrop','rgb-crop.webp'),('maskCrop','mask-crop.png')]},sourceHashes=before))
    (HERE/'motion-examples.json').write_text(json.dumps(dict(status='PASS',maskFolder='images2_blur7_valid_gt5',sampledFrames=80,selection='Every twelfth frame of the four existing clips; illustrative high target foreground and low surrounding foreground. No model changes or new inference.',nativeSize=[1920,1080],croppedSize=[128,128],sourcePixelsPreserved=True,examples=examples),indent=2)+'\n')
    print(json.dumps({'status':'PASS','examples':[e['stem'] for e in examples],'images':12,'sourcePixelsPreserved':True}))
if __name__=='__main__':main()
