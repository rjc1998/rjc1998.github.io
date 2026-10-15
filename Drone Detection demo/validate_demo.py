#!/usr/bin/env python3
"""Validate generated asset provenance and exact native crop pixels. No inference."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    args=parser.parse_args()
    data=json.loads((HERE/'demo-data.json').read_text())
    build=json.loads((HERE/'build-report.json').read_text())
    for relative,digest in build['sourceHashes'].items():
        assert hashlib.sha256((args.research_root/relative).read_bytes()).hexdigest()==digest, relative
    needed={(c['id'],f['sourceFrame']):f for c in data['clips'] for f in c['frames']}
    original={}
    predictions=0
    for model in data['models']:
        folder=args.research_root/model['source']
        report=json.loads((folder/'report.json').read_text())
        metrics=json.loads((folder/'metrics.json').read_text())
        assert model['checkpoint_sha256']==report['checkpoint_sha256']
        assert model['ap50']==metrics['ap50'] and model['map5095']==metrics['map50_95']
        seen=set()
        with gzip.open(folder/'predictions.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line)
                frame=int(Path(row['image']).stem.rsplit('_',1)[1])
                key=(row['video'],frame)
                if key not in needed: continue
                seen.add(key); item=needed[key]
                assert np.allclose(item['gt'],row['gt_xyxy'],atol=.0001)
                expected=[box+[score] for box,score in zip(row['boxes_xyxy'],row['confidence']) if score>=data['confidence']]
                actual=item['predictions'][model['id']]
                assert len(expected)==len(actual)
                if actual: assert np.allclose(actual,expected,atol=.0001)
                predictions+=len(actual)
                original[key]=Path(row['image'])
        assert len(seen)==len(needed)
    count=0
    for clip in data['clips']:
        assert (HERE/clip['video']).is_file() and (HERE/clip['poster']).is_file()
        assert clip['lastSourceFrame']-clip['firstSourceFrame']+1==clip['frameCount']==240
        sheets=[Image.open(HERE/p).convert('RGB') for p in clip['sheets']]
        for i,frame in enumerate(clip['frames']):
            assert frame['sourceFrame']==clip['firstSourceFrame']+i
            x,y,w,h=frame['crop']
            assert 0<=x<=1920-w and 0<=y<=1080-h and w==h==128
            with Image.open(original[(clip['id'],frame['sourceFrame'])]) as im:
                expected=np.asarray(im.convert('RGB').crop((x,y,x+w,y+h)))
            slot=i%64; sx=(slot%8)*128; sy=(slot//8)*128
            actual=np.asarray(sheets[i//64].crop((sx,sy,sx+128,sy+128)))
            assert np.array_equal(expected,actual), (clip['id'],i)
            count+=1
    total=sum(p.stat().st_size for p in (HERE/'assets').rglob('*') if p.is_file())
    result=dict(status='PASS',sourceHashes=len(build['sourceHashes']),frames=count,
                modelFramePairs=count*2,predictionBoxes=predictions,losslessCropsVerified=count,
                metricsMatch=True,assetsBytes=total,sourceWrites=False,inferenceRun=False)
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
