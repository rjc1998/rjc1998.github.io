#!/usr/bin/env python3
"""Build a selected-example replay from frozen predictions; never runs inference.
All output stays beside this script. Source datasets and research artifacts are read-only.
Requires existing numpy, Pillow and OpenCV with a VP9 encoder.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
TARGETS = {
    'phantom09': ('A clear encounter', 'A larger target makes the difference between finding a drone and fitting its outline easier to inspect.'),
    'phantom47': ('Changing scale', 'Follow the same target through a short sequence and compare the two models at any frame.'),
    'phantom46': ('Only a few pixels', 'A very small drone is difficult to see at full-frame scale. The native-pixel detail view reveals the target.'),
    'phantom58': ('Small target, shared view', 'Inspect a second small-target sequence with identical timing and crops for both models.'),
}
SOURCES = {
    'map50': 'runs/ard_mav/evaluations/yolov8_head_configurations_valid_gt5_test_v1/p2_p3_p4_p5',
    'map5095': 'runs/ard_mav/evaluations/cbam_top3_valid_gt5_test_v1/yolov8_p2_p5',
}
NAMES = {'map50': 'YOLOv8n · P2 / P3 / P4 / P5', 'map5095': 'YOLOv8n · P2 / P5 + CBAM'}
CONFIDENCE = .25
CELL = 128
GRID = 8
LENGTH = 240

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def rounded(boxes):
    return [[round(float(x), 4) for x in b] for b in boxes]

def load(root):
    rows, models, provenance = {}, [], {}
    for key, relative in SOURCES.items():
        folder = root / relative
        report = json.loads((folder / 'report.json').read_text())
        metrics = json.loads((folder / 'metrics.json').read_text())
        print(key, 'metric keys:', list(metrics), flush=True)
        rows[key] = {v: {} for v in TARGETS}
        source = folder / 'predictions.jsonl.gz'
        provenance[relative + '/predictions.jsonl.gz'] = sha(source)
        with gzip.open(source, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['video'] in TARGETS:
                    frame = int(Path(row['image']).stem.rsplit('_', 1)[1])
                    rows[key][row['video']][frame] = row
        assert report['status'] == 'COMPLETE'
        assert report['requested']['conf'] == CONFIDENCE
        models.append(dict(id=key, name=NAMES[key], badge='Best mAP50' if key == 'map50' else 'Best mAP50–95',
                           ap50=metrics['ap50'], map5095=metrics['map50_95'],
                           checkpoint_sha256=report['checkpoint_sha256'],
                           source=relative, checkpoint_epoch=report['checkpoint_epoch']))
        for name in ('report.json', 'metrics.json', 'dataset.json'):
            provenance[relative + '/' + name] = sha(folder / name)
    datasets = [json.loads((root / p / 'dataset.json').read_text()) for p in SOURCES.values()]
    assert datasets[0]['manifest_sha256'] == datasets[1]['manifest_sha256']
    return rows, models, provenance, datasets[0]

def select_window(rows, video):
    a, b = rows['map50'][video], rows['map5095'][video]
    common = sorted(set(a) & set(b))
    candidates = []
    for offset in range(0, len(common) - LENGTH + 1, 5):
        frames = common[offset:offset + LENGTH]
        if frames[-1] - frames[0] != LENGTH - 1:
            continue
        scores = []
        for source in (a, b):
            scores.append(sum(source[f]['counts']['tp'] - .35 * source[f]['counts']['fp'] for f in frames) / LENGTH)
        tiny_fraction = sum(a[f]['gt_native_size'][0] < 8 for f in frames) / LENGTH
        # Prefer jointly successful examples; do not tune predictions or thresholds.
        score = min(scores) + .1 * sum(scores) + (.8 * tiny_fraction if video == 'phantom46' else 0)
        candidates.append((score, -frames[0], frames))
    if not candidates:
        raise RuntimeError(f'No common continuous {LENGTH}-frame window: {video}')
    return max(candidates)[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.research_root.resolve()
    out = HERE / 'assets'
    if out.exists():
        raise SystemExit('assets/ already exists; preserve it and choose a separately versioned demo directory to rebuild.')
    rows, models, provenance, dataset = load(root)
    data_root = Path(dataset['source_root'])
    out.mkdir()
    demo = dict(schema=1, confidence=CONFIDENCE, split='ARD-MAV test', evaluatedFrames=28100,
                nativeWidth=1920, nativeHeight=1080, cropSize=CELL, sheetGrid=GRID,
                selection='Selected successful continuous windows from saved test predictions; not representative accuracy samples.',
                models=models, clips=[])
    checks = []
    for video, (title, description) in TARGETS.items():
        frames = select_window(rows, video)
        original_video = data_root / 'videos' / (video + '.mp4')
        cap = cv2.VideoCapture(str(original_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert 1 < fps < 121, (video, fps)
        directory = out / video
        directory.mkdir()
        movie = directory / 'overview.webm'
        writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*'VP90'), fps, (1280, 720))
        if not writer.isOpened():
            raise RuntimeError('OpenCV VP9 encoder unavailable; no substitute inference or installs attempted.')
        clip = dict(id=video, title=title, description=description, fps=fps, frameCount=LENGTH,
                    firstSourceFrame=frames[0], lastSourceFrame=frames[-1], duration=LENGTH/fps,
                    video=f'assets/{video}/overview.webm', poster=f'assets/{video}/poster.webp',
                    sheets=[], frames=[], counts={k: dict(tp=0, fp=0, fn=0) for k in SOURCES})
        sheet = None
        sizes = []
        for i, frame in enumerate(frames):
            a = rows['map50'][video][frame]
            b = rows['map5095'][video][frame]
            assert a['image'] == b['image'] and a['label_sha256'] == b['label_sha256']
            assert np.allclose(a['gt_xyxy'], b['gt_xyxy'], atol=.001)
            image_path = Path(a['image']).resolve()
            assert image_path.is_relative_to(data_root.resolve())
            im = Image.open(image_path).convert('RGB')
            assert im.size == (1920, 1080)
            gt = a['gt_xyxy']
            assert len(gt) == 1, 'This demo expects one annotated target per selected frame.'
            cx = (gt[0][0] + gt[0][2]) / 2
            cy = (gt[0][1] + gt[0][3]) / 2
            x = max(0, min(1920 - CELL, round(cx - CELL/2)))
            y = max(0, min(1080 - CELL, round(cy - CELL/2)))
            crop = im.crop((x, y, x + CELL, y + CELL))
            if i % (GRID*GRID) == 0:
                sheet = Image.new('RGB', (GRID*CELL, GRID*CELL))
            slot = i % (GRID*GRID)
            sheet.paste(crop, ((slot % GRID)*CELL, (slot // GRID)*CELL))
            if slot == GRID*GRID - 1 or i == LENGTH - 1:
                name = f'crop-{i // (GRID*GRID):02d}.webp'
                sheet.save(directory / name, lossless=True, method=4)
                clip['sheets'].append(f'assets/{video}/{name}')
            overview = im.resize((1280, 720), Image.Resampling.LANCZOS)
            writer.write(cv2.cvtColor(np.asarray(overview), cv2.COLOR_RGB2BGR))
            if i == 0:
                overview.save(directory / 'poster.webp', quality=90)
                crop.resize((512,512), Image.Resampling.NEAREST).save(directory / 'detail.webp', lossless=True)
            record = dict(sourceFrame=frame, gt=rounded(gt), crop=[x,y,CELL,CELL], predictions={})
            for key in SOURCES:
                row = rows[key][video][frame]
                record['predictions'][key] = [rounded([box])[0] + [round(float(conf),6)]
                    for box, conf in zip(row['boxes_xyxy'], row['confidence']) if conf >= CONFIDENCE]
                for field in ('tp','fp','fn'):
                    clip['counts'][key][field] += row['counts'][field]
            clip['frames'].append(record)
            sizes.append(a['gt_native_size'][0])
        writer.release()
        # Decode every encoded frame: check coverage, shape and timing metadata.
        check_cap = cv2.VideoCapture(str(movie))
        count = 0
        while True:
            ok, decoded = check_cap.read()
            if not ok:
                break
            assert decoded.shape[:2] == (720, 1280)
            count += 1
        encoded_fps = check_cap.get(cv2.CAP_PROP_FPS)
        check_cap.release()
        assert count == LENGTH and abs(encoded_fps-fps) < .05
        clip['nativeSizeRange'] = [round(min(sizes),1), round(max(sizes),1)]
        demo['clips'].append(clip)
        checks.append(dict(video=video, sourceRange=[frames[0],frames[-1]], decodedFrames=count, fps=fps,
                           gtAgrees=True, contiguous=True, nativeSizeRange=clip['nativeSizeRange'],
                           counts=clip['counts'], bytes=movie.stat().st_size))
        print(json.dumps(checks[-1]), flush=True)
    # Showcase poster: same original frame, a visible annotation, and a native-pixel inset.
    first = demo['clips'][0]
    image = Image.open(out / first['id'] / 'poster.webp').convert('RGB')
    draw = ImageDraw.Draw(image)
    box = first['frames'][0]['gt'][0]
    draw.rectangle(tuple(v*2/3 for v in box), outline='#e4ae73', width=3)
    detail = Image.open(out / first['id'] / 'detail.webp').resize((240,240), Image.Resampling.NEAREST)
    image.paste(detail, (1000,440))
    draw.rectangle((998,438,1242,682), outline='#e4ae73', width=3)
    image.save(out / 'cover.webp', quality=90)
    (HERE / 'demo-data.json').write_text(json.dumps(demo, separators=(',',':')) + '\n')
    unchanged = all(sha(root / relative) == digest for relative, digest in provenance.items())
    assert unchanged
    report = dict(status='PASS', selectionRule='240 consecutive shared frames; maximize minimum per-model mean(TP - 0.35*FP), plus 0.1*sum(means); phantom46 adds 0.8*fraction(native size <8). Candidates every 5 positions. Ties prefer earliest.',
                  confidence=CONFIDENCE, inferenceRun=False, sourcesUnchanged=unchanged,
                  sourceHashes=provenance, clips=checks, publication='LOCAL PREVIEW ONLY; dataset redistribution permission not established by this build.')
    (HERE / 'build-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: assets and provenance written only in', HERE, flush=True)

if __name__ == '__main__':
    main()
