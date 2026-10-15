# Drone detection portfolio demo

A buildless, local-first GitHub Pages demo of two frozen RGB + motion detectors, styled to match the portfolio's bird and music demos. Linked from the portfolio homepage. No inference server, framework, package installation, or model weights are required to view it.

## Preview

From the portfolio repository root, run:

```sh
python -m http.server 8765 --bind 127.0.0.1
```

Open <http://localhost:8765/Drone%20Detection%20demo/index.html>. A `file://` URL cannot fetch the JSON reliably. Stop the server with Ctrl+C.

## Contents and controls

- Four 240-frame clips from phantom09, phantom47, phantom46, and phantom58, using their source video frame rates.
- Best mAP50: YOLOv8n P2/P3/P4/P5 without CBAM (81.4271% / 43.8028%).
- Best mAP50–95: YOLOv8n P2/P5 with CBAM (80.9499% / 44.4988%).
- Only the selected clip is downloaded; it is fully buffered before playback so seeking also works on simple local servers. Assets total approximately 92 MiB. One video decoder drives both full-frame canvases. Model switching preserves clip and time.
- Play/pause, frame stepping, timeline seeking, slow playback, box toggles and crop guides.
- Identical annotation-centered 128×128 native-pixel crops in both detail panels. Crops are losslessly stored from the original JPEGs in WebP sheets, avoiding overview-video compression. They are not new model inputs and do not constitute tracking.
- Orange model boxes use the fixed 0.25 confidence threshold. Teal boxes are the saved dataset annotations; they have no confidence scores.

`demo-data.json` contains native coordinates, frame IDs, model provenance, aggregate scores and clip metadata. `build-report.json` contains source hashes, selection criteria and asset checks. `validation.json` records all 960 crop/prediction checks; `browser-validation.json` records successful Chrome checks at 1440×1100 and 390×844. Local QA artifacts and the isolated Chrome profile live under ignored `.qa/`.

## Reproduction

`generate_demo_assets.py` reads the two existing evaluation exports and original dataset frames. Run it with an existing Python environment containing OpenCV (VP9 support), NumPy and Pillow:

```sh
python "Drone Detection demo/generate_demo_assets.py" --research-root /path/to/y11_a2a_d2d
```

The generator writes only beside itself, refuses to replace an existing `assets/` directory, does not import training code, and never runs inference. Rebuild in a separately versioned demo directory to preserve prior assets. The research source root is supplied at execution, not embedded as a private absolute path in the public data.

Windows are selected using saved per-frame counts at the fixed evaluation threshold. They maximize jointly successful detection, with a small-target preference for phantom46. These curated test examples are not representative performance samples. Full metrics cover 28,100 drone-present ARD-MAV test frames. Source checkpoints were validation-selected; selecting demo models by previously exposed test scores is descriptive and exploratory. Pending attention runs are excluded.

The generator verifies consecutive frame IDs, matching GT/label hashes between models, original image dimensions, all 240 decoded video frames per clip, video FPS, and unchanged source-file hashes. Browser QA separately checks model/clip switching, seek boundaries, playback controls and pixel-identical comparison panels with overlays disabled.

## Attribution and publication

ARD-MAV: Hanqing Guo, Ye Zheng, Yin Zhang, Zhi Gao and Shiyu Zhao, *Global-Local MAV Detection under Challenging Conditions based on Appearance and Motion* (2024). Official source: <https://github.com/WindyLab/Global-Local-MAV-Detection> (checked 2026-10-14).

Prepared for local review only; no commit, push or deployment was performed. The official repository links the dataset and citation, but an explicit footage-redistribution license was not established in this task. Confirm permission for public excerpts before publishing. Attribution is not itself a redistribution license.
