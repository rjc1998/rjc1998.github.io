# Drone detection portfolio demo

A buildless, local-first GitHub Pages demo of two frozen RGB + motion detectors, styled to match the portfolio's bird and music demos. Linked from the portfolio homepage. No inference server, framework, package installation, or model weights are required to view it.

## Preview

From the portfolio repository root, run:

```sh
python -m http.server 8765 --bind 127.0.0.1
```

Open <http://localhost:8765/Drone%20Detection%20demo/index.html>. A `file://` URL cannot fetch the JSON reliably. Stop the server with Ctrl+C.

## Contents and controls

- Eight approximately 15-second clips (450 frames, or 447 for phantom47) from phantom09, phantom47, phantom46, phantom58, phantom05, phantom08, phantom30, and phantom41, using their source video frame rates.
- Best mAP50: YOLOv8n P2/P3/P4/P5 without CBAM (81.4271% / 43.8028%).
- Best mAP50–95: YOLOv8n P2/P5 with CBAM (80.9499% / 44.4988%).
- Only the selected clip is downloaded; it is fully buffered before playback so seeking also works on simple local servers. One video decoder drives both full-frame canvases. Model switching preserves clip and time.
- Play/pause, frame stepping, timeline seeking, slow playback and box toggles.
- Expandable network architecture descriptions in each model panel. Opening a description preserves the selected model and playback position.
- Added scenes: a cityscape pan (phantom05), a landscaped waterway (phantom08), trees beside a pond (phantom30), and a straight-down view beside a running track (phantom41).
- Identical annotation-centered 128×128 native-pixel crops in both detail panels. Crops are losslessly stored from the original JPEGs in WebP sheets, avoiding overview-video compression. They are not new model inputs and do not constitute tracking.
- Orange model boxes use the fixed 0.25 confidence threshold. Teal boxes are the saved dataset annotations; they have no confidence scores.

`demo-data.json` contains native coordinates, frame IDs, model provenance, aggregate scores and clip metadata. `build-report.json` contains source hashes, selection criteria and asset checks. `validation.json` records all 3,597 crop/prediction checks; `browser-validation.json` records successful Chrome checks at 1440×1100 and 390×844. Local QA artifacts and the isolated Chrome profile live under ignored `.qa/`.

## Reproduction

`generate_demo_assets.py` reads the two existing evaluation exports and original dataset frames. Run it with an existing Python environment containing OpenCV (WebM support), NumPy and Pillow:

```sh
python "Drone Detection demo/generate_demo_assets.py" --research-root /path/to/y11_a2a_d2d
```

The generator writes only beside itself, does not import training code, and never runs inference. A fresh build refuses to replace an existing `assets/` directory. Use `--append` to add missing clips to an existing manifest while preserving existing video and crop assets; it verifies that the model records and source hashes still match. Existing output directories for unfinished clips are not overwritten. Rebuild in a separately versioned demo directory when necessary. The research source root is supplied at execution, not embedded as a private absolute path in the public data.

The original eight-second windows were selected using saved per-frame counts at the fixed evaluation threshold. Those original windows maximize jointly successful detection, with a small-target preference for phantom46. These curated test examples are not representative performance samples. Full metrics cover 28,100 drone-present ARD-MAV test frames. Source checkpoints were validation-selected; selecting demo models by previously exposed test scores is descriptive and exploratory. Pending attention runs are excluded.

The generator verifies consecutive frame IDs, matching GT/label hashes between models, original image dimensions, every decoded video frame per clip, video FPS, and unchanged source-file hashes. Browser QA separately checks model/clip switching, seek boundaries, playback controls and pixel-identical comparison panels with overlays disabled.

## Attribution and publication

ARD-MAV: Hanqing Guo, Ye Zheng, Yin Zhang, Zhi Gao and Shiyu Zhao, *Global-Local MAV Detection under Challenging Conditions based on Appearance and Motion* (2024). Official source: <https://github.com/WindyLab/Global-Local-MAV-Detection> (checked 2026-10-14).

Prepared for local review only; no commit, push or deployment was performed. The official repository links the dataset and citation, but an explicit footage-redistribution license was not established in this task. Confirm permission for public excerpts before publishing. Attribution is not itself a redistribution license.

## Updating browser assets

The HTML references JavaScript and CSS with content-hash query strings so an updated page does not reuse older cached assets. The application also versions its `demo-data.json` request. After changing the data, update that query value in `app.js` to the first 12 characters of the JSON SHA-256 hash. Then refresh the HTML query values from the final asset hashes and publish those files together.

## Appearance and motion examples

The research introduction now includes three selectable, matched RGB/motion-mask examples with full-resolution views and identical 128×128 crops. `motion-examples.json` records frame IDs, exact coordinates, source hashes and pixel-preservation checks. `generate_motion_examples.py` sampled 80 existing masks from the four already selected video windows and exported 12 lossless images; no masks or model predictions were recomputed. The examples show high target response with relatively little surrounding foreground, rather than average mask quality.

Mask processing was checked against `dataset_util/FD5_mask_valid.py`, `dataset_util/generate_mask5_valid.py`, and the dataset mask generation report: grayscale, Gaussian blur7, alignment, t−2/t/t+2 valid-neighbor absolute differences and a strict >5 threshold. Fusion descriptions were checked against the saved YOLO11/YOLO26 fusion comparison; metrics are explained above model selection.

The standalone `motion-gallery.js` changes only the example panel. Update the HTML content-hash query strings whenever changing this script or `styles.css`, as well as `app.js`.

## Fifteen-second clips and target dimensions

The current clips extend each original window with consecutive neighboring frames to `round(15 × source FPS)` frames. Seven clips preserve every original frame. Phantom08 uses the nearest complete window (frames 851–1300), because gaps around the original frames 588–827 prevent a continuous 15-second extension. Extra context is centered where source coverage permits; no performance rescoring, inference, or frame interpolation is used.

From the original eight-second build, the extension command is:

```sh
python "Drone Detection demo/generate_demo_assets.py" --research-root /path/to/y11_a2a_d2d --extend-seconds 15 --codec VP80
```

Extended videos and lossless crop sheets live in `assets/clips-15s/`; the previous eight-second assets remain preserved. Reusing an existing extension directory is refused by default. `--reuse-exported` permits completed clips to be reused after checking decoded frame coverage; incomplete clips still fail validation. The active videos use VP8 WebM. The existing static appearance/motion examples are unchanged.

The shared label between the cropped views shows the current ground-truth box width × height, rounded to original-image pixels. It updates when playing or seeking and stays the same when switching models.
