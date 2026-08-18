# Bird Classifier Portfolio Demo

This directory contains a fully static comparison of the trained SimCLR and
ProxyNCA linear classifiers. PyTorch is used only by the offline generator;
the deployed page consists of HTML, CSS, JavaScript, JSON, and optimized WebP
images.

## Generate the demo

From the repository root, make sure these files are available:

- `cub2011/CUB_200_2011/` — extracted CUB-200-2011 dataset
- `simclr_linear_classifier_resnet50.pth`
- `proxynca_linear_classifier_resnet50.pth`

Install the same PyTorch environment used by the project, then run:

```powershell
python demo/generate_demo_assets.py
```

The generator deterministically selects 12 test images, calculates both
models' predictions, creates Grad-CAM overlays, retrieves five nearest
training images per model, and writes `site/demo-data.json` plus WebP assets.
Use `--cpu` when CUDA is unavailable, or see `--help` to override paths.

Validate an existing generated bundle without loading either model:

```powershell
python demo/generate_demo_assets.py --validate-only
```

## Preview and deploy

Browsers do not allow `fetch()` from a page opened directly with `file://`.
Serve the site locally instead:

```powershell
python -m http.server 8000 --directory demo/site
```

Open `http://localhost:8000`. To deploy, publish the contents of
`demo/site` as the static-host root. GitHub Pages, Netlify, and
similar static hosts need no build command.

The deployed bundle does not require the `.pth` checkpoints, CUB source data,
Python, or a GPU. Generated embeddings remain in memory during generation and
are never written to the site.

## Interpretation

- **Confidence** is the raw softmax score. It has not been calibrated.
- **Grad-CAM** highlights regions that influenced the predicted class; it is
  an attribution technique, not a literal representation of model reasoning.
- **Nearest neighbors** are the five most similar training images after L2
  normalization, ranked by cosine similarity in the selected encoder's
  embedding space. Test queries are never included in the retrieval corpus.

## Dataset attribution

The images are drawn from the
[Caltech-UCSD Birds-200-2011 dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/).
The demo preserves source image IDs and split information in its JSON manifest.
Review the dataset's terms before publishing the curated image assets.
