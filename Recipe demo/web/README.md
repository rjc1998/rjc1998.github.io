# Portfolio-ready static demo

`korean-recipe-generator-demo/` is a complete static export of the Stage 8 site. Copy that folder
into the root of `rjc1998.github.io` (or any static host) without installing Node, Python, a backend,
or model dependencies.

After copying it into the portfolio repository, the demo URL will be:

`https://rjc1998.github.io/korean-recipe-generator-demo/`

The export uses relative asset paths, so renaming the folder is safe. Open it through a local HTTP
server rather than directly through `file://` when testing locally:

```bash
python -m http.server 8000 --directory demo/web
```

Then visit `http://localhost:8000/korean-recipe-generator-demo/`.

To regenerate the folder after editing the React source:

```bash
cd portfolio/demo
npm ci
npm run build:portfolio
```

The generated folder bundles the frozen Stage 7 data into its JavaScript and makes no requests to
this source repository. Google Fonts are optional; local fallbacks preserve the layout offline.
