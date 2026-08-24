# rjc1998.github.io
Repository for portfolio and git hub page

## Project visibility

The project cards shown on the portfolio homepage are controlled by the four `data-show-*`
settings on the `<main>` element in `index.html`:

```html
<main
  data-show-bird="true"
  data-show-music="true"
  data-show-recipe="true"
  data-show-rag="true"
>
```

Change a project's value to `"false"`, commit, and publish to remove that project from the
homepage. There is no visitor-facing visibility control.

GitHub Pages only serves static public files. Hiding a homepage card does not secure its demo URL;
to prevent all public access, remove that demo's files from the published site or use a host with
authentication.
