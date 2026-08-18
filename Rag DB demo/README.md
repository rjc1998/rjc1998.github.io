# GitHub Pages portfolio demo

This static HTML/CSS/JavaScript demo presents the completed local experiment.
Visitors can select 25 post-cutoff Resident Evil Requiem questions, compare
Llama 3 8B with and without RAG at the fixed `k=3` setting, reveal the benchmark
answer, and inspect the three retrieved passages.

GitHub Pages cannot run Ollama or PostgreSQL, so the page replays actual saved
experiment outputs rather than generating new answers in the browser.

The portfolio answers use a separate best-effort presentation prompt that does
not permit `UNKNOWN`. These outputs must not be substituted for the controlled
evaluation results.

## Refresh data

```powershell
.\.venv\Scripts\python.exe demo/build_demo_data.py
```

To regenerate every demo answer without abstentions instead, ensure Ollama and
PostgreSQL are running and execute:

```powershell
.\.venv\Scripts\python.exe demo/generate_demo_answers.py
```

## Preview locally

```powershell
.\.venv\Scripts\python.exe -m http.server 8000
```

Open `http://localhost:8000/demo/`. To publish, enable GitHub Pages for the
repository branch; the demo will be available at the site's `/demo/` path.
