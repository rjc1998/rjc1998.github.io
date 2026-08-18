"""Export completed evaluation responses for the static GitHub Pages demo."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
QUESTIONS=ROOT/"data"/"questions"/"evaluation_questions.json"
RESPONSES=ROOT/"data"/"results"/"cisa_kev"/"eval_responses.jsonl"
OUTPUT=Path(__file__).resolve().parent/"data"/"demo_results.json"
K_VALUE=5

def build_demo_payload(questions_path:Path=QUESTIONS,responses_path:Path=RESPONSES)->list[dict[str,Any]]:
    """Join each unseen benchmark item to baseline and fixed-k RAG runs."""
    questions=json.loads(questions_path.read_text(encoding="utf-8"))["unseen"]
    responses={row["response_id"]:row for line in responses_path.read_text(encoding="utf-8").splitlines() if line.strip() for row in [json.loads(line)]}
    payload=[]
    for question in questions:
        qid=question["id"]; baseline=responses[f"{qid}_baseline"]
        rag=responses[f"{qid}_rag_k{K_VALUE}"]
        payload.append({"id":qid,"question":question["question"],"reference_answer":question["reference_answer"],"fact_date":question["fact_date"],"source_urls":question.get("source_urls",[]),"baseline":{"answer":baseline["answer"],"generation_seconds":baseline["generation_seconds"]},"rag":{"answer":rag["answer"],"generation_seconds":rag["generation_seconds"],"retrieved_chunks":rag["retrieved_chunks"]}})
    return payload

def main()->None:
    """Write browser-readable data without credentials or connection strings."""
    payload=build_demo_payload(); OUTPUT.parent.mkdir(parents=True,exist_ok=True); OUTPUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); print(f"Wrote {len(payload)} demo questions to {OUTPUT}")

if __name__=="__main__": main()
