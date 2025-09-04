#!/usr/bin/env python3
"""
Resume Ranker — local, single-file Python app (Universal LLM API Adapter)

What it does
------------
• Loads all resumes from a folder (PDF, DOCX, TXT)
• Extracts text locally
• Sends your single job/prompt query + all resumes to ONE LLM chosen in config.yml
• The LLM infers evaluation criteria (no YAML criteria/weights)
• Returns strict JSON with Top 5: file, score (0–5), summary
• Prints a ranked table and saves a JSON report

Quick start
-----------
1) Python 3.10+ (3.11+ recommended)
2) pip install docx2txt pypdf pyyaml rich llm-api-adapter
3) Put resumes in ./resumes
4) Create a provider-only config.yml (see bottom)
5) Run:
   python script_resume_ranking.py ./resumes "marketing expert in google ads" --config config.yml
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import docx2txt
from pypdf import PdfReader
import yaml
from rich.console import Console
from rich.table import Table
from rich import box

# ==== Universal LLM API Adapter imports (as in your example) ====
from llm_api_adapter.models.messages.chat_message import AIMessage, Prompt, UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


# -----------------------------
# File loading
# -----------------------------
def read_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        return f"[PDF read error: {e}]"

def read_text_from_docx(path: Path) -> str:
    try:
        return docx2txt.process(str(path)) or ""
    except Exception as e:
        return f"[DOCX read error: {e}]"

def read_text_from_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"[TXT read error: {e}]"

def load_resume_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_text_from_pdf(path)
    if ext == ".docx":
        return read_text_from_docx(path)
    if ext in {".txt", ".md"}:
        return read_text_from_txt(path)
    return ""


# -----------------------------
# Data model
# -----------------------------
@dataclass
class ResumeResult:
    file: Path
    score: float
    summary: str


# -----------------------------
# Prompt template
#   IMPORTANT: all literal JSON braces are doubled {{ }} so str.format doesn't treat them as placeholders.
# -----------------------------
PROMPT_TEMPLATE = """
You are a recruiting assistant. A set of resumes is provided along with a job query.

Your tasks (do all of them):
1) Infer reasonable evaluation criteria from the job query (e.g., role fit, core skills, seniority, domain experience, impact).
2) For each resume, assign ONE overall score from 0.0 to 5.0 (half points allowed) based ONLY on evidence in the resume.
3) Write a concise 3–5 sentence summary for each resume (most relevant experience, highlights, any gaps).
4) Return the TOP 5 resumes (or fewer if <5 provided), sorted by score DESC, in STRICT JSON.

STRICT JSON OUTPUT — nothing else:
{{
  "results": [
    {{"file": "<filename>", "score": <float>, "summary": "<3-5 sentence summary>"}},
    ... up to 5 entries ...
  ]
}}

JOB QUERY:
{job_query}

RESUMES (each delimited):
{resumes_block}
""".strip()


# -----------------------------
# Helpers
# -----------------------------
def strict_json_from_model_text(text: str) -> Dict[str, Any]:
    """Parse JSON; if the model adds extra text, extract the first {...} block."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


# -----------------------------
# Core evaluation (Universal Adapter)
# -----------------------------
def evaluate_resumes(
    job_query: str,
    resumes: Dict[str, str],
    provider_cfg: Dict[str, Any],
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> List[ResumeResult]:
    # Build a single block with all resumes
    blocks: List[str] = []
    for fname, text in resumes.items():
        text = text or ""
        if len(text) > 16000:
            text = text[:16000] + "\n[TRUNCATED]"
        blocks.append(f"=== {fname} ===\n{text}")
    resumes_block = "\n\n".join(blocks)

    prompt_text = PROMPT_TEMPLATE.format(job_query=job_query, resumes_block=resumes_block)

    # Universal adapter expects organization like "openai" | "anthropic" | "google"
    org = provider_cfg.get("organization", "google")
    model = provider_cfg["model"]
    #api_key = os.getenv(provider_cfg.get("api_key_env")) if provider_cfg.get("api_key_env") else provider_cfg.get("api_key")
    #api_key = "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"
    api_key = 'AIzaSyDFqDZEaIyx5Pk_tckh5IqsSXXTvDdmV6Q'
    print(api_key)
    base_url = provider_cfg.get("base_url")  # optional for custom endpoints (vLLM/Ollama proxies, etc.)

    adapter = UniversalLLMAPIAdapter(
        organization=org,
        model=model,
        api_key=api_key 
    )

    # Messages API (like your example)
    messages = [
        Prompt("You output ONLY strict JSON and never include code fences."),
        UserMessage(prompt_text),
    ]

    chat_params = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0,
    }

    resp = adapter.generate_chat_answer(**chat_params)
    raw = resp.content  # the text content from the model

    data = strict_json_from_model_text(raw)

    results: List[ResumeResult] = []
    for r in data.get("results", [])[:5]:
        results.append(
            ResumeResult(
                file=Path(r["file"]),
                score=float(r["score"]),
                summary=str(r["summary"]),
            )
        )
    return results


# -----------------------------
# Config + CLI
# -----------------------------
def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main():
    parser = argparse.ArgumentParser(description="Rank resumes against a prompt; LLM infers criteria; show Top 5.")
    parser.add_argument("resumes_folder", type=str, help="Folder with PDF/DOCX/TXT resumes")
    parser.add_argument("job_query", type=str, help="Job/prompt query")
    parser.add_argument("--config", type=str, default="config.yml", help="YAML with provider info ONLY")
    parser.add_argument("--out", type=str, default="rankings", help="Output basename (without extension)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    providers = cfg.get("providers", [])
    if not providers:
        raise SystemExit("No providers configured in config.yml")

    provider_cfg = providers[0]  # use the first one

    # Collect resumes
    folder = Path(args.resumes_folder)
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}],
        key=lambda p: p.name.lower(),
    )
    console = Console()
    console.print(f"Evaluating {len(files)} resumes against: [bold]{args.job_query}[/bold]\n")

    if not files:
        raise SystemExit(f"No resume files found in {folder.resolve()} (accepted: .pdf .docx .txt .md)")

    resumes = {f.name: load_resume_text(f) for f in files}

    # Evaluate
    results = evaluate_resumes(
        args.job_query, resumes, provider_cfg, temperature=args.temperature, max_tokens=args.max_tokens
    )

    # Table
    table = Table(title="Top 5 Resume Rankings", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Rank", justify="right")
    table.add_column("File")
    table.add_column("Score", justify="right")
    table.add_column("Summary")

    for i, r in enumerate(results, 1):
        table.add_row(str(i), r.file.name, f"{r.score:.2f}", r.summary)
    console.print(table)

    # Save JSON
    out_path = Path(args.out).with_suffix(".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json_results = []
        for r in results:
            json_results.append({
                "file": str(r.file),  # Convert the Path object to a string here
                "score": r.score,
                "summary": r.summary
            })
        json.dump({"job_query": args.job_query, "results": json_results}, f, ensure_ascii=False, indent=2)
    console.print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
