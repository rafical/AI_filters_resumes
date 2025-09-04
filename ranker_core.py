#!/usr/bin/env python3
"""
ranker_core.py
Core logic for Resume Ranker — Universal LLM API Adapter for cloud models,
with a native Ollama fallback for local models.

Exports:
- ResumeResult (dataclass)
- load_config(path) -> dict
- load_resumes(folder: Path) -> dict[str, str]
- evaluate_resumes(job_query, resumes, provider_cfg, temperature=0.2, max_tokens=1000) -> list[ResumeResult]
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import docx2txt
import yaml
from pypdf import PdfReader

# Universal LLM API Adapter (cloud / universal path)
from llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


# -----------------------------
# Data model
# -----------------------------
@dataclass
class ResumeResult:
    file: str      # store as string for easy JSON
    score: float
    summary: str


# -----------------------------
# File loading
# -----------------------------
def _read_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        return f"[PDF read error: {e}]"

def _read_text_from_docx(path: Path) -> str:
    try:
        return docx2txt.process(str(path)) or ""
    except Exception as e:
        return f"[DOCX read error: {e}]"

def _read_text_from_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"[TXT read error: {e}]"

def load_resumes(folder: Path) -> Dict[str, str]:
    """Return {filename -> text} for .pdf/.docx/.txt/.md files in folder."""
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}],
        key=lambda p: p.name.lower(),
    )
    texts: Dict[str, str] = {}
    for f in files:
        ext = f.suffix.lower()
        if ext == ".pdf":
            txt = _read_text_from_pdf(f)
        elif ext == ".docx":
            txt = _read_text_from_docx(f)
        else:
            txt = _read_text_from_txt(f)
        texts[f.name] = txt
    return texts


# -----------------------------
# Criteria extraction & matching
# -----------------------------
_STOPWORDS = {
    "and","or","the","a","an","to","of","in","on","with","for","at","as","by","from",
    "about","into","through","over","after","before","between","without","within","is",
    "are","be","being","been","that","this","these","those","i","you","we","they"
}

def _extract_criteria(job_query: str) -> List[str]:
    """
    Split by commas/semicolons/and/& and words; keep alphanum terms (len>=3) not in stopwords.
    """
    # Replace separators with commas
    s = re.sub(r"[;/|]+", ",", job_query.lower())
    s = re.sub(r"\band\b|\&", ",", s)
    # Tokenize but keep phrases separated by commas if multi-word
    parts = [p.strip() for p in s.split(",") if p.strip()]
    tokens: List[str] = []
    for p in parts:
        # If the part is a short phrase, keep as phrase; else split to words
        words = re.findall(r"[a-z0-9\+\#\.-]{3,}", p)
        if len(words) <= 2 and len(" ".join(words)) <= 25:
            # keep the phrase as a single criterion if it's concise
            phrase = " ".join(w for w in words if w not in _STOPWORDS)
            if phrase:
                tokens.append(phrase)
        else:
            for w in words:
                if w not in _STOPWORDS:
                    tokens.append(w)
    # de-dup preserving order
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _count_matches(criteria: List[str], resume_text: str) -> Tuple[List[str], float]:
    """
    Return (matched_terms, coverage_ratio). Match by word/phrase boundaries (case-insensitive).
    """
    text = resume_text.lower()
    matched: List[str] = []
    for c in criteria:
        pattern = r"\b" + re.escape(c) + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            matched.append(c)
    coverage = (len(matched) / max(1, len(criteria))) if criteria else 0.0
    return matched, coverage

def _extract_resume_skills_one_line(resume_text: str, limit: int = 8) -> str:
    """
    Heuristic: try to read a 'Skills' section; fall back to top frequent keywords.
    Returns a single line 'A, B, C'.
    """
    t = resume_text
    # Try "Skills:" section (up to 300 chars)
    m = re.search(r"skills?\s*[:\-]\s*(.{0,300})", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        chunk = m.group(1)
        # split by comma/semicolon or bullets
        items = re.split(r"[,;\n•\-\u2022]+", chunk)
        items = [re.sub(r"[^A-Za-z0-9\+\#\.\-\s]", "", i).strip() for i in items]
        items = [i for i in items if len(i) >= 2][:limit]
        if items:
            return ", ".join(items)

    # Fallback: keyword frequency
    words = re.findall(r"[A-Za-z0-9\+\#\.\-]{3,}", t)
    words = [w.lower() for w in words if w.lower() not in _STOPWORDS]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]
    return ", ".join([w for w, _ in top])


# -----------------------------
# Prompt template
# -----------------------------
PROMPT_SINGLE_RESUME = """
You are a recruiting assistant.

JOB CRITERIA (list extracted from the job query):
{criteria_list}

RESUME (single candidate):
=== {filename} ===
{resume_text}

TASKS
1) Use ONLY the evidence in this resume to evaluate alignment to the criteria above.
2) Write a concise 3–5 sentence summary that explicitly mentions WHICH of the criteria are matched,
   and briefly justify with very short phrases from the resume (no quotes needed).
3) Output STRICT JSON only with:
{{
  "summary": "<3-5 sentence summary that includes the matched criteria names>",
  "score": <float from 0.0 to 5.0>
}}
""".strip()


def _strict_json_from_model_text(text: str) -> Dict[str, Any]:
    """Parse JSON; if the model adds extra text, extract the first {...} block."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


# -----------------------------
# Ollama (native) path - UPDATED
# -----------------------------
def _flatten_messages_for_generate(messages: list[dict]) -> tuple[str, str]:
    """
    Convert chat messages to system and prompt text for /api/generate.
    """
    system_messages = [m.get("content", "") for m in messages if m.get("role") == "system"]
    user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
    
    system_text = "\n".join(system_messages).strip()
    prompt_text = "\n".join(user_messages).strip()
    
    return system_text, prompt_text


def _call_ollama(provider_cfg: dict, messages: list[dict], temperature: float, max_tokens: int) -> str:
    """
    Universal Ollama caller that works with both /api/chat and /api/generate.
    """
    base_url = provider_cfg.get("base_url", "http://localhost:11434").rstrip("/")
    model = provider_cfg["model"]
    fmt = provider_cfg.get("format")
    
    print(f"DEBUG: Trying to connect to {base_url}")
    print(f"DEBUG: Model: {model}")
    
    # First try /api/chat (preferred for chat-style messages)
    try:
        chat_payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if fmt:
            chat_payload["format"] = fmt
            
        print(f"DEBUG: Trying /api/chat with payload: {json.dumps(chat_payload, indent=2)}")
        
        response = requests.post(f"{base_url}/api/chat", json=chat_payload, timeout=60)
        print(f"DEBUG: /api/chat response status: {response.status_code}")
        print(f"DEBUG: /api/chat response headers: {dict(response.headers)}")
        
        response.raise_for_status()
        data = response.json()
        print(f"DEBUG: /api/chat success!")
        return data.get("message", {}).get("content", "")
        
    except requests.RequestException as chat_error:
        print(f"DEBUG: /api/chat failed: {chat_error}")
        
        # If /api/chat fails, fall back to /api/generate
        try:
            system_text, prompt_text = _flatten_messages_for_generate(messages)
            
            generate_payload = {
                "model": model,
                "prompt": prompt_text,
                "stream": False,
                "options": {
                    "temperature": float(temperature),
                    "num_predict": int(max_tokens),
                },
            }
            
            if system_text:
                generate_payload["system"] = system_text
            if fmt:
                generate_payload["format"] = fmt
                
            print(f"DEBUG: Trying /api/generate with payload: {json.dumps(generate_payload, indent=2)}")
            
            response = requests.post(f"{base_url}/api/generate", json=generate_payload, timeout=60)
            print(f"DEBUG: /api/generate response status: {response.status_code}")
            print(f"DEBUG: /api/generate response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            data = response.json()
            print(f"DEBUG: /api/generate success!")
            return data.get("response", "")
            
        except requests.RequestException as generate_error:
            error_msg = (
                f"Both Ollama endpoints failed.\n"
                f"Chat error: {chat_error}\n"
                f"Generate error: {generate_error}\n"
                f"Check your Ollama configuration and model availability."
            )
            print(f"DEBUG: {error_msg}")
            raise Exception(error_msg)      
        
    except requests.RequestException as chat_error:
        # If /api/chat fails, fall back to /api/generate
        try:
            system_text, prompt_text = _flatten_messages_for_generate(messages)
            
            generate_payload = {
                "model": model,
                "prompt": prompt_text,
                "stream": False,
                "options": {
                    "temperature": float(temperature),
                    "num_predict": int(max_tokens),
                },
            }
            
            if system_text:
                generate_payload["system"] = system_text
            if fmt:
                generate_payload["format"] = fmt
                
            response = requests.post(f"{base_url}/api/generate", json=generate_payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
            
        except requests.RequestException as generate_error:
            raise Exception(
                f"Both Ollama endpoints failed.\n"
                f"Chat error: {chat_error}\n"
                f"Generate error: {generate_error}\n"
                f"Check your Ollama configuration and model availability."
            )


# -----------------------------
# Core evaluation - UPDATED
# -----------------------------
def evaluate_resumes(
    job_query: str,
    resumes: Dict[str, str],
    provider_cfg: Dict[str, Any],
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> List[ResumeResult]:
    """
    Logic:
      - Extract criteria from job_query.
      - For each resume:
          * If coverage < 20% => score 0 and provide a 1-line skills hint.
          * Else, call LLM (Universal Adapter for cloud; native HTTP for Ollama),
            ask for strict JSON {summary, score}, then parse robustly.
    """
    if not isinstance(provider_cfg, dict):
        raise ValueError("provider_cfg must be a dict from config.yml")

    provider = (provider_cfg.get("provider") or "").lower().strip()
    if not provider:
        raise ValueError("Missing 'provider' in provider_cfg")

    # build criteria once
    criteria = _extract_criteria(job_query)
    results: List[ResumeResult] = []

    # Prepare Universal Adapter *only for non-ollama providers*
    adapter: Optional[UniversalLLMAPIAdapter] = None
    if provider != "ollama":
        model = provider_cfg.get("model")
        if not model:
            raise ValueError("Missing 'model' in provider_cfg for universal adapter provider")

        # read API key from env var name (api_key_env) OR direct api_key field
        api_key = provider_cfg.get("api_key_env")
        if not api_key:
            raise ValueError(
                "No API key found. Provide 'api_key_env' (env var name) "
                "or 'api_key' (literal key) in config.yml for this provider."
            )

        # Some adapters use "organization", some use "provider". Use whichever is given.
        org_or_provider = provider_cfg.get("organization") or provider_cfg.get("provider") or "default"

        adapter = UniversalLLMAPIAdapter(
            organization=org_or_provider,
            model=model,
            api_key=api_key,
        )

    for filename, text in resumes.items():
        text = (text or "")
        if len(text) > 20000:
            text = text[:20000] + "\n[TRUNCATED]"

        matched, coverage = _count_matches(criteria, text)
        name_guess = Path(filename).stem

        if criteria and coverage < 0.20:
            # Below 20% coverage → force score 0 and 1-line skills note
            skills_line = _extract_resume_skills_one_line(text, limit=8)
            summary = (
                f"Criteria is not according to the resume of {name_guess}; "
                f"this candidate has these skills: {skills_line}."
            )
            results.append(ResumeResult(file=filename, score=0.0, summary=summary))
            continue

        # Otherwise call the model for this resume
        criteria_list = ", ".join(criteria) if criteria else "(no criteria extracted)"
        prompt_text = PROMPT_SINGLE_RESUME.format(
            criteria_list=criteria_list,
            filename=filename,
            resume_text=text,
        )

        if provider == "ollama":
            # native Ollama: pass plain dict messages
            messages_ollama = [
                {"role": "system", "content": "You output ONLY strict JSON and never include code fences."},
                {"role": "user", "content": prompt_text},
            ]
            raw = _call_ollama(provider_cfg, messages_ollama, temperature, max_tokens)
        else:
            # universal adapter: use its message classes
            messages = [
                Prompt("You output ONLY strict JSON and never include code fences."),
                UserMessage(prompt_text),
            ]
            resp = adapter.generate_chat_answer(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
            )
            # adapter responses typically expose `.content`
            raw = getattr(resp, "content", str(resp))

        # Parse JSON
        try:
            data = _strict_json_from_model_text(raw)
        except Exception:
            # if parsing fails, fall back to a minimal result rather than crash
            results.append(ResumeResult(file=filename, score=0.0, summary=f"[Parse error] {raw[:300]}"))
            continue

        # Normalize fields
        try:
            score = float(data.get("score", 0))
        except Exception:
            score = 0.0
        summary = str(data.get("summary", "")).strip()

        # If the model forgot to mention matched criteria, append them briefly
        if matched and all(m.lower() not in summary.lower() for m in matched[:2]):
            summary = summary.rstrip(". ") + f". Matched criteria: {', '.join(matched)}."

        results.append(ResumeResult(file=filename, score=score, summary=summary))

    return results


# -----------------------------
# Config
# -----------------------------
def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}