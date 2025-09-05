# Resume Ranker (Tkinter GUI + Universal LLM API Adapter)

A desktop tool that **reads resumes (PDF/DOCX/TXT/MD)** from a folder, sends each to a selected **LLM provider** (e.g., **Gemini** via a universal adapter or **Ollama** for local models), and returns a **ranked list** with concise **summaries** based on a **job/prompt** you enter.

> Built from `app_tk.py` (GUI) and `ranker_core.py` (logic). Features include a progress modal, sortable results, CSV/Excel export, and a configurable model dropdown loaded from `config.yml`.

---

## ✨ Features

* **Multimodel support via `config.yml`**

  * Cloud models through a **Universal LLM API Adapter** (e.g., Gemini)
  * Local models via **Ollama** (`/api/chat` with fallback to `/api/generate`)
* **Tkinter GUI** with:

  * Multi‑line prompt input (word wrap + scrollbar)
  * Folder picker for resumes
  * **Progress modal** (cancelable)
  * **Results table** with wrapped summaries & **score-desc sorting**
  * **Clear** action (prompt, folder, table, status)
  * **Export to CSV or Excel (.xlsx)**
  * **Model dropdown** auto‑loaded from `config.yml`
  * Optional **app/dialog icons** via `logo_rafical_ico.ico`
* **Robust resume parsing**: PDF (`pypdf`), DOCX (`docx2txt`), plain text, and Markdown
* **Heuristic criteria extraction** from your job prompt
* **Fail‑soft JSON parsing**: if a model returns extra tokens, first JSON block is parsed
* **Guardrails**:

  * Resumes with <20% criteria coverage are scored **0.0** with a brief skills hint
  * Overlong resumes are truncated to \~20k characters
* **JSON snapshot** of latest run: `rankings_gui.json`

---

## 🖼️ UI Overview

* **Job / Prompt**: Enter skills, requirements, and context. Example:

  > `Senior Python + NLP role; skills: Python, spaCy, FastAPI, vector DB, AWS; 5+ yrs`
* **Resumes folder**: Choose a directory containing `.pdf`, `.docx`, `.txt`, or `.md`.
* **Model**: Pick from the dropdown (sources from `config.yml`).
* **Temperature / Max tokens**: Tuning knobs forwarded to the LLM.
* **Run Ranking**: Processes each resume → **score (0–5)** + **3–5 sentence summary**.
* **Export**: Save results as CSV or Excel (`pandas` + `openpyxl` required for Excel).

---

## 🗂️ Project Structure

```
.
├── app_tk.py           # Tkinter GUI (progress modal, table, exports, config-driven models)
├── ranker_core.py      # Core logic (file I/O, criteria, adapters, scoring)
├── config.yml          # Model/provider configuration (examples below)
├── logo_rafical_ico.ico# (optional) app/dialog icon
└── requirements.txt    # (optional) pin your deps
```

---

## ⚙️ Installation

**Python**: 3.10+ recommended.

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install deps
pip install -U pip
pip install requests pypdf docx2txt pyyaml
# Optional for Excel export
pip install pandas openpyxl

# Universal adapter dependency (your package/source)
# If published on PyPI:
pip install llm-api-adapter  # or the correct package name
# If local path:
# pip install -e ./path/to/llm_api_adapter
```

> **Note on the adapter**: The code imports from `llm_api_adapter`:
>
> ```python
> from llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
> from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
> ```
>
> Make sure this library is installed and compatible.

---

## 🔧 Configuration (`config.yml`)

Define one or more providers; these populate the **Model** dropdown.

```yaml
# config.yml
providers:
  # 1) Gemini (cloud via Universal LLM API Adapter)
  - provider: gemini
    model: gemini-1.5-pro
    # IMPORTANT — see Note below about api_key vs api_key_env
    api_key: ${GEMINI_API_KEY}  # replace at runtime or bake the literal key here

  # 2) Ollama (local)
  - provider: ollama
    model: llama3:8b-instruct  # any local model available to Ollama
    base_url: http://localhost:11434
    # optional: force JSON formatting if your model supports it
    # format: json
```

### ⚠️ Important: API key field naming

In `ranker_core.py`, the universal‑adapter branch expects **either** `api_key` (literal key string) **or** `api_key_env` (intended to be an env var name). *However*, the current implementation **passes the field value directly** to the adapter as the API key without resolving environment variables. In practice, use **`api_key`** and place the actual key string there (or interpolate before load), or modify the code to resolve env vars.

**Quick fix (recommended)**: put the real key in `api_key`, or pre‑expand `"${GEMINI_API_KEY}"` during your config load.

> If you prefer env resolution in code, change:
>
> ```python
> api_key = provider_cfg.get("api_key_env")
> ```
>
> to something like:
>
> ```python
> api_key = os.getenv(provider_cfg.get("api_key_env")) or provider_cfg.get("api_key")
> ```

---

## ▶️ Running the App

```bash
python app_tk.py
```

1. Choose a **Resumes folder**.
2. Enter a **Job / Prompt** (skills, requirements, seniority, domain).
3. Select a **Model** from the dropdown.
4. Click **Run Ranking**.
5. Watch the **Progress** dialog; cancel if needed.
6. Review **scores** (sorted high→low) and **summaries**.
7. **Export** to CSV or Excel.

> Results are also saved to `rankings_gui.json` with the “job\_query”, selected model label, and detailed results.

---

## 🤖 How Scoring Works (high‑level)

1. **Extract criteria** from your prompt (split on commas/and/&; remove stopwords; keep short phrases)
2. For each resume:

   * If **coverage < 20%** of criteria → assign **score 0.0** + short skills line
   * Else, call the selected **LLM** with a strict‑JSON prompt asking for:

     ```json
     {
       "summary": "<3-5 sentence summary mentioning matched criteria>",
       "score": <float 0.0–5.0>
     }
     ```
   * Parse the first JSON object found in the response (robust to extra text)
   * If the summary forgot explicit criteria names, append a brief “Matched criteria: …” list

**Ollama path**: tries `/api/chat` first, then falls back to `/api/generate` with a flattened system/user prompt. Temperature and max token hints are forwarded in `options`.

---

## 📦 Input Formats

* **PDF**: extracted via `pypdf` (text only; complex layouts may degrade)
* **DOCX**: extracted via `docx2txt`
* **TXT/MD**: read as UTF‑8 (errors ignored)

> Very long resumes are truncated to \~20,000 chars with `[TRUNCATED]` marker.

---

## 🧪 Example Prompts

* *“Backend engineer; Go, PostgreSQL, Docker, Kubernetes, AWS; 4+ years; fintech is a plus”*
* *“ML engineer with Python, scikit‑learn, XGBoost, experiment tracking; strong SQL; MLOps a bonus”*
* *“Data analyst; SQL, dbt, Tableau/Power BI, Python (pandas); stakeholder communication”*

---

## 🛠️ Troubleshooting

* **No providers in dropdown** → Check `config.yml` → `providers:` is present and valid.
* **Gemini/Cloud errors** → Ensure `api_key` is set (see **Important** note), network access is allowed, and the adapter supports your provider/model.
* **Ollama errors** → Verify `ollama serve` is running, `base_url` is correct, and the model is pulled (`ollama pull <model>`).
* **Excel export disabled** → Install `pandas` and `openpyxl`.
* **JSON parse errors** → The app will fall back to a `score=0.0` with partial content; consider lowering temperature or enabling JSON mode (where available).

---

## 🔒 Security & Privacy

* Resumes are processed locally for parsing.
* If you choose a **cloud** provider, **resume text is sent** to that provider. Review your data policies.
* Consider redaction (PII) or a **local model via Ollama** for sensitive data.

---

## 🗺️ Roadmap Ideas

* RAG‑style skill extraction/validation
* Per‑criterion scoring breakdown
* Deduplication & candidate metadata extraction (email, phone, links)
* Batch prompts / saved roles
* Fine‑tuned prompts per role family
* Headless/CLI mode

---

## 📄 License

Choose a license (e.g., MIT) and add it here.

---

## 🙌 Acknowledgements

* [`pypdf`](https://pypi.org/project/pypdf/), [`docx2txt`](https://pypi.org/project/docx2txt/)
* [`Ollama`](https://ollama.com) for local models
* Your **Universal LLM API Adapter** for one‑stop cloud model access
