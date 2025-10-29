## 🏥 Unified Patient Intelligence (UPI) — Agentic AI for Healthcare

Transform fragmented clinical data into actionable, clinician-ready intelligence with a layered, multi‑agent system. This project demonstrates an end‑to‑end agentic pipeline that continuously aggregates patient data, reasons over trends, summarizes what matters, and drafts next actions — a “continuous clinical co‑pilot.”

---

### 🩺 Problem Statement: Clinical Data Overload & Fragmented Care

Healthcare data is exploding: clinical notes, labs, imaging, prescriptions, EHR entries, wearables. Yet, less than 3% is used for decisions (GE HealthCare, 2025). Clinicians must manually reconcile siloed systems under time pressure, leading to:

- Missed warning signs (e.g., subtle creatinine rise hidden in reports)
- Disjointed care across departments
- Cognitive overload and burnout

What’s missing is a reasoning layer that continuously monitors, integrates, and interprets data across silos.

---

### 💡 Solution: Agentic AI for Unified Patient Intelligence (UPI)

An agentic system that autonomously ingests multi‑modal patient data, reasons over longitudinal trends, summarizes the clinical picture, and drafts care coordination steps. It acts as a continuous clinical co‑pilot — always watching, flagging, and briefing.

---

### 🧩 Core Capabilities

- **Data Aggregation Agent**: Connects to EHR/lab/imaging sources (mocked via SQLite) and normalizes into a unified view
- **Clinical Reasoning Agent**: Detects abnormalities and risky trends across time
- **Summarization Agent**: Produces concise, clinician‑ready patient summaries
- **Coordination Agent (planned)**: Drafts referrals, follow‑ups, reminders, and workflow automations
- **Learning Loop (planned)**: Adapts alerts/summaries to clinician preferences

---

### ⚙️ Layered Agentic Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                              User Layer                               │
│                 Clinician dashboard / feedback (planned)              │
└──────────────────────────────────────────────────────────────────────┘
                 ▲                                        │
                 │                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           Action Layer (planned)                      │
│     Referrals • Scheduling • Alerts • EHR pushes • Email/Calendar     │
└──────────────────────────────────────────────────────────────────────┘
                 ▲                                        │
                 │                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       Reasoning Layer (this repo)                     │
│  Coordinator ──► Lab Agent ──► Summarization Agent ──► Draft Actions  │
│          ▲                │                 ▲                         │
│          │                └────── uses data from ──────┘              │
└──────────────────────────────────────────────────────────────────────┘
                 ▲
                 │
┌──────────────────────────────────────────────────────────────────────┐
│                          Data Layer (this repo)                       │
│     Data Agent • SQLite (mock EHR/labs) • Pandas • Schema discovery   │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 🧪 Hackathon/MVP Scope

Goal: show an agent can unify and reason over fragmented clinical data and produce clinician‑useful outputs.

- Mock dataset: SQLite database (`healthcare.db`) with lab reports and patient info
- Flow: Data Aggregation → Reasoning (flags/trends) → Summarization → Draft referral
- Output: Console printout per patient with flags, explanation, summary, referral draft

Example target output:

```text
Patient: John Doe
Flag: Worsening kidney function
Summary: eGFR dropped 12% since last visit; HbA1c rising.
Action: Auto‑scheduled nephrology referral (pending approval).
```

---

### 🗂️ Repository Structure

```text
Uaf-Hackathon/
├─ README.md
├─ requirements.txt
├─ healthcare.db                      # optional copy at root (primary under src/mock_data)
└─ src/
   ├─ main.py                         # CLI demo: runs the end‑to‑end workflow
   ├─ backend/
   │  └─ db_connector.py              # (foundation for API/server integration)
   ├─ Agents/
   │  ├─ orchestrator.py              # Coordinator: wires agents in a LangGraph‑style DAG
   │  ├─ data_agent.py                # Data access, schema discovery, NL→SQL intent
   │  ├─ lab_agent.py                 # Clinical reasoning over lab trends
   │  ├─ summarizer_agent.py          # Clinician‑style summary + referral draft
   │  ├─ referral_agent.py            # (planned) action/coordination
   │  ├─ clinical_bot.py              # (optional bot interface)
   │  ├─ prompts.py                   # Prompt templates
   │  ├─ schemas.py                   # Pydantic models
   │  └─ voiceai/                     # (optional) voice interfaces
   └─ mock_data/
      └─ healthcare.db                # mock SQLite with lab_reports, etc.
```

---

### 🧰 Tech Stack

- **Python**: Orchestration, agents, data pipeline
- **LangChain/LangGraph (style)**: Multi‑agent reasoning workflow
- **LLM**: Groq/OpenAI/Gemini via LangChain integrations
- **Data**: Pandas + SQLite (mock EHR); FAISS/Chroma (embeddings, optional)
- **API (planned)**: FastAPI + Uvicorn
- **Monitoring (optional)**: Weights & Biases / MLflow

---

### 🚀 Quickstart (Windows‑friendly)

Prerequisites:

- Python 3.10+
- A Groq API key (recommended) or alternative LLM provider

1) Clone and enter the project

```powershell
git clone <your-repo-url> Uaf-Hackathon
cd Uaf-Hackathon
```

2) Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

3) Set environment variables (.env recommended)

Create a `.env` file at the repo root:

```env
GROQ_API_KEY=your_groq_api_key_here
# Optional: for other providers if you switch models
# OPENAI_API_KEY=...
# GOOGLE_API_KEY=...
```

4) Ensure the mock database exists

- The demo expects `src/mock_data/healthcare.db`.
- If you only have `healthcare.db` at the repo root, copy it:

```powershell
copy .\healthcare.db .\src\mock_data\healthcare.db
```

5) Run the end‑to‑end demo

```powershell
python -m src.main
```

You should see per‑patient output with:

- Available tables
- Lab flags (abnormal findings, high‑risk diseases)
- Explanations from the reasoning agent
- Clinician summary and referral draft

---

### 🔍 How It Works (At a Glance)

- `Coordinator` orchestrates a simple DAG: `DataAgent → LabAgent → SummarizationAgent`.
- `DataAgent` queries SQLite via Pandas, discovers schemas, and can interpret NL queries.
- `LabAgent` inspects longitudinal labs to surface flags and risks; augments with LLM explanation.
- `SummarizationAgent` combines agent outputs into a concise clinician briefing and a referral draft.

---

### 🛡️ Notes on Safety & Scope

- This is a hackathon/MVP prototype for research/education; not for clinical use.
- Outputs may be incomplete or inaccurate; always require human clinical oversight.

---

### 📈 Roadmap

Phase | Milestone | Outcome
--- | --- | ---
MVP | Multi‑agent integration over mock data | Working demo for hackathons
Phase 2 | Add FHIR/EHR connectors | Real interoperability
Phase 3 | Domain fine‑tuning on clinical text | Higher accuracy & trust
Phase 4 | Pilot deployments | Reduce cognitive overload & missed diagnoses

Planned near‑term enhancements:

- FastAPI endpoints to expose agent outputs to a frontend
- React/Next.js dashboard for flagged patients and summaries
- Vector embeddings (FAISS/Chroma) for semantic retrieval over notes/imaging
- Feedback loop to adapt summaries to clinician preferences
- Expanded coordination agent for referrals/scheduling/alerts

---

### 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue to discuss what you’d like to change. Add tests where practical and keep the agent interfaces clear and modular.

---

### 📜 License

Add your preferred license here (e.g., MIT).



