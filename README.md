# 🏛️ RajaswaMitra — a private AI agent for Indian revenue officers

RajaswaMitra is an agentic AI assistant for officers in **Income Tax, GST & Customs, State Land Revenue** and **general office work** (RTI, correspondence, establishment). It runs **entirely on your own computer or office server**. No case data goes to the internet.

What it does for you:
- Answers questions from **your office's own documents** (Acts, circulars, SOPs) and names the source it used.
- Runs **exact calculators**: income tax under the old and new regimes, interest u/s 234A/B/C, GST interest u/s 50, and date or limitation periods.
- **Drafts** show-cause notices, orders, hearing notices, RTI replies and letters in Government format, in English or Hindi.
- **Checks its own draft** in a reviewer step, then **waits for you to approve, edit, revise or reject** it.
- **Masks** PAN, Aadhaar, GSTIN, mobile numbers, e-mail addresses, IFSC codes and bank account numbers before the AI sees them, and keeps an **audit log**.

> ⚠️ This is a drafting and research aid. The officer is always the decision-maker. Verify every provision, rate and figure before you rely on it.

---

## 1. Quick start on one computer (about 15 minutes)

1. **Install Python 3.11 or later** from python.org. On Windows, tick *"Add Python to PATH"*.
2. **Install Ollama**, the local AI engine, from https://ollama.com. Then open a terminal and run:
   ```
   ollama pull qwen2.5:7b
   ```
   This model handles Hindi and English well and runs on a PC with 16 GB of RAM. Other choices: `llama3.1:8b`, `gemma2:9b`, or `qwen2.5:14b` if you have a GPU.
3. **Start RajaswaMitra:**
   - Windows: double-click `run_windows.bat`
   - Linux or macOS: `./run_linux.sh`
4. A browser tab opens at http://localhost:8501. In the sidebar, enter your name and designation, then click **Test connection**.

No model yet? Choose **Demo mode** in the sidebar to try the workflow and the calculators.

## 2. Share it with your whole office

The launch scripts already listen on the office network. Colleagues open
**`http://<your-PC-IP>:8501`** in their browser. To find your IP address, run `ipconfig` on Windows or `hostname -I` on Linux. Each person gets their own conversation, while the model and the knowledge base are shared.

If colleagues can't connect, allow port 8501 through the Windows firewall.

**For a server serving many users**, run it with Docker:
```
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:7b
```
For a GPU, uncomment the GPU line in `docker-compose.yml`.

**To share it with other offices**, give them this folder (zip it or put it on an internal Git server). It contains no case data. The audit log in `data/` stays local.

> Keep it on the internal network. Don't expose port 8501 to the internet without adding NIC/departmental authentication in front of it (for example a reverse proxy with SSO).

## 2b. Free public demo link (Streamlit Community Cloud + Groq)

For a demo that anyone can open, with no real case data:
1. Get a free API key at https://console.groq.com/keys.
2. Put this folder in a GitHub repository. Don't commit `.streamlit/secrets.toml`.
3. At https://share.streamlit.io choose **Create app**, pick the repository, and set the main file to `app.py`. Under **Advanced settings**, choose Python 3.12 and paste the contents of `.streamlit/secrets.toml.example`, with your real key, into **Secrets**.
4. Deploy. You get a link like `https://<name>.streamlit.app`.

`RM_PUBLIC_DEMO = "1"` locks the model settings, so visitors can't send your key to another server. It also turns off uploads and hides the audit log, and shows a "no real data" warning.

## 3. Add your own documents (this matters most for accuracy)

Open the **📚 Knowledge base** tab and upload PDFs or Word files of Acts, rules, notifications, circulars and SOPs, under the right domain. You can also copy files into `knowledge_base/<domain>/` and click **Reload**.
The sample notes included in that folder are short summaries that show the format. Replace them with the official texts.

## 4. Keep the rates up to date

Tax slabs, rebate, surcharge and interest rates are stored in **`config/tax_rules.json`**, so you can change them without touching the code. They currently hold FY 2025-26, and FY 2026-27, for which Budget 2026 made no slab changes. To add a new year, add a block, or point it at an earlier year with `"same_as"`.
Note that the **Income-tax Act, 2025 applies from 1 April 2026**. Earlier years stay under the 1961 Act, and the calculators label which Act they used.

## 5. How the agent works (LangGraph)

```
START → PII guard → Router → Planner → Research (knowledge base) → Calculate (tools)
      → Drafter → Reviewer ──(issues: one self-correction)──► Drafter
                     │
                     ▼
              Officer review  (graph PAUSES: approve / edit / revise / reject)
                     │ revise ─► Drafter
                     ▼
                  Finalise (restore identifiers for the officer, write audit log) → END
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface: Agent, Calculators, Knowledge base, How it works, Audit log |
| `revenue_agent/graph.py` | LangGraph state, nodes, routing and the human-in-the-loop interrupt |
| `revenue_agent/prompts.py` | Role prompts (router, planner, extractor, drafter, reviewer) |
| `revenue_agent/tools/calculators.py` | Tested calculators. The AI only selects them and fills in the inputs |
| `revenue_agent/knowledge.py` | Local document search (TF-IDF, no model download needed) |
| `revenue_agent/pii.py` | Masks and restores identifiers |
| `revenue_agent/llm.py` | Model providers: Ollama, OpenAI-compatible local servers, Demo |
| `revenue_agent/audit.py` | SQLite audit trail, storing masked text only |
| `config/tax_rules.json` | Editable rate tables |
| `tests/` | `python -m pytest tests` checks the calculators, masking and the full graph |

### Use other local model servers
Choose **LM Studio / vLLM / llama.cpp** in the sidebar and set the server URL, for example `http://localhost:1234/v1`. Or set these environment variables: `RM_PROVIDER`, `RM_MODEL`, `RM_BASE_URL`.

### Extend it
- **New calculator:** add a function to `calculators.py`, register it in `CALCULATORS` and `CALCULATOR_SPECS`, and add a test.
- **New domain**, for example Stamps & Registration or Excise: add it to `DOMAINS` in `prompts.py` and create a `knowledge_base/<domain>/` folder.
- **New step**, for example a precedent search: add a node in `build_graph()`.

## 6. Limits
- The calculators cover the normal-rate income of resident individuals. They don't cover special-rate capital gains, firms or companies.
- The AI can make mistakes. The reviewer step and your approval are safeguards, not guarantees.
- Answers are only as good as the documents you load. Keep the knowledge base current.
