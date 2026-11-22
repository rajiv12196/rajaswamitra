"""RajaswaMitra — web interface.

Run:  streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Colleagues on the office network open http://<this-PC-IP>:8501
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from langgraph.types import Command

from revenue_agent import audit
from revenue_agent.graph import GRAPH_DOT, build_graph
from revenue_agent.knowledge import KB_DIR, SUPPORTED, KnowledgeBase
from revenue_agent.llm import LLMConfig, check_connection, get_llm
from revenue_agent.prompts import DOMAINS
from revenue_agent.tools import calculators as calc

st.set_page_config(page_title="RajaswaMitra – Revenue Officer's AI Assistant", page_icon="🏛️", layout="wide")

# On Streamlit Community Cloud, settings come from the app's Secrets.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass
# PUBLIC_DEMO: lock model settings (so the API key can't be sent elsewhere),
# disable uploads and hide the shared audit log.
PUBLIC = os.getenv("RM_PUBLIC_DEMO", "0") == "1"


# ---------------------------------------------------------------- shared resources (one per server)
@st.cache_resource
def shared_graph():
    return build_graph()


@st.cache_resource
def shared_kb():
    return KnowledgeBase()


graph, kb = shared_graph(), shared_kb()

ss = st.session_state
ss.setdefault("thread_id", str(uuid.uuid4()))
ss.setdefault("messages", [])        # [{"role", "content", "trace"?}]
ss.setdefault("pending", None)       # interrupt payload while awaiting officer review


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🏛️ RajaswaMitra")
    st.caption("Private AI assistant for revenue officers. Runs on your own computer or office server.")

    with st.expander("👤 Officer profile", expanded=True):
        name = st.text_input("Name", ss.get("name", ""), placeholder="e.g. A. Sharma")
        designation = st.text_input("Designation", ss.get("designation", ""), placeholder="e.g. Income Tax Officer, Ward 2(1)")
        office = st.text_input("Office", ss.get("office", ""), placeholder="e.g. O/o the ITO, Ward 2(1), Pune")
        ss.update(name=name, designation=designation, office=office)

    if PUBLIC:
        _c = LLMConfig()
        provider, model, base_url = _c.provider, _c.model, _c.base_url
        st.info(f"🌐 Public demo · model: {model}")
    else:
      with st.expander("⚙️ AI model (local)", expanded=False):
          providers = ["ollama", "openai_compatible", "mock"]
          provider = st.selectbox("Provider", providers, index=providers.index(LLMConfig().provider) if LLMConfig().provider in providers else 0,
                                  format_func={"ollama": "Ollama (recommended)", "openai_compatible": "LM Studio / vLLM / llama.cpp",
                                               "mock": "Demo mode (no model)"}.get)
          env_url = LLMConfig().base_url
          default_url = env_url if provider == LLMConfig().provider else ("http://localhost:11434" if provider == "ollama" else "http://localhost:1234/v1")
          model = st.text_input("Model", LLMConfig().model)
          base_url = st.text_input("Server URL", default_url)
          if st.button("Test connection"):
              ok, msg = check_connection(LLMConfig(provider, model, base_url))
              (st.success if ok else st.error)(msg)

    domain_choice = st.selectbox("Domain", ["auto"] + list(DOMAINS),
                                 format_func=lambda d: "Auto-detect" if d == "auto" else DOMAINS[d].split(" (")[0])
    language = st.radio("Answer language", ["en", "hi"], horizontal=True,
                        format_func={"en": "English", "hi": "हिन्दी"}.get)
    if st.button("🆕 New conversation", width="stretch"):
        ss.thread_id, ss.messages, ss.pending = str(uuid.uuid4()), [], None
        st.rerun()
    st.caption("⚠️ Output is a draft aid. The officer remains the decision-maker. Verify every provision and figure.")


def run_config():
    return {"configurable": {
        "thread_id": ss.thread_id,
        "llm": get_llm(LLMConfig(provider, model, base_url)),
        "kb": kb,
        "language": language,
        "officer": {"name": name or "[Name]", "designation": designation or "[Designation]", "office": office or "[Office]"},
    }}


def stream(inp):
    """Run the graph, showing each step live. Returns interrupt payload or None."""
    trace, interrupted = [], None
    with st.status("Agent working…", expanded=True) as status:
        try:
            for update in graph.stream(inp, run_config(), stream_mode="updates"):
                for node, val in update.items():
                    if node == "__interrupt__":
                        interrupted = val[0].value
                        continue
                    for line in (val or {}).get("trace", []) if isinstance(val, dict) else []:
                        trace.append(line)
                        st.write(line)
            status.update(label="Waiting for your review" if interrupted else "Done", state="complete")
        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"{type(e).__name__}: {e}\n\nIs the model server running? Try **Test connection** in the sidebar, "
                     "or switch to Demo mode.")
    return trace, interrupted


# ---------------------------------------------------------------- tabs
t_agent, t_calc, t_kb, t_how, t_audit = st.tabs(["🤖 Agent", "🧮 Calculators", "📚 Knowledge base", "🗺️ How it works", "📜 Audit log"])

# ================= AGENT
with t_agent:
    if PUBLIC:
        st.warning("**Public demo.** Do not enter real taxpayer, landholder or case data. "
                   "Requests are processed by a cloud AI service. For real work, run RajaswaMitra on an office computer.")
    st.markdown("#### Ask a question, request a calculation, or get a notice / order / reply drafted")
    examples = [
        "Compare old and new regime tax for a salaried employee with income of 18 lakh and 80C deductions of 1.5 lakh for FY 2025-26",
        "Draft a show cause notice under GST for delayed payment of tax of Rs 4,50,000 due on 20-07-2025 and paid on 15-10-2025, with interest",
        "Draft a reply to an RTI application asking for copies of a third party's income tax return",
        "Explain the procedure to decide a disputed mutation after a sale deed, and draft the hearing notice",
    ]
    with st.expander("💡 Example requests"):
        for ex in examples:
            if st.button(ex, key=ex):
                ss.queued = ex

    for m in ss.messages:
        with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🏛️"):
            st.markdown(m["content"])
            if m.get("trace"):
                with st.expander("Agent steps"):
                    st.write("\n".join(f"- {t}" for t in m["trace"]))

    # ---- officer review panel (human-in-the-loop)
    if ss.pending:
        p = ss.pending
        with st.container(border=True):
            st.markdown("### 👤 Officer review — nothing is final until you approve")
            rv = p.get("review", {})
            c1, c2 = st.columns(2)
            if rv.get("issues"):
                c1.warning("**Reviewer flagged:**\n" + "\n".join(f"- {i}" for i in rv["issues"]))
            if rv.get("checklist"):
                c2.info("**Please verify:**\n" + "\n".join(f"- {i}" for i in rv["checklist"]))
            if p.get("sources") or p.get("calculations"):
                with st.expander(f"📎 Evidence used: {len(p.get('sources', []))} extract(s), {len(p.get('calculations', []))} calculation(s)"):
                    for h in p.get("sources", []):
                        st.markdown(f"**{h['source']}** (match {h['score']})")
                        st.caption(h["text"][:800])
                    for r in p.get("calculations", []):
                        st.json(r, expanded=False)
            edited = st.text_area("Draft (you may edit directly)", p["draft"], height=420)
            feedback = st.text_input("Instructions for revision (if any)", placeholder="e.g. Give 15 days to reply; add hearing date 10-12-2026")
            b1, b2, b3 = st.columns(3)
            decision = None
            if b1.button("✅ Approve", type="primary", width="stretch"):
                decision = {"decision": "approve", "edited_draft": edited if edited != p["draft"] else ""}
            if b2.button("🔁 Revise with instructions", width="stretch", disabled=not feedback):
                decision = {"decision": "revise", "feedback": feedback, "edited_draft": edited if edited != p["draft"] else ""}
            if b3.button("🛑 Reject", width="stretch"):
                decision = {"decision": "reject"}
            if decision:
                trace, interrupted = stream(Command(resume=decision))
                if interrupted:
                    ss.pending = interrupted
                else:
                    ss.pending = None
                    final = graph.get_state(run_config()).values.get("final", "")
                    ss.messages.append({"role": "assistant", "trace": trace,
                                        "content": final if final else "_Draft rejected. Nothing was finalised._"})
                st.rerun()

    prompt = st.chat_input("Type your request…", disabled=bool(ss.pending)) or ss.pop("queued", None)
    if prompt and not ss.pending:
        history = "\n".join(f"{m['role']}: {m['content'][:1500]}" for m in ss.messages[-6:])
        ss.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)
        trace, interrupted = stream({"request": prompt, "history": history,
                                     "domain_hint": "" if domain_choice == "auto" else domain_choice})
        if interrupted:
            ss.pending = interrupted
            ss.messages[-1]["trace"] = trace
        st.rerun()

    if ss.messages and ss.messages[-1]["role"] == "assistant" and not ss.pending:
        st.download_button("⬇️ Download last answer (.md)", ss.messages[-1]["content"], file_name=f"rajaswamitra_{date.today()}.md")

# ================= CALCULATORS (deterministic, no AI needed)
with t_calc:
    st.markdown("#### Exact calculators — these run on tested code, not the AI")
    st.caption(f"Rates from `config/tax_rules.json`. {calc.load_rules()['_note']}")
    kind = st.selectbox("Calculator", ["Income tax / compare regimes", "Interest u/s 234A", "Interest u/s 234B",
                                       "Interest u/s 234C", "GST interest u/s 50", "Date / limitation period"])
    res = None
    if kind.startswith("Income tax"):
        c1, c2, c3 = st.columns(3)
        gi = c1.number_input("Gross total income (₹)", 0, step=10000, value=1800000)
        fy = c2.selectbox("Financial year", ["2025-26", "2026-27"])
        sal = c3.checkbox("Salaried / pensioner (standard deduction)", True)
        age = c1.selectbox("Age", ["below_60", "60_to_80", "above_80"])
        ded = c2.number_input("Deductions for old regime (80C, 80D, …) ₹", 0, step=5000, value=150000)
        if st.button("Calculate", type="primary"):
            res = calc.compare_regimes(gi, fy, sal, age, ded)
            a, b = res["new_regime"], res["old_regime"]
            df = pd.DataFrame({
                "New regime": [a["total_income"], a["tax_on_total_income"], a["rebate"], a["surcharge"], a["health_education_cess"], a["total_tax_payable"]],
                "Old regime": [b["total_income"], b["tax_on_total_income"], b["rebate"], b["surcharge"], b["health_education_cess"], b["total_tax_payable"]],
            }, index=["Total income", "Tax on income", "Rebate", "Surcharge", "Cess", "Total tax payable"])
            st.dataframe(df.style.format("₹{:,.0f}"), width="stretch")
            st.success(f"Lower tax under the **{res['lower_tax_regime']} regime** — saving ₹{res['saving']:,.0f}")
    elif kind == "Interest u/s 234A":
        c1, c2, c3 = st.columns(3)
        td = c1.number_input("Tax due after TDS/advance tax (₹)", 0, value=50000)
        dd = c2.date_input("Due date of return", date(2025, 9, 15), format="DD/MM/YYYY")
        fd = c3.date_input("Date of filing", date(2025, 12, 10), format="DD/MM/YYYY")
        if st.button("Calculate", type="primary"):
            res = calc.interest_234a(td, dd, fd)
    elif kind == "Interest u/s 234B":
        c1, c2, c3, c4 = st.columns(4)
        at = c1.number_input("Assessed tax (₹)", 0, value=200000)
        ap = c2.number_input("Advance tax paid (₹)", 0, value=100000)
        fy = c3.selectbox("FY", ["2024-25", "2025-26", "2026-27"], index=1)
        up = c4.date_input("Up to date", date(2026, 7, 31), format="DD/MM/YYYY")
        if st.button("Calculate", type="primary"):
            res = calc.interest_234b(at, ap, fy, up)
    elif kind == "Interest u/s 234C":
        tx = st.number_input("Tax on returned income less TDS (₹)", 0, value=300000)
        c = st.columns(4)
        vals = [c[i].number_input(f"Paid by {d} (cumulative)", 0, value=v, key=d)
                for i, (d, v) in enumerate([("15 Jun", 30000), ("15 Sep", 120000), ("15 Dec", 200000), ("15 Mar", 300000)])]
        if st.button("Calculate", type="primary"):
            res = calc.interest_234c(tx, *vals)
            st.dataframe(pd.DataFrame(res["instalments"]), width="stretch")
    elif kind == "GST interest u/s 50":
        c1, c2, c3 = st.columns(3)
        amt = c1.number_input("Tax amount (₹)", 0, value=450000)
        dd = c2.date_input("Due date", date(2025, 7, 20), format="DD/MM/YYYY")
        pd_ = c3.date_input("Payment date", date(2025, 10, 15), format="DD/MM/YYYY")
        itc = st.checkbox("Wrongly availed and utilised ITC (s.50(3))")
        if st.button("Calculate", type="primary"):
            res = calc.gst_interest_s50(amt, dd, pd_, itc)
    else:
        mode = st.radio("Mode", ["Add a period to a date", "Gap between two dates"], horizontal=True)
        c1, c2, c3, c4 = st.columns(4)
        sd = c1.date_input("Start date", date.today(), format="DD/MM/YYYY")
        if mode.startswith("Add"):
            y = c2.number_input("Years", 0, value=0)
            mo = c3.number_input("Months", 0, value=3)
            d = c4.number_input("Days", 0, value=0)
            if st.button("Calculate", type="primary"):
                res = calc.period_calculator(sd, d, mo, y)
        else:
            ed = c2.date_input("End date", date.today(), format="DD/MM/YYYY")
            if st.button("Calculate", type="primary"):
                res = calc.period_calculator(sd, end=ed)
    if res:
        if "interest" in res and not isinstance(res.get("interest"), dict):
            st.metric("Interest", f"₹{res['interest']:,.0f}")
        if "result_date" in res:
            st.metric("Result date", f"{calc.parse_date(res['result_date']):%d-%m-%Y} ({res['weekday']})")
        with st.expander("Full working", expanded=False):
            st.json(res)

# ================= KNOWLEDGE BASE
with t_kb:
    st.markdown("#### Your office's own documents — the agent answers from these")
    st.caption("Add Acts, rules, circulars, instructions, SOPs and formats (PDF, Word, text). They stay on this machine.")
    c1, c2 = st.columns([2, 1])
    up_domain = c2.selectbox("Save under", list(DOMAINS) + ["general"])
    files = None if PUBLIC else c1.file_uploader("Upload documents", type=[e.strip(".") for e in SUPPORTED], accept_multiple_files=True)
    if PUBLIC:
        c1.info("Uploading is turned off in the public demo. In your own copy, add documents here.")
    if files and st.button("Save and index", type="primary"):
        target = KB_DIR if up_domain == "general" else KB_DIR / up_domain
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            (target / Path(f.name).name).write_bytes(f.getvalue())
        st.success(f"Indexed {kb.reload()} passages from {len(kb.files())} files.")
    if st.button("🔄 Reload knowledge base"):
        st.success(f"Indexed {kb.reload()} passages from {len(kb.files())} files.")
    st.dataframe(pd.DataFrame([{"file": str(p.relative_to(KB_DIR)), "size_kb": round(p.stat().st_size / 1024, 1)} for p in kb.files()]),
                 width="stretch", hide_index=True)
    q = st.text_input("Test a search")
    if q:
        for h in kb.search(q, k=5):
            with st.expander(f"{h['source']}  (score {h['score']})"):
                st.write(h["text"])

# ================= HOW IT WORKS
with t_how:
    st.markdown("#### The agent's workflow (LangGraph)")
    st.graphviz_chart(GRAPH_DOT, width="stretch")
    st.markdown("""
| Step | What it does |
|---|---|
| **PII guard** | Replaces PAN, Aadhaar, GSTIN, mobile, e-mail, IFSC and bank account numbers with placeholders before the AI sees the text |
| **Router** | Works out the domain (Income Tax, GST/Customs, Land Revenue, Office) and the type of task |
| **Planner** | Breaks the problem into steps, as an experienced officer would |
| **Research** | Searches your knowledge base and passes the relevant extracts with their source |
| **Calculate** | Picks a calculator and fills in the numbers; the figures come from tested code |
| **Drafter** | Writes the answer or document in Government format, citing sources |
| **Reviewer** | A second pass that checks claims against sources and figures, and sends it back once if needed |
| **Officer review** | Pauses so you can approve, edit, ask for changes or reject |
| **Finalise** | Restores the real identifiers for you and writes a masked audit entry |
""")

# ================= AUDIT
with t_audit:
    st.markdown("#### Audit log (identifiers masked)")
    rows = [] if PUBLIC else audit.recent()
    if PUBLIC:
        st.info("The audit log is hidden in the public demo.")
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No entries yet.")
