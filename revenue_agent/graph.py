"""The RajaswaMitra agent as a LangGraph state machine.

  START → pii_guard → router → planner → research → calculate → drafter → reviewer
                                                          ↑          │
                                                          └─revise───┤ (max 1 self-revision)
                                                                     ↓
                                    officer_review (INTERRUPT: approve / revise / reject)
                                         │ revise → drafter
                                         ↓
                                      finalize → END

Runtime objects (llm, knowledge base, officer profile) are passed in
config["configurable"] so one compiled graph serves every user on the LAN.
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import audit, prompts
from .llm import chat, parse_json
from .pii import mask, unmask
from .tools.calculators import run_calculator

MAX_SELF_REVISIONS = 1


class AgentState(TypedDict, total=False):
    request: str
    masked_request: str
    pii_map: dict
    history: str
    domain_hint: str
    domain: str
    task_type: str
    language: str
    plan: list[str]
    retrieved: list[dict]
    tool_results: list[dict]
    draft: str
    review: dict
    self_revisions: int
    officer_feedback: str
    decision: str
    final: str
    trace: Annotated[list[str], operator.add]


def _cfg(config: RunnableConfig) -> dict:
    return config.get("configurable", {})


# ------------------------------------------------------------------ nodes

def pii_guard(state: AgentState, config: RunnableConfig):
    masked, mapping = mask(state["request"])
    hist_masked, mapping = mask(state.get("history", ""), mapping)
    n = len(mapping)
    return {"masked_request": masked, "pii_map": mapping, "history": hist_masked,
            "self_revisions": 0, "officer_feedback": "", "decision": "",
            "trace": [f"🛡️ PII guard: masked {n} identifier(s)" if n else "🛡️ PII guard: no identifiers found"]}


def router(state: AgentState, config: RunnableConfig):
    llm = _cfg(config)["llm"]
    out = parse_json(chat(llm, prompts.ROUTER, state["masked_request"]),
                     {"domain": "office_admin", "task_type": "question", "language": "en"})
    domain = state.get("domain_hint") or out.get("domain", "office_admin")
    if domain not in prompts.DOMAINS:
        domain = "office_admin"
    task = out.get("task_type", "question")
    if task not in ("question", "calculation", "drafting", "analysis"):
        task = "question"
    lang = _cfg(config).get("language") or out.get("language", "en")
    return {"domain": domain, "task_type": task, "language": lang,
            "trace": [f"🧭 Router: domain = {domain}, task = {task}"]}


def planner(state: AgentState, config: RunnableConfig):
    llm = _cfg(config)["llm"]
    user = f"Domain: {prompts.DOMAINS[state['domain']]}\nTask type: {state['task_type']}\nRequest: {state['masked_request']}"
    steps = parse_json(chat(llm, prompts.PLANNER, user), {"steps": []}).get("steps", [])[:6]
    return {"plan": [str(s) for s in steps], "trace": [f"🗺️ Planner: {len(steps)} step(s)"]}


def research(state: AgentState, config: RunnableConfig):
    kb = _cfg(config).get("kb")
    hits = kb.search(state["masked_request"], k=3, domain=state["domain"]) if kb else []
    return {"retrieved": hits,
            "trace": [f"📚 Knowledge base: {len(hits)} relevant extract(s)" +
                      (f" from {', '.join(sorted({h['source'] for h in hits}))}" if hits else "")]}


def calculate(state: AgentState, config: RunnableConfig):
    llm = _cfg(config)["llm"]
    calls = parse_json(chat(llm, prompts.EXTRACTOR, state["masked_request"]), {"calls": []}).get("calls", [])
    results = []
    for c in calls[:4]:
        if isinstance(c, dict) and c.get("name"):
            results.append(run_calculator(c["name"], c.get("args") or {}))
    names = ", ".join(r.get("calculator", "?") for r in results) or "none needed"
    return {"tool_results": results, "trace": [f"🧮 Calculators: {names}"]}


def drafter(state: AgentState, config: RunnableConfig):
    c = _cfg(config)
    officer = c.get("officer", {})
    lang = "Hindi (Devanagari) with legal terms also in English in brackets" if state.get("language") == "hi" else "English"
    parts = [
        f"Officer: {officer.get('name', '[Name]')}, {officer.get('designation', '[Designation]')}, {officer.get('office', '[Office]')}",
        f"Domain: {prompts.DOMAINS[state['domain']]}  |  Task type: {state['task_type']}",
        f"Earlier conversation (may be empty):\n{state.get('history', '')[-3000:]}",
        f"REQUEST:\n{state['masked_request']}",
        "PLAN:\n" + "\n".join(f"- {s}" for s in state.get("plan", [])),
        "KNOWLEDGE-BASE EXTRACTS:\n" + ("\n\n".join(f"[source: {h['source']}]\n{h['text'][:900]}" for h in state.get("retrieved", []))
                                         or "(none found — say that provisions must be verified)"),
        "CALCULATOR RESULTS (authoritative figures):\n" + (json.dumps(state.get("tool_results", []), ensure_ascii=False)
                                                          if state.get("tool_results") else "(none)"),
    ]
    if state.get("review", {}).get("issues") and state.get("draft"):
        parts.append("PREVIOUS DRAFT:\n" + state["draft"])
        parts.append("REVIEWER ISSUES TO FIX:\n" + "\n".join(f"- {i}" for i in state["review"]["issues"]))
    if state.get("officer_feedback"):
        parts.append("PREVIOUS DRAFT:\n" + state.get("draft", ""))
        parts.append("OFFICER'S INSTRUCTIONS (highest priority):\n" + state["officer_feedback"])
    draft = chat(c["llm"], prompts.DRAFTER % lang, "\n\n".join(parts))
    return {"draft": draft, "trace": ["✍️ Drafter: draft ready" + (" (revised)" if state.get("draft") else "")]}


def reviewer(state: AgentState, config: RunnableConfig):
    llm = _cfg(config)["llm"]
    user = ("EXTRACTS:\n" + "\n".join(f"[{h['source']}] {h['text'][:400]}" for h in state.get("retrieved", [])) +
            "\n\nCALCULATOR RESULTS:\n" + json.dumps(state.get("tool_results", []), ensure_ascii=False) +
            "\n\nDRAFT:\n" + state["draft"])
    review = parse_json(chat(llm, prompts.REVIEWER, user), {"verdict": "ok", "issues": [], "checklist": []})
    review.setdefault("issues", [])
    review.setdefault("checklist", [])
    return {"review": review,
            "trace": [f"🔎 Reviewer: {review.get('verdict', 'ok')} — {len(review['issues'])} issue(s)"]}


def after_review(state: AgentState) -> str:
    if state["review"].get("verdict") == "revise" and state["review"]["issues"] and state.get("self_revisions", 0) < MAX_SELF_REVISIONS:
        return "self_revise"
    return "officer_review"


def self_revise(state: AgentState):
    return {"self_revisions": state.get("self_revisions", 0) + 1, "trace": ["🔁 Sending back to drafter for self-correction"]}


def officer_review(state: AgentState) -> Command[Literal["drafter", "finalize"]]:
    """Pause the graph. The GUI shows the draft and resumes with the officer's decision."""
    answer = interrupt({"draft": unmask(state["draft"], state.get("pii_map", {})),
                        "review": state.get("review", {}),
                        "sources": state.get("retrieved", []),
                        "calculations": state.get("tool_results", [])})
    decision = answer.get("decision", "approve")
    update: dict[str, Any] = {"decision": decision, "trace": [f"👤 Officer: {decision}"]}
    if answer.get("edited_draft"):
        masked_edit, mapping = mask(answer["edited_draft"], state.get("pii_map", {}))
        update["draft"], update["pii_map"] = masked_edit, mapping
    if decision == "revise":
        update["officer_feedback"] = answer.get("feedback", "")
        update["review"] = {}
        return Command(goto="drafter", update=update)
    return Command(goto="finalize", update=update)


def finalize(state: AgentState, config: RunnableConfig):
    c = _cfg(config)
    decision = state.get("decision") or "approve"
    final = unmask(state["draft"], state.get("pii_map", {})) if decision == "approve" else ""
    audit.log(c.get("officer", {}).get("name", "unknown"), c.get("thread_id", ""), state.get("domain"),
              state.get("task_type"), state["masked_request"], decision, state["draft"])
    return {"final": final, "trace": ["✅ Finalised and logged" if decision == "approve" else "🛑 Rejected and logged"]}


# ------------------------------------------------------------------ build

def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    for name, fn in [("pii_guard", pii_guard), ("router", router), ("planner", planner),
                     ("research", research), ("calculate", calculate), ("drafter", drafter),
                     ("reviewer", reviewer), ("self_revise", self_revise),
                     ("officer_review", officer_review), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "pii_guard")
    g.add_edge("pii_guard", "router")
    g.add_edge("router", "planner")
    g.add_edge("planner", "research")
    g.add_edge("research", "calculate")
    g.add_edge("calculate", "drafter")
    g.add_edge("drafter", "reviewer")
    g.add_conditional_edges("reviewer", after_review, {"self_revise": "self_revise", "officer_review": "officer_review"})
    g.add_edge("self_revise", "drafter")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


GRAPH_DOT = """digraph G {
 rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
 START [shape=circle, label="", width=0.25, fillcolor="#333"]; END [shape=doublecircle, label="", width=0.2, fillcolor="#333"];
 START -> pii_guard -> router -> planner -> research -> calculate -> drafter -> reviewer;
 reviewer -> self_revise [label="issues", style=dashed]; self_revise -> drafter [style=dashed];
 reviewer -> officer_review [label="ok"];
 officer_review [fillcolor="#fff4d6", label="officer_review\\n(human approval)"];
 officer_review -> drafter [label="revise", style=dashed];
 officer_review -> finalize [label="approve / reject"]; finalize -> END;
}"""
