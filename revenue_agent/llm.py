"""Language-model providers — all local.

- ollama : Ollama on this PC or an office server (default http://localhost:11434)
- openai_compatible : any local OpenAI-style server (LM Studio, vLLM, llama.cpp server)
- mock : no model at all; deterministic stand-in so the app and tests run anywhere
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("RM_PROVIDER", "ollama"))
    model: str = field(default_factory=lambda: os.getenv("RM_MODEL", "qwen2.5:7b"))
    base_url: str = field(default_factory=lambda: os.getenv("RM_BASE_URL", "http://localhost:11434"))
    temperature: float = 0.1


class MockLLM:
    """Rule-based stand-in. Reads the ROLE tag in the system prompt."""

    def invoke(self, messages):
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        user = messages[-1].content
        role = re.search(r"ROLE:\s*(\w+)", system)
        role = role.group(1) if role else "DRAFTER"
        return AIMessage(content=getattr(self, f"_{role.lower()}", self._drafter)(user))

    def _router(self, text):
        t = text.lower()
        domain = ("gst_customs" if any(w in t for w in ["gst", "itc", "customs", "gstr", "cgst", "e-way"]) else
                  "land_revenue" if any(w in t for w in ["mutation", "khasra", "patwari", "tehsil", "land", "jamabandi", "khatauni", "survey no"]) else
                  "income_tax" if any(w in t for w in ["income", "itr", "tds", "234", "assessment", "pan", "regime", "advance tax"]) else
                  "office_admin")
        task = ("drafting" if any(w in t for w in ["draft", "notice", "letter", "reply", "order", "write"]) else
                "calculation" if any(w in t for w in ["calculate", "compute", "interest", "how much", "tax on", "days", "date"]) else
                "question")
        return json.dumps({"domain": domain, "task_type": task, "language": "en"})

    def _planner(self, text):
        return json.dumps({"steps": ["Identify facts and the period involved",
                                     "Find applicable provisions in the knowledge base",
                                     "Run any calculation with the calculator tools",
                                     "Draft the answer / document with citations",
                                     "List points the officer must verify"]})

    def _extractor(self, text):
        t = text.lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lac|crore|cr)", t)
        if m and any(w in t for w in ["tax", "regime", "income", "salary"]):
            amt = float(m.group(1)) * (1e5 if m.group(2).startswith("la") else 1e7)
            name = "compare_regimes" if "regime" in t or "compare" in t else "income_tax"
            return json.dumps({"calls": [{"name": name, "args": {"gross_income": amt, "is_salaried": "salar" in t}}]})
        return json.dumps({"calls": []})

    def _drafter(self, text):
        return ("**[Demo mode — no language model connected]**\n\n"
                "This is a placeholder draft so you can see the workflow. Connect Ollama in the sidebar "
                "to get real answers.\n\nThe extracts and calculations the agent found are listed under "
                "'Evidence used'. A real model writes the answer from them.\n\n"
                "**Points to verify:** applicable period, current notification, figures.")

    def _reviewer(self, text):
        return json.dumps({"issues": [], "verdict": "ok", "checklist": ["Verify section numbers against the Act in force for the period"]})


def get_llm(cfg: LLMConfig | None = None):
    cfg = cfg or LLMConfig()
    if cfg.provider == "mock":
        return MockLLM()
    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=cfg.model, base_url=cfg.base_url, temperature=cfg.temperature, num_ctx=8192)
    if cfg.provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=os.getenv("RM_API_KEY", "local"),
                          temperature=cfg.temperature, max_tokens=int(os.getenv("RM_MAX_TOKENS", "1500")),
                          max_retries=4, timeout=120)
    raise ValueError(f"Unknown provider {cfg.provider}")


def chat(llm, system: str, user: str) -> str:
    out = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content
    if isinstance(out, list):  # some providers return content blocks
        out = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in out)
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


def parse_json(text: str, default: dict) -> dict:
    """Pull the first JSON object out of a model reply; fall back to default."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return default


def check_connection(cfg: LLMConfig) -> tuple[bool, str]:
    try:
        reply = chat(get_llm(cfg), "ROLE: PING. Reply with the single word OK.", "ping")
        return True, f"Connected: {cfg.provider} / {cfg.model} replied '{reply.strip()[:40]}'"
    except Exception as e:
        return False, f"Not reachable: {type(e).__name__}: {e}"
