import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from langgraph.types import Command

from revenue_agent.graph import build_graph
from revenue_agent.knowledge import KnowledgeBase
from revenue_agent.llm import LLMConfig, get_llm
from revenue_agent.pii import mask, unmask
from revenue_agent.tools import calculators as c


# ---------------- income tax (FY 2025-26, values worked by hand)
def test_new_regime_12_75_lakh_salary_nil():
    r = c.compute_income_tax(1275000, regime="new", is_salaried=True)
    assert r["total_income"] == 1200000 and r["total_tax_payable"] == 0


def test_new_regime_marginal_relief_just_above_12_lakh():
    r = c.compute_income_tax(1210000, regime="new", is_salaried=False)
    # slab tax 61,500; relief limits tax to the 10,000 excess over 12 lakh; +4% cess
    assert r["tax_on_total_income"] == 61500 and r["rebate"] == 51500
    assert r["total_tax_payable"] == 10400


def test_new_regime_18_lakh_salary():
    r = c.compute_income_tax(1800000, regime="new", is_salaried=True)
    # TI 17,25,000: 20k + 40k + 60k + 25% ... = 20000+40000+60000+(125000*0.20)=145000 ; cess 5800
    assert r["total_income"] == 1725000
    assert r["tax_on_total_income"] == 145000
    assert r["total_tax_payable"] == 150800


def test_old_regime_basic():
    r = c.compute_income_tax(1000000, regime="old", is_salaried=True, deductions=150000)
    # TI 8,00,000: 12,500 + 60,000 = 72,500 ; cess 2,900
    assert r["total_income"] == 800000 and r["total_tax_payable"] == 75400


def test_old_regime_rebate_5_lakh():
    assert c.compute_income_tax(500000, regime="old", is_salaried=False)["total_tax_payable"] == 0


def test_surcharge_marginal_relief_new_regime():
    r = c.compute_income_tax(5010000, regime="new", is_salaried=False)
    tax_50l = c.compute_income_tax(5000000, regime="new", is_salaried=False)["tax_on_total_income"]
    assert r["tax_on_total_income"] + r["surcharge"] == pytest.approx(tax_50l + 10000, abs=1)


def test_fy_2026_27_inherits_rates():
    assert c.compute_income_tax(1800000, fy="2026-27")["total_tax_payable"] == 150800


# ---------------- interest and dates
def test_months_or_part():
    from datetime import date
    assert c.months_or_part(date(2025, 7, 31), date(2025, 8, 31)) == 1
    assert c.months_or_part(date(2025, 7, 31), date(2025, 9, 1)) == 2
    assert c.months_or_part(date(2026, 3, 31), date(2026, 7, 31)) == 4


def test_234a():
    r = c.interest_234a(50050, "15-09-2025", "10-12-2025")
    assert r["principal_rounded"] == 50000 and r["months"] == 3 and r["interest"] == 1500


def test_234b_not_applicable_when_90pct_paid():
    assert c.interest_234b(100000, 95000, "2025-26", "31-07-2026")["interest"] == 0


def test_234b():
    r = c.interest_234b(200000, 100000, "2025-26", "31-07-2026")
    assert r["months"] == 4 and r["interest"] == 4000


def test_234c():
    r = c.interest_234c(300000, 30000, 120000, 200000, 300000)
    # Jun: 30k < 12% (36k) -> shortfall 45k-30k=15k*3% = 450
    # Sep: 120k >= 36% (108k) -> 0 ; Dec: 225k-200k=25k*3% = 750 ; Mar: 0
    assert r["interest"] == 1200


def test_gst_interest():
    r = c.gst_interest_s50(450000, "20-07-2025", "15-10-2025")
    assert r["days_delayed"] == 87 and r["interest"] == round(450000 * 0.18 * 87 / 365)


def test_period_add():
    assert c.period_calculator("31-01-2026", months=1)["result_date"] == "2026-02-28"


# ---------------- PII
def test_pii_mask_roundtrip():
    text = "PAN ABCDE1234F, GSTIN 27ABCDE1234F1Z5, mobile 9876543210, Aadhaar 2345 6789 0123"
    m, mp = mask(text)
    assert "ABCDE1234F" not in m and "9876543210" not in m and "2345 6789 0123" not in m
    assert unmask(m, mp) == text


def test_pii_reuses_existing_placeholders():
    m1, mp = mask("PAN ABCDE1234F")
    m2, mp2 = mask("Again ABCDE1234F and PQRSX9876Z", mp)
    assert "[PAN_1]" in m2 and "[PAN_2]" in m2 and mp2["[PAN_1]"] == "ABCDE1234F"


# ---------------- graph end-to-end in demo mode
def _cfg(tid):
    return {"configurable": {"thread_id": tid, "llm": get_llm(LLMConfig(provider="mock")),
                             "kb": KnowledgeBase(), "officer": {"name": "Test Officer"}}}


def test_graph_interrupts_then_approves():
    g = build_graph()
    cfg = _cfg("t1")
    out = g.invoke({"request": "Compare regime tax for salary of 18 lakh, PAN ABCDE1234F"}, cfg)
    assert "__interrupt__" in out
    st = g.get_state(cfg).values
    assert st["domain"] == "income_tax"
    assert st["tool_results"][0]["calculator"] == "compare_regimes"
    assert "ABCDE1234F" not in st["masked_request"]
    out = g.invoke(Command(resume={"decision": "approve"}), cfg)
    assert out["decision"] == "approve" and out["final"]


def test_graph_revise_loop_and_reject():
    g = build_graph()
    cfg = _cfg("t2")
    g.invoke({"request": "Draft RTI reply for third party ITR request"}, cfg)
    out = g.invoke(Command(resume={"decision": "revise", "feedback": "Cite section 8(1)(j)"}), cfg)
    assert "__interrupt__" in out  # back for review after revision
    out = g.invoke(Command(resume={"decision": "reject"}), cfg)
    assert out["final"] == "" and out["decision"] == "reject"


def test_kb_search_finds_rti():
    hits = KnowledgeBase().search("time limit to reply RTI application first appeal", domain="office_admin")
    assert hits and "rti" in hits[0]["source"]
