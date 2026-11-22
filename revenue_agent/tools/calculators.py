"""Deterministic calculators. The LLM never does arithmetic: it only picks a
calculator and fills its arguments; these functions produce the numbers.

Rates come from config/tax_rules.json so an office can update them without
touching code. Section references are to the Income-tax Act, 1961 unless
stated; the Income-tax Act, 2025 has corresponding (renumbered) provisions.
"""
from __future__ import annotations

import calendar
import json
import math
from datetime import date, timedelta
from pathlib import Path

RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "tax_rules.json"


def load_rules(path: Path = RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _year_rules(fy: str, rules: dict | None = None) -> dict:
    rules = rules or load_rules()
    it = rules["income_tax"]
    if fy not in it:
        raise ValueError(f"No rate table for FY {fy}. Available: {', '.join(k for k in it if not k.startswith('_'))}")
    yr = it[fy]
    seen = set()
    while "same_as" in yr:
        if fy in seen:
            raise ValueError("Circular 'same_as' in tax_rules.json")
        seen.add(fy)
        law = yr.get("law")
        fy = yr["same_as"]
        yr = dict(it[fy])
        if law:
            yr["law"] = law
    return yr


def _slab_tax(income: float, slabs: list, start: float = 0.0) -> tuple[float, list]:
    """slabs: [[upper_limit_or_None, rate], ...] applied from `start`."""
    tax, lower, rows = 0.0, start, []
    for upper, rate in slabs:
        if income <= lower:
            break
        top = income if upper is None else min(income, upper)
        part = max(0.0, top - lower)
        t = part * rate
        rows.append({"from": lower, "to": top, "rate": rate, "tax": round(t, 2)})
        tax += t
        if upper is None:
            break
        lower = upper
    return tax, rows


def _surcharge(income: float, tax: float, bands: list, tax_fn) -> tuple[float, float]:
    """Return (surcharge, marginal_relief). tax_fn(x) gives basic tax at income x."""
    rate, threshold, prev_rate = 0.0, None, 0.0
    for i, (limit, r) in enumerate(bands):
        if income > limit:
            prev_rate = rate
            rate, threshold = r, limit
    if rate == 0.0:
        return 0.0, 0.0
    sc = tax * rate
    # Marginal relief: tax + surcharge may not exceed tax(+surcharge) at the
    # threshold plus the income above the threshold.
    at_threshold = tax_fn(threshold) * (1 + prev_rate)
    cap = at_threshold + (income - threshold)
    relief = max(0.0, (tax + sc) - cap)
    return sc - relief, relief


def compute_income_tax(
    gross_income: float,
    fy: str = "2025-26",
    regime: str = "new",
    is_salaried: bool = True,
    age_band: str = "below_60",
    deductions: float = 0.0,
) -> dict:
    """Tax on total income for a resident individual (normal-rate income only).

    deductions: Chapter VI-A etc. — allowed only in the old regime here.
    Special-rate income (capital gains u/s 111A/112/112A etc.) is not covered.
    """
    gross_income = float(gross_income)
    yr = _year_rules(fy)
    cfg = yr["new_regime"] if regime == "new" else yr["old_regime"]
    sd = cfg["standard_deduction_salary"] if is_salaried else 0
    ded = float(deductions) if regime == "old" else 0.0
    total_income = max(0.0, gross_income - sd - ded)
    total_income = math.floor(total_income / 10) * 10  # round to nearest 10 below (s.288A)

    if regime == "new":
        tax_fn = lambda x: _slab_tax(x, cfg["slabs"])[0]
        basic, rows = _slab_tax(total_income, cfg["slabs"])
    else:
        exemption = cfg["basic_exemption"][age_band]
        slabs = [[u, r] for u, r in cfg["slabs_above_exemption"] if u is None or u > exemption]
        tax_fn = lambda x: _slab_tax(x, slabs, start=exemption)[0]
        basic, rows = _slab_tax(total_income, slabs, start=exemption)

    rebate = 0.0
    limit = cfg["rebate_income_limit"]
    if total_income <= limit:
        rebate = min(basic, cfg["rebate_max"])
    elif regime == "new" and cfg.get("rebate_marginal_relief"):
        excess = total_income - limit
        if basic > excess:
            rebate = basic - excess
    after_rebate = basic - rebate

    surcharge, sc_relief = _surcharge(total_income, after_rebate, cfg["surcharge"], tax_fn)
    cess = (after_rebate + surcharge) * yr["cess"]
    total = after_rebate + surcharge + cess
    return {
        "calculator": "income_tax",
        "law": yr.get("law"),
        "fy": fy,
        "regime": regime,
        "gross_income": gross_income,
        "standard_deduction": sd,
        "other_deductions": ded,
        "total_income": total_income,
        "slab_breakup": rows,
        "tax_on_total_income": round(basic, 2),
        "rebate": round(rebate, 2),
        "surcharge": round(surcharge, 2),
        "surcharge_marginal_relief": round(sc_relief, 2),
        "health_education_cess": round(cess, 2),
        "total_tax_payable": round(total / 10) * 10,  # s.288B rounding
        "note": "Normal-rate income of a resident individual only. Verify rates in config/tax_rules.json.",
    }


def compare_regimes(gross_income: float, fy: str = "2025-26", is_salaried: bool = True,
                    age_band: str = "below_60", deductions: float = 0.0) -> dict:
    new = compute_income_tax(gross_income, fy, "new", is_salaried, age_band, 0)
    old = compute_income_tax(gross_income, fy, "old", is_salaried, age_band, deductions)
    better = "new" if new["total_tax_payable"] <= old["total_tax_payable"] else "old"
    return {"calculator": "compare_regimes", "fy": fy, "new_regime": new, "old_regime": old,
            "lower_tax_regime": better,
            "saving": abs(new["total_tax_payable"] - old["total_tax_payable"])}


# ---------------------------------------------------------------- dates

def parse_date(d) -> date:
    if isinstance(d, date):
        return d
    s = str(d).strip()
    for sep in ("-", "/", "."):
        parts = s.split(sep)
        if len(parts) == 3:
            if len(parts[0]) == 4:  # YYYY-MM-DD
                return date(int(parts[0]), int(parts[1]), int(parts[2]))
            return date(int(parts[2]), int(parts[1]), int(parts[0]))  # DD-MM-YYYY (Indian)
    raise ValueError(f"Unrecognised date '{d}'. Use DD-MM-YYYY or YYYY-MM-DD.")


def add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def months_or_part(start_exclusive: date, end_inclusive: date) -> int:
    """Number of months or part of a month after `start_exclusive` up to
    `end_inclusive` (each part of a month counts as a full month)."""
    if end_inclusive <= start_exclusive:
        return 0
    months = (end_inclusive.year - start_exclusive.year) * 12 + (end_inclusive.month - start_exclusive.month)
    if add_months(start_exclusive, months) < end_inclusive:
        months += 1
    return months


def period_calculator(start, days: int = 0, months: int = 0, years: int = 0, end=None) -> dict:
    """Add a period to a date, or measure the gap between two dates."""
    s = parse_date(start)
    if end:
        e = parse_date(end)
        return {"calculator": "period", "start": s.isoformat(), "end": e.isoformat(),
                "days_between": (e - s).days,
                "months_or_part": months_or_part(s, e),
                "note": "days_between excludes the start date."}
    r = add_months(s, months + 12 * years) + timedelta(days=days)
    return {"calculator": "period", "start": s.isoformat(),
            "added": {"years": years, "months": months, "days": days},
            "result_date": r.isoformat(), "weekday": r.strftime("%A"),
            "note": "If the result falls on a holiday, check the applicable rule on extension (e.g. General Clauses Act, s.10)."}


# ---------------------------------------------------------------- interest

def _round_down_100(x: float) -> float:
    return math.floor(max(0.0, x) / 100) * 100  # Rule 119A


def interest_234a(tax_due: float, due_date, filing_date) -> dict:
    """Late filing of return: 1% per month or part on tax due after
    advance tax/TDS/TCS, from the day after the due date to the filing date."""
    rate = load_rules()["interest"]["income_tax_monthly_rate"]
    base = _round_down_100(tax_due)
    m = months_or_part(parse_date(due_date), parse_date(filing_date))
    return {"calculator": "interest_234A", "section": "234A, Income-tax Act 1961 (corresponding provision in 2025 Act)",
            "principal_rounded": base, "months": m, "rate_per_month": rate,
            "interest": round(base * rate * m)}


def interest_234b(assessed_tax: float, advance_tax_paid: float, fy: str, upto_date) -> dict:
    """Default in advance tax: applies if advance tax < 90% of assessed tax.
    1% per month or part from 1 April of the next year to `upto_date`."""
    rate = load_rules()["interest"]["income_tax_monthly_rate"]
    start_year = int(fy.split("-")[0])
    fy_end = date(start_year + 1, 3, 31)
    applicable = advance_tax_paid < 0.9 * assessed_tax
    base = _round_down_100(assessed_tax - advance_tax_paid) if applicable else 0
    m = months_or_part(fy_end, parse_date(upto_date)) if applicable else 0
    return {"calculator": "interest_234B", "section": "234B, Income-tax Act 1961 (corresponding provision in 2025 Act)",
            "applicable": applicable, "shortfall_rounded": base, "months": m,
            "rate_per_month": rate, "interest": round(base * rate * m)}


def interest_234c(tax_on_returned_income: float, paid_by_jun15: float, paid_by_sep15: float,
                  paid_by_dec15: float, paid_by_mar15: float) -> dict:
    """Deferment of advance-tax instalments (non-presumptive assessee).
    Paid amounts are CUMULATIVE totals by each date."""
    cfg = load_rules()["interest"]
    rate = cfg["income_tax_monthly_rate"]
    paid = [paid_by_jun15, paid_by_sep15, paid_by_dec15, paid_by_mar15]
    rows, total = [], 0
    for inst, p in zip(cfg["advance_tax_schedule"], paid):
        required = tax_on_returned_income * inst["cumulative_pct"]
        safe = inst["safe_harbour_pct"]
        if safe is not None and p >= tax_on_returned_income * safe:
            shortfall = 0
        else:
            shortfall = _round_down_100(required - p)
        i = round(shortfall * rate * inst["months"])
        total += i
        rows.append({"due": inst["due"], "required": round(required), "paid": p,
                     "shortfall": shortfall, "months": inst["months"], "interest": i})
    return {"calculator": "interest_234C", "section": "234C, Income-tax Act 1961 (corresponding provision in 2025 Act)",
            "instalments": rows, "interest": total}


def gst_interest_s50(tax_amount: float, due_date, payment_date, wrong_itc: bool = False) -> dict:
    """Interest u/s 50 CGST Act on delayed payment, simple interest per day."""
    cfg = load_rules()["interest"]
    rate = cfg["gst_s50_wrong_itc_annual_rate" if wrong_itc else "gst_s50_annual_rate"]
    days = max(0, (parse_date(payment_date) - parse_date(due_date)).days)
    interest = float(tax_amount) * rate * days / 365
    return {"calculator": "gst_interest_s50", "section": "50, CGST Act 2017 (and corresponding SGST/IGST)",
            "tax_amount": tax_amount, "days_delayed": days, "annual_rate": rate,
            "interest": round(interest)}


CALCULATORS = {
    "income_tax": compute_income_tax,
    "compare_regimes": compare_regimes,
    "period": period_calculator,
    "interest_234a": interest_234a,
    "interest_234b": interest_234b,
    "interest_234c": interest_234c,
    "gst_interest_s50": gst_interest_s50,
}

CALCULATOR_SPECS = """
income_tax(gross_income, fy="2025-26"|"2026-27", regime="new"|"old", is_salaried=true, age_band="below_60"|"60_to_80"|"above_80", deductions=0)
compare_regimes(gross_income, fy, is_salaried, age_band, deductions)
period(start="DD-MM-YYYY", days=0, months=0, years=0, end=null)   # add a period OR give end to measure gap
interest_234a(tax_due, due_date, filing_date)
interest_234b(assessed_tax, advance_tax_paid, fy, upto_date)
interest_234c(tax_on_returned_income, paid_by_jun15, paid_by_sep15, paid_by_dec15, paid_by_mar15)
gst_interest_s50(tax_amount, due_date, payment_date, wrong_itc=false)
""".strip()


def run_calculator(name: str, args: dict) -> dict:
    fn = CALCULATORS.get(name)
    if fn is None:
        return {"error": f"Unknown calculator '{name}'"}
    try:
        return fn(**args)
    except Exception as e:  # report, don't crash the agent
        return {"calculator": name, "error": f"{type(e).__name__}: {e}", "args": args}
