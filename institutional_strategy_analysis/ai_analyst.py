# -*- coding: utf-8 -*-
"""
institutional_strategy_analysis/ai_analyst.py
──────────────────────────────────────────────
Builds rich analytical prompts and calls Claude for deep institutional
investment strategy analysis at CIO / manager-selection level.

Public API
──────────
    run_ai_analysis(display_df, context)                          -> AnalysisResult
    run_focused_analysis(full_df, manager, track, peers, context) -> AnalysisResult
    run_comparison_analysis(df, mgr_a, trk_a, mgr_b, trk_b, ctx) -> AnalysisResult
    compute_manager_scorecard(full_df, manager, track)            -> list[dict]
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ── API key resolution ────────────────────────────────────────────────────────

def _get_api_key() -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "").strip()


def _get_model() -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_MODEL" in st.secrets:
            return str(st.secrets["OPENAI_MODEL"]).strip()
    except Exception:
        pass
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


def _call_openai(prompt: str, system: str = "", max_tokens: int = 1400) -> tuple[str, Optional[str]]:
    api_key = _get_api_key()
    if not api_key:
        return "", "לא הוגדר OPENAI_API_KEY ב-Settings → Secrets."

    payload = {
        "model": _get_model(),
        "messages": [
            {"role": "system", "content": system or "אתה אנליסט השקעות."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.5,
        "max_completion_tokens": max_tokens,
    }

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        if resp.status_code == 200:
            data = resp.json()
            text = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content", "")
            return str(text).strip(), None if text else "תגובה ריקה מהמודל."
        if resp.status_code == 401:
            return "", "מפתח OpenAI לא תקין."
        if resp.status_code == 429:
            return "", "חריגה ממגבלת קצב הבקשות או מהמכסה. נסה שוב מאוחר יותר."
        try:
            err = resp.json()
        except Exception:
            err = {}
        msg = err.get("error", {}).get("message") if isinstance(err, dict) else None
        return "", msg or f"שגיאת API: HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return "", "תם הזמן הקצוב (90 שניות). נסה שוב."
    except Exception as e:
        return "", f"שגיאת תקשורת: {e}"


# ── Statistical computation ───────────────────────────────────────────────────

def _compute_rich_stats(df: pd.DataFrame, alloc: str, manager: str, track: str) -> dict:
    sub = df[
        (df["allocation_name"] == alloc) &
        (df["manager"] == manager) &
        (df["track"] == track)
    ].sort_values("date")

    if sub.empty or len(sub) < 2:
        return {}

    vals = sub["allocation_value"].dropna()
    if len(vals) < 2:
        return {}

    if "frequency" in sub.columns:
        m_sub = sub[sub["frequency"] == "monthly"]
        monthly_vals = m_sub["allocation_value"].dropna() if not m_sub.empty else vals
    else:
        monthly_vals = vals

    diffs = monthly_vals.diff().dropna()

    slope = 0.0
    if len(monthly_vals) >= 3:
        x = np.arange(len(monthly_vals))
        slope = float(np.polyfit(x, monthly_vals.values, 1)[0])

    reversals = 0
    if len(diffs) >= 2:
        signs = np.sign(diffs.values)
        reversals = int(np.sum(np.diff(signs) != 0))

    dynamism = float(diffs.abs().mean()) if not diffs.empty else 0.0

    max_date = sub["date"].max()

    yr_ago_df  = sub[sub["date"] <= max_date - pd.DateOffset(months=12)]
    yr_ago_val = float(yr_ago_df["allocation_value"].iloc[-1]) if not yr_ago_df.empty else float("nan")
    change_12m = round(float(vals.iloc[-1]) - yr_ago_val, 2) if not np.isnan(yr_ago_val) else float("nan")

    mo3_ago_df = sub[sub["date"] <= max_date - pd.DateOffset(months=3)]
    mo3_val    = float(mo3_ago_df["allocation_value"].iloc[-1]) if not mo3_ago_df.empty else float("nan")
    change_3m  = round(float(vals.iloc[-1]) - mo3_val, 2) if not np.isnan(mo3_val) else float("nan")

    recent_direction = "—"
    if len(diffs) >= 3:
        last3 = diffs.iloc[-3:].mean()
        if last3 > 0.3:    recent_direction = "עולה"
        elif last3 < -0.3: recent_direction = "יורדת"
        else:              recent_direction = "יציבה"

    return {
        "current":          round(float(vals.iloc[-1]), 2),
        "mean":             round(float(vals.mean()), 2),
        "min":              round(float(vals.min()), 2),
        "max":              round(float(vals.max()), 2),
        "std":              round(float(vals.std()), 2),
        "range_pp":         round(float(vals.max() - vals.min()), 2),
        "slope_monthly":    round(slope, 3),
        "dynamism":         round(dynamism, 3),
        "reversals":        reversals,
        "change_12m":       change_12m,
        "change_3m":        change_3m,
        "recent_direction": recent_direction,
        "mom_avg":          round(float(diffs.mean()), 3) if not diffs.empty else 0,
        "mom_max":          round(float(diffs.abs().max()), 3) if not diffs.empty else 0,
        "n_monthly":        int((sub["frequency"] == "monthly").sum()) if "frequency" in sub.columns else 0,
        "n_yearly":         int((sub["frequency"] == "yearly").sum()) if "frequency" in sub.columns else 0,
        "date_first":       sub["date"].min().strftime("%Y-%m"),
        "date_last":        sub["date"].max().strftime("%Y-%m"),
    }


def _compute_manager_profile(df: pd.DataFrame, manager: str, track: str) -> dict:
    sub = df[(df["manager"] == manager) & (df["track"] == track)]
    if sub.empty:
        return {}

    allocs    = sub["allocation_name"].unique()
    per_alloc = {a: _compute_rich_stats(df, a, manager, track) for a in allocs}
    per_alloc = {k: v for k, v in per_alloc.items() if v}

    fx_key  = next((k for k in allocs if any(x in k for x in ['מט"ח', 'מטח', 'fx', 'FX', 'currency'])), None)
    fgn_key = next((k for k in allocs if any(x in k for x in ['חו"ל', 'חול', 'foreign', 'Foreign'])), None)
    eq_key  = next((k for k in allocs if any(x in k for x in ['מניות', 'מנייתי', 'equity', 'Equity'])), None)

    hedging_ratio = None
    if fx_key and fgn_key:
        fx_now  = per_alloc.get(fx_key, {}).get("current")
        fgn_now = per_alloc.get(fgn_key, {}).get("current")
        if fx_now is not None and fgn_now and fgn_now > 0:
            hedging_ratio = round(fx_now / fgn_now * 100, 1)

    dyn_vals         = [v["dynamism"] for v in per_alloc.values() if "dynamism" in v]
    overall_dynamism = round(float(np.mean(dyn_vals)), 3) if dyn_vals else 0.0
    total_reversals  = sum(v.get("reversals", 0) for v in per_alloc.values())

    risk_trend = None
    if eq_key and eq_key in per_alloc:
        s = per_alloc[eq_key].get("slope_monthly", 0)
        if s > 0.3:    risk_trend = "מגדיל סיכון (מניות עולות)"
        elif s < -0.3: risk_trend = "מקטין סיכון (מניות יורדות)"
        else:          risk_trend = "יציב"

    return {
        "per_alloc":        per_alloc,
        "hedging_ratio":    hedging_ratio,
        "overall_dynamism": overall_dynamism,
        "total_reversals":  total_reversals,
        "risk_trend":       risk_trend,
        "fx_key":           fx_key,
        "fgn_key":          fgn_key,
        "eq_key":           eq_key,
    }


def _cross_manager_snapshot(df: pd.DataFrame, alloc: str) -> str:
    sub = df[df["allocation_name"] == alloc].copy()
    if sub.empty:
        return "  (אין נתונים)"
    idx  = sub.groupby(["manager", "track"])["date"].idxmax()
    snap = sub.loc[idx].sort_values("allocation_value", ascending=False)
    return "\n".join(
        f"  {row['manager']} [{row['track']}]: {row['allocation_value']:.1f}%"
        for _, row in snap.iterrows()
    )


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """
אתה כותב סקירה ללקוח בסגנון פמילי אופיס, בעברית טבעית, זורמת וברורה.
המטרה היא להסביר איך מנהל ההשקעות מתנהל בפועל, מה בולט בו ביחס לאחרים וביחס לעצמו לאורך זמן, ולמי הסגנון הזה עשוי להתאים.
הכתיבה צריכה להיות סיפורית ורציפה, לא רשימת מכולת, לא מסמך אקדמי, ולא מלאה בז'רגון.
מותר להשתמש במספרים מתוך הנתונים כשזה מועיל, אבל צריך לשלב אותם בשפה טבעית.
אל תשתמש בכותרות משנה עם ##, אל תחלק לתיבות, ואל תכתוב סעיפים.
כתוב 2–4 פסקאות רציפות, כאילו יועץ השקעות או פמילי אופיס מסביר ללקוח מה הוא רואה.
הבחנות מקצועיות חשובות: יחס מט"ח מול חו"ל הוא רק אינדיקציה חלקית לגידור, לא מדד ודאי;
סטיית תקן של שינויי אלוקציה מעידה על תזזיתיות ניהולית ולא על סיכון תיק כולל; דינמיות גבוהה אינה בהכרח יתרון או חסרון.
בסוף הסקירה תכלול גם למי זה בעיקר מתאים, ובאילו מצבי עולם הגוף עשוי לבלוט לטובה או פחות לטובה.
אל תמציא נתונים, ואל תציג מסקנה נחרצת כשהמידע רק מרמז.
"""


def _compact_manager_block(df: pd.DataFrame, manager: str, track: str) -> str:
    profile = _compute_manager_profile(df, manager, track)
    if not profile or not profile.get("per_alloc"):
        return ""
    lines = [f"גוף: {manager} | מסלול: {track}"]
    lines.append(f"דינמיות כוללת: {profile['overall_dynamism']:.3f} | שינויי כיוון: {profile['total_reversals']} | מגמת סיכון: {profile.get('risk_trend','—')}")
    if profile.get("hedging_ratio") is not None:
        lines.append(f'יחס מט"ח/חו"ל: {profile["hedging_ratio"]:.1f}%')
    for alloc, s in profile["per_alloc"].items():
        c3 = s.get("change_3m", float("nan"))
        c12 = s.get("change_12m", float("nan"))
        c3s = f"{c3:+.1f}pp" if not np.isnan(c3) else "—"
        c12s = f"{c12:+.1f}pp" if not np.isnan(c12) else "—"
        lines.append(
            f"{alloc}: נוכחי {s['current']}%, ממוצע היסטורי {s['mean']}%, טווח {s['min']}%-{s['max']}%, "
            f"סטיית תקן {s['std']}pp, שינוי 3ח {c3s}, שינוי 12ח {c12s}, מגמה {s['recent_direction']}, דינמיות {s['dynamism']:.3f}"
        )
    return "\n".join(lines)


# ── Prompt: Market-wide analysis (all selected managers) ─────────────────────

def _build_full_prompt(display_df: pd.DataFrame, context: dict, user_question: str = "") -> str:
    managers = context.get("managers", [])
    tracks = context.get("tracks", [])
    allocation_names = context.get("allocation_names", [])
    sel_range = context.get("selected_range", "הכל")

    blocks = []
    for mgr in managers:
        for trk in tracks:
            block = _compact_manager_block(display_df, mgr, trk)
            if block:
                blocks.append(block)

    cross_lines = []
    for alloc in allocation_names:
        snap = _cross_manager_snapshot(display_df, alloc)
        if snap and snap != "  (אין נתונים)":
            cross_lines.append(f"{alloc}:\n{snap}")

    question = user_question.strip() or "כתוב שיחה חופשית שמסבירה מה בולט בנתונים, מי נראה דינמי יותר, מי שמרני יותר, ואיך נכון להסביר ללקוח את סגנון הניהול של הגופים."
    return f"""טווח נתונים: {sel_range}

נתוני בסיס:
{chr(10).join(blocks) or '(אין נתונים)'}

השוואת snapshot בין גופים:
{chr(10).join(cross_lines) or '(אין נתונים)'}

הנחיה: {question}

כתוב תשובה אחת רציפה, כמו סקירת פמילי אופיס ללקוח. אפשר לפתוח למשל בסגנון: 'מה שבולט כאן הוא ש...', 'בתמונה הכוללת נראה ש...', 'ללקוח שמחפש...'.
הסבר בפשטות מה בולט, מה השתנה לאחרונה, מי יציב יותר ומי דינמי יותר, ולמי כל סגנון עשוי להתאים. שלב השוואה לאחרים כשזה רלוונטי. אל תחלק את התשובה לסעיפים, כותרות או תיבות."""


# ── Prompt: Focused single-manager analysis vs peer group ────────────────────

def _build_focused_prompt(
    full_df: pd.DataFrame,
    manager: str,
    track: str,
    peer_managers: list,
    context: dict,
) -> str:
    focus_profile = _compute_manager_profile(full_df, manager, track)
    if not focus_profile or not focus_profile["per_alloc"]:
        return ""

    all_in_track = full_df[full_df["track"] == track]["manager"].unique().tolist()
    peers = [m for m in (peer_managers or all_in_track) if m != manager and m in all_in_track]
    if not peers:
        return ""

    peer_blocks = []
    for pm in peers:
        block = _compact_manager_block(full_df, pm, track)
        if block:
            peer_blocks.append(block)

    focus_block = _compact_manager_block(full_df, manager, track)
    return f"""מנהל לניתוח: {manager} | מסלול: {track}
טווח נתונים: {context.get('date_min', '?')} – {context.get('date_max', '?')}

נתוני הגוף הנבחן:
{focus_block}

עמיתים להשוואה:
{chr(10).join(peer_blocks) or '(אין נתוני עמיתים)'}

כתוב סקירה אחת רציפה בעברית, בסגנון פמילי אופיס.
הסקירה צריכה להסביר בפשטות איך {manager} מנהל את המסלול הזה ביחס לאחרים וגם ביחס להיסטוריה של עצמו.
התמקד במה שבולט באמת: חשיפה יחסית לחו"ל, מניות, מט"ח ולא-סחיר; תזזיתיות מול יציבות; האם בחודשים האחרונים הוא מגדיל או מקטין סיכון; והאם נראה שיש שינוי כיוון.
תן דוגמאות ורעיונות לניסוח, למשל: 'המסלול הכללי של {manager} בולט ב...', 'בחודשים האחרונים אנחנו רואים ש...', 'המשמעות האפשרית היא...', 'ללקוח שמעדיף יציבות...', 'בתרחיש של שוק עולה/שקל נחלש/סביבת אי-ודאות...'.
הסבר גם למי זה בעיקר מתאים, ובאילו מצבי עולם הגוף הזה כנראה יבלוט לטובה או פחות לטובה.
אל תחלק לסעיפים, כותרות או רשימות. כתוב 2–4 פסקאות לכל היותר."""


# ── Prompt: Head-to-head comparison ──────────────────────────────────────────

def _build_comparison_prompt(
    display_df: pd.DataFrame,
    mgr_a: str, trk_a: str,
    mgr_b: str, trk_b: str,
) -> str:
    prof_a = _compact_manager_block(display_df, mgr_a, trk_a)
    prof_b = _compact_manager_block(display_df, mgr_b, trk_b)
    if not prof_a or not prof_b:
        return ""
    return f"""השוואה בין שני גופים במסלול הכללי/נבחר.

גוף א:
{prof_a}

גוף ב:
{prof_b}

כתוב סקירה אחת רציפה, בסגנון פמילי אופיס, שמשווה בין שני הגופים בשפה פשוטה ונעימה ללקוח.
המטרה היא להסביר מי נראה דינמי יותר, מי יציב יותר, מי נוטה יותר לחו"ל, למניות, ללא-סחיר או לחשיפת מט"ח, ומה זה אומר בפועל.
הסבר גם למי כל גוף עשוי להתאים יותר, ובאילו מצבי עולם כל אחד מהם עשוי לבלוט לטובה — למשל שוק מניות חזק, תקופה תנודתית, שקל נחלש, או סביבה שמעדיפה יציבות.
אפשר לפתוח למשל כך: 'בהשוואה בין שני הגופים רואים ש...', 'הגישה של {mgr_a} נראית...', 'לעומת זאת אצל {mgr_b}...'.
אל תחלק לכותרות, טבלאות או bullet points. כתוב 2–4 פסקאות רציפות."""


def _build_question_prompt(full_df: pd.DataFrame, manager: Optional[str], track: Optional[str], question: str) -> str:
    df = full_df.copy()
    if manager:
        df = df[df["manager"] == manager]
    if track:
        df = df[df["track"] == track]
    combos = df[["manager", "track"]].drop_duplicates().values.tolist() if not df.empty else []
    blocks = []
    for mgr, trk in combos[:12]:
        block = _compact_manager_block(df, mgr, trk)
        if block:
            blocks.append(block)
    return f"""שאלה חופשית על נתוני סגנון ניהול השקעות של גופים מוסדיים.

הנתונים הרלוונטיים:
{chr(10).join(blocks) or '(אין נתונים)'}

שאלת המשתמש: {question}

ענה בעברית טבעית, בסגנון פמילי אופיס. תענה ישירות לשאלה, תסביר מה באמת ניתן להסיק מהנתונים ומה רק מרומז, ותשתמש במספרים רק כשהם תורמים להבנה. אל תחלק לכותרות או תיבות."""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    raw_text: str = ""
    sections: dict = field(default_factory=dict)
    error: Optional[str] = None

    def parse_sections(self):
        if self.raw_text:
            self.sections = {"סקירה": self.raw_text.strip()}


# ── Public API ────────────────────────────────────────────────────────────────

def run_ai_analysis(display_df: pd.DataFrame, context: dict, user_question: str = "") -> AnalysisResult:
    if display_df.empty:
        return AnalysisResult(error="אין נתונים לניתוח.")
    prompt = _build_full_prompt(display_df, context, user_question=user_question)
    text, err = _call_openai(prompt, system=_SYSTEM, max_tokens=1400)
    result = AnalysisResult(raw_text=text, error=err)
    if text:
        result.parse_sections()
    return result


def run_focused_analysis(
    full_df: pd.DataFrame,
    manager: str,
    track: str,
    peer_managers,
    context: dict,
) -> AnalysisResult:
    if full_df.empty:
        return AnalysisResult(error="אין נתונים לניתוח.")
    prompt = _build_focused_prompt(full_df, manager, track, peer_managers, context)
    if not prompt:
        return AnalysisResult(error="לא נמצאו נתונים מספיקים לניתוח מיקוד.")
    text, err = _call_openai(prompt, system=_SYSTEM, max_tokens=1600)
    result = AnalysisResult(raw_text=text, error=err)
    if text:
        result.parse_sections()
    return result


def run_comparison_analysis(
    display_df: pd.DataFrame,
    mgr_a: str, trk_a: str,
    mgr_b: str, trk_b: str,
    context: dict,
) -> AnalysisResult:
    if display_df.empty:
        return AnalysisResult(error="אין נתונים לניתוח.")
    if mgr_a == mgr_b and trk_a == trk_b:
        return AnalysisResult(error="יש לבחור שני גופים/מסלולים שונים.")
    prompt = _build_comparison_prompt(display_df, mgr_a, trk_a, mgr_b, trk_b)
    if not prompt:
        return AnalysisResult(error="לא נמצאו נתונים לאחד מהגופים הנבחרים.")
    text, err = _call_openai(prompt, system=_SYSTEM, max_tokens=1400)
    result = AnalysisResult(raw_text=text, error=err)
    if text:
        result.parse_sections()
    return result




def run_question_analysis(
    full_df: pd.DataFrame,
    question: str,
    manager: Optional[str] = None,
    track: Optional[str] = None,
) -> AnalysisResult:
    if full_df.empty:
        return AnalysisResult(error="אין נתונים לניתוח.")
    if not question.strip():
        return AnalysisResult(error="יש לכתוב שאלה.")
    prompt = _build_question_prompt(full_df, manager, track, question)
    text, err = _call_openai(prompt, system=_SYSTEM, max_tokens=1400)
    result = AnalysisResult(raw_text=text, error=err)
    if text:
        result.parse_sections()
    return result

# ── Quick scorecard (no API call) ─────────────────────────────────────────────

def compute_manager_scorecard(full_df: pd.DataFrame, manager: str, track: str) -> list:
    """
    Returns per-allocation relative stats for the quick scorecard widget.
    No API call needed — pure statistics.
    """
    profile = _compute_manager_profile(full_df, manager, track)
    if not profile:
        return []

    all_managers = full_df[full_df["track"] == track]["manager"].unique().tolist()
    rows = []
    for alloc, fs in profile["per_alloc"].items():
        peer_vals = []
        for pm in all_managers:
            if pm == manager:
                continue
            ps = _compute_rich_stats(full_df, alloc, pm, track)
            if ps:
                peer_vals.append(ps["current"])

        if not peer_vals:
            continue

        peer_mean = round(float(np.mean(peer_vals)), 2)
        ranking   = sorted(
            [(manager, fs["current"])] + [(None, v) for v in peer_vals],
            key=lambda x: -x[1],
        )
        rank = next((i + 1 for i, (m, _) in enumerate(ranking) if m == manager), None)

        rows.append({
            "alloc":      alloc,
            "current":    fs["current"],
            "peer_mean":  peer_mean,
            "diff_mean":  round(fs["current"] - peer_mean, 2),
            "rank":       rank,
            "n_total":    len(ranking),
            "direction":  fs.get("recent_direction", "—"),
            "change_3m":  fs.get("change_3m", float("nan")),
            "change_12m": fs.get("change_12m", float("nan")),
            "dynamism":   fs.get("dynamism", 0),
        })
    return rows