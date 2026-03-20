# -*- coding: utf-8 -*-
"""
institutional_strategy_analysis/ui.py
───────────────────────────────────────
Self-contained Streamlit UI for "ניתוח אסטרטגיות מוסדיים".
Renders as an st.expander at the bottom of the main app.

Entry point (one line in streamlit_app.py):
    from institutional_strategy_analysis.ui import render_institutional_analysis
    render_institutional_analysis()

All session-state keys are prefixed "isa_" to avoid any collision.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

# ── Sheet URL ─────────────────────────────────────────────────────────────────
# ▼▼▼  Set your Google Sheets URL here  ▼▼▼
ISA_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1e9zjj1OWMYqUYoK6YFYvYwOnN7qbydYDyArHbn8l9pE/edit"
)
# ▲▲▲─────────────────────────────────────────────────────────────────────────

# ── Lazy imports (never execute at import time) ───────────────────────────────

def _load_data():
    from institutional_strategy_analysis.loader     import load_raw_blocks
    from institutional_strategy_analysis.series_builder import get_time_bounds
    import streamlit as st

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached(url: str):
        return load_raw_blocks(url)

    return _cached(ISA_SHEET_URL)


def _build_series(df_y, df_m, rng, custom_start, filters):
    from institutional_strategy_analysis.series_builder import build_display_series
    return build_display_series(df_y, df_m, rng, custom_start, filters)


def _options(df_y, df_m):
    from institutional_strategy_analysis.series_builder import get_available_options
    return get_available_options(df_y, df_m)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_plotly(fig, key=None):
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except TypeError:
        st.plotly_chart(fig)


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def _clamp(val: date, lo: date, hi: date) -> date:
    return max(lo, min(hi, val))


# ── Debug panel ───────────────────────────────────────────────────────────────

def _render_debug(df_yearly, df_monthly, debug_info, errors):
    with st.expander("🛠️ מידע אבחון (debug)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("גליונות שנטענו", len(debug_info))
            st.metric("שורות שנתי", len(df_yearly))
            st.metric("שורות חודשי", len(df_monthly))
        with col2:
            if not df_yearly.empty:
                yr = df_yearly["date"]
                st.metric("טווח שנתי", f"{yr.min().year} – {yr.max().year}")
            if not df_monthly.empty:
                mr = df_monthly["date"]
                st.metric("טווח חודשי",
                          f"{mr.min().strftime('%Y-%m')} – {mr.max().strftime('%Y-%m')}")

        if debug_info:
            rows = []
            for d in debug_info:
                rows.append({
                    "גליון": d.get("sheet", "?"),
                    "header row": d.get("header_row", "?"),
                    "freq col": d.get("freq_col", "—"),
                    "שורות שנתיות": d.get("yearly_rows", 0),
                    "שורות חודשיות": d.get("monthly_rows", 0),
                    "טווח שנתי": d.get("yearly_range", "—"),
                    "טווח חודשי": d.get("monthly_range", "—"),
                    "שגיאה": d.get("error", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if errors:
            for e in errors:
                st.warning(e)



# ── AI Analysis renderer ──────────────────────────────────────────────────────

_SECTION_ICONS = {
    # Market analysis
    "מיצוב יחסי לפי רכיב":                           "🎯",
    "דינמיות ואפיון סגנון ניהול":                    "⚡",
    'אסטרטגיית גידור מט"ח':                          "🛡️",
    "תנועות פוזיציה אחרונות (3–12 חודשים)":          "🔄",
    "ניתוח סיכון קבוצתי":                            "⚠️",
    "יתרונות וחסרונות לפי גוף":                      "✅",
    "תובנה אסטרטגית וסיכום":                         "💡",
    # Focused analysis
    "מיצוב יחסי (Relative Positioning)":             "🎯",
    "ניתוח היסטורי עצמי (Historical Self-Analysis)": "📜",
    "סגנון ניהול ועקביות (Management Style & Consistency)": "⚡",
    "אסטרטגיית גידור ומטבע (Hedging & Currency Strategy)": "🛡️",
    "מומנטום ותנועות אחרונות (Recent Momentum)":     "🔄",
    "פרופיל סיכון (Risk Assessment)":                "⚠️",
    "גזר דין — בחירת מנהל (Manager Selection Verdict)": "👑",
    # Comparison
    "סיכום מנהלי (Executive Summary)":               "📌",
    "השוואה יחסית לפי רכיב":                        "⚖️",
    "הבדלי סגנון ניהול (Management Style Delta)":    "⚡",
    "אסטרטגיית גידור (Hedging Comparison)":         "🛡️",
    "המלצה לפי פרופיל משקיע":                       "👤",
    # Legacy fallbacks
    "ניתוח לפי גוף ומסלול":    "🏢",
    "ניתוח סיכון":              "⚠️",
    "תובנה אסטרטגית":          "💡",
    "סיכום מנהלי":             "📌",
}

_EXPANDED_SECTIONS = {
    "מיצוב יחסי",
    "סיכום מנהלי",
    "גזר דין",
    "תובנה אסטרטגית",
}


def _render_api_key_input():
    """Check OpenAI key from secrets/env only. Returns True if set."""
    try:
        import os
        key = ""
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            key = str(st.secrets["OPENAI_API_KEY"]).strip()
        key = key or os.getenv("OPENAI_API_KEY", "").strip()
        if key:
            st.success("✅ חיבור OpenAI פעיל דרך OPENAI_API_KEY")
            return True
    except Exception:
        pass
    st.error("🔑 להפעלת הניתוח יש להגדיר OPENAI_API_KEY ב-Settings → Secrets")
    return False


def _render_analysis_result(result, cache_key: str, dl_key: str, refresh_key: str):
    if result.error:
        st.error(f"⚠️ {result.error}")
        if st.button("נסה שוב", key=f"{refresh_key}_retry_{cache_key}"):
            st.session_state.pop(cache_key, None)
            st.session_state.pop(f"{cache_key}_sig", None)
            st.rerun()
        return

    narrative = (result.raw_text or "").strip()
    if not narrative:
        st.info("לא התקבל ניתוח.")
        return

    st.markdown(f"""
<div style='background:#fcfcfd;border:1px solid #e2e8f0;border-radius:14px;
     padding:20px 22px;margin:10px 0 14px 0;direction:rtl;line-height:1.95;
     font-size:18px;color:#0f172a;box-shadow:0 1px 3px rgba(15,23,42,.05)'>
  {narrative.replace(chr(10), '<br><br>')}
</div>""", unsafe_allow_html=True)

    col_dl, col_rf, _ = st.columns([1, 1, 4])
    with col_dl:
        st.download_button("⬇️ שמור ניתוח", data=narrative.encode("utf-8"), file_name=f"{dl_key}.txt", mime="text/plain", key=f"isa_dl_{dl_key}_{cache_key}", use_container_width=True)
    with col_rf:
        if st.button("🔄 רענן", key=f"{refresh_key}_{cache_key}", help="הרץ מחדש את הניתוח", use_container_width=True):
            st.session_state.pop(cache_key, None)
            st.session_state.pop(f"{cache_key}_sig", None)
            st.rerun()


def _scorecard_badge(diff: float) -> str:
    """Return an HTML badge for relative positioning."""
    if diff > 3:   return "<span style='background:#16a34a;color:#fff;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700'>▲ גבוה משמעותית</span>"
    if diff > 1:   return "<span style='background:#4ade80;color:#14532d;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700'>▲ מעל ממוצע</span>"
    if diff < -3:  return "<span style='background:#dc2626;color:#fff;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700'>▼ נמוך משמעותית</span>"
    if diff < -1:  return "<span style='background:#f87171;color:#7f1d1d;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700'>▼ מתחת לממוצע</span>"
    return "<span style='background:#e2e8f0;color:#475569;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700'>◼ ממוצע</span>"


def _direction_badge(direction: str) -> str:
    if direction == "עולה":   return "🟢 עולה"
    if direction == "יורדת":  return "🔴 יורדת"
    return "⚪ יציבה"


def _render_quick_scorecard(full_df: pd.DataFrame, manager: str, track: str):
    """Render a quick stat-based scorecard card before running AI."""
    import numpy as np
    try:
        from institutional_strategy_analysis.ai_analyst import compute_manager_scorecard
        rows = compute_manager_scorecard(full_df, manager, track)
    except Exception:
        return

    if not rows:
        return

    st.markdown("""
<div style='background:#f0f4ff;border:1px solid #c7d7fe;border-radius:10px;
     padding:14px 18px;margin:10px 0 4px 0;direction:rtl'>
  <div style='font-size:13px;font-weight:700;color:#1e3a8a;margin-bottom:10px'>
    📊 סקירה מהירה — מיצוב יחסי לפי רכיב
  </div>""", unsafe_allow_html=True)

    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        diff = row["diff_mean"]
        c3   = row.get("change_3m", float("nan"))
        c12  = row.get("change_12m", float("nan"))
        import math
        c3s  = f"{c3:+.1f}pp" if not math.isnan(c3)  else "—"
        c12s = f"{c12:+.1f}pp" if not math.isnan(c12) else "—"
        with col:
            st.markdown(f"""
<div style='background:#fff;border:1px solid #e2e8f0;border-radius:8px;
     padding:10px;text-align:center;direction:rtl'>
  <div style='font-size:11px;color:#64748b;margin-bottom:4px'>{row['alloc']}</div>
  <div style='font-size:22px;font-weight:900;color:#1e3a8a'>{row['current']}%</div>
  <div style='font-size:11px;color:#64748b'>ממוצע קבוצה: {row['peer_mean']}%</div>
  <div style='font-size:11px;margin:4px 0'>{_scorecard_badge(diff)}</div>
  <div style='font-size:10px;color:#94a3b8;margin-top:4px'>
    3ח: {c3s} | 12ח: {c12s}<br/>
    דירוג: {row['rank']}/{row['n_total']} | {_direction_badge(row['direction'])}
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_ai_section(
    display_df: pd.DataFrame,
    full_df: pd.DataFrame,
    context: dict,
    sel_mgr: list,
    sel_tracks: list,
):
    st.markdown("---")
    st.markdown("""
<div style='background:linear-gradient(135deg,#0f2657 0%,#1d4ed8 60%,#2563eb 100%);
     border-radius:14px;padding:18px 24px;margin-bottom:18px;direction:rtl'>
  <div style='color:#fff;font-size:20px;font-weight:900;letter-spacing:-0.5px'>🤖 ניתוח AI — סגנון ניהול השקעות</div>
  <div style='color:#dbeafe;font-size:13px;margin-top:6px;line-height:1.8'>
    סקירה סיפורית בסגנון פמילי אופיס: איך הגוף מנהל השקעות בפועל, למי זה מתאים,
    ובאילו מצבי עולם הוא עשוי לבלוט לטובה ביחס לאחרים.
  </div>
</div>""", unsafe_allow_html=True)

    if not _render_api_key_input():
        return

    mode_labels = {
        "focused": "1. סקירה כללית על סגנון ניהול ההשקעות של גוף אחד",
        "compare": "2. השוואה בין שני גופים",
        "free": "3. שיחה חופשית לשאלות לגבי הנתונים",
    }
    mode = st.radio("סוג ניתוח", options=list(mode_labels.keys()), format_func=lambda k: mode_labels[k], key="isa_ai_mode")

    all_mgrs = sorted(full_df["manager"].unique().tolist()) if not full_df.empty else sorted(sel_mgr)
    all_tracks = sorted(full_df["track"].unique().tolist()) if not full_df.empty else list(sel_tracks)
    default_track = "כללי" if "כללי" in all_tracks else (all_tracks[0] if all_tracks else "")

    if mode == "focused":
        c1, c2 = st.columns(2)
        with c1:
            focus_mgr = st.selectbox("גוף מנהל", options=all_mgrs, index=0, key="isa_focus_mgr")
        tracks = sorted(full_df[full_df["manager"] == focus_mgr]["track"].unique().tolist()) if not full_df.empty else all_tracks
        tracks = tracks or [default_track]
        with c2:
            focus_trk = st.selectbox("מסלול", options=tracks, index=(tracks.index(default_track) if default_track in tracks else 0), key="isa_focus_trk")
        cache_key = f"isa_focus_{focus_mgr}_{focus_trk}".replace(" ", "_")
        if cache_key not in st.session_state:
            if st.button("🧠 צור סקירה", key="isa_focus_btn", type="primary"):
                with st.spinner("OpenAI מנתח את סגנון הניהול..."):
                    from institutional_strategy_analysis.ai_analyst import run_focused_analysis
                    st.session_state[cache_key] = run_focused_analysis(full_df if not full_df.empty else display_df, focus_mgr, focus_trk, None, context)
                st.rerun()
        else:
            _render_analysis_result(st.session_state[cache_key], cache_key, f"focused_{focus_mgr}", "isa_focus_refresh")

    elif mode == "compare":
        c1, c2, c3 = st.columns(3)
        with c1:
            mgr_a = st.selectbox("גוף א", options=all_mgrs, index=0, key="isa_cmp_mgr_a")
        with c2:
            mgr_b_opts = [m for m in all_mgrs if m != mgr_a] or all_mgrs
            mgr_b = st.selectbox("גוף ב", options=mgr_b_opts, index=0, key="isa_cmp_mgr_b")
        shared_tracks = sorted(set(full_df[full_df["manager"] == mgr_a]["track"].unique()) & set(full_df[full_df["manager"] == mgr_b]["track"].unique())) if not full_df.empty else all_tracks
        shared_tracks = shared_tracks or [default_track]
        with c3:
            cmp_trk = st.selectbox("מסלול להשוואה", options=shared_tracks, index=(shared_tracks.index(default_track) if default_track in shared_tracks else 0), key="isa_cmp_trk")
        cache_key = f"isa_cmp_{mgr_a}_{mgr_b}_{cmp_trk}".replace(" ", "_")
        if cache_key not in st.session_state:
            if st.button("⚖️ צור השוואה", key="isa_cmp_btn", type="primary"):
                with st.spinner("OpenAI יוצר השוואה בין שני הגופים..."):
                    from institutional_strategy_analysis.ai_analyst import run_comparison_analysis
                    st.session_state[cache_key] = run_comparison_analysis(full_df if not full_df.empty else display_df, mgr_a, cmp_trk, mgr_b, cmp_trk, context)
                st.rerun()
        else:
            _render_analysis_result(st.session_state[cache_key], cache_key, f"comparison_{mgr_a}_{mgr_b}", "isa_cmp_refresh")

    else:
        c1, c2 = st.columns(2)
        with c1:
            q_mgr = st.selectbox("גוף (אופציונלי)", options=["— כל הגופים —"] + all_mgrs, index=0, key="isa_q_mgr")
        with c2:
            q_trk = st.selectbox("מסלול (אופציונלי)", options=["— כל המסלולים —"] + all_tracks, index=(all_tracks.index(default_track)+1 if default_track in all_tracks else 0), key="isa_q_trk")
        question = st.text_area("מה תרצה לבדוק?", key="isa_free_question", height=110, placeholder="לדוגמה: האם הראל במסלול הכללי נראה דינמי יותר מהמתחרים, ומה זה אומר ללקוח שמעדיף יציבות?")
        cache_key = f"isa_free_{abs(hash((q_mgr, q_trk, question)))}"
        if cache_key not in st.session_state:
            if st.button("💬 שאל את ה-AI", key="isa_free_btn", type="primary"):
                with st.spinner("OpenAI בונה תשובה על בסיס הנתונים..."):
                    from institutional_strategy_analysis.ai_analyst import run_question_analysis
                    st.session_state[cache_key] = run_question_analysis(full_df if not full_df.empty else display_df, question, None if q_mgr == "— כל הגופים —" else q_mgr, None if q_trk == "— כל המסלולים —" else q_trk)
                st.rerun()
        else:
            _render_analysis_result(st.session_state[cache_key], cache_key, "free_question", "isa_free_refresh")


# ── Main entry point ──────────────────────────────────────────────────────────

def render_institutional_analysis():
    """Render the full "ניתוח אסטרטגיות מוסדיים" section."""

    with st.expander("📐 ניתוח אסטרטגיות מוסדיים", expanded=False):

        # ── Load data ─────────────────────────────────────────────────────
        with st.spinner("טוען נתונים..."):
            try:
                df_yearly, df_monthly, debug_info, errors = _load_data()
            except Exception as e:
                st.error(f"שגיאת טעינה: {e}")
                return

        if df_yearly.empty and df_monthly.empty:
            st.error("לא נטענו נתונים. בדוק את קישור הגיליון ואת הרשאות הגישה.")
            for e in errors:
                st.warning(e)
            return

        _render_debug(df_yearly, df_monthly, debug_info, errors)

        # ── Available options ─────────────────────────────────────────────
        opts = _options(df_yearly, df_monthly)

        # ── Filters ───────────────────────────────────────────────────────
        st.markdown("#### 🎛️ סינון")
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            sel_mgr = st.multiselect(
                "מנהל השקעות",
                options=opts["managers"],
                default=opts["managers"],
                help="בחר גוף מוסדי אחד או יותר. הנתונים מציגים את אסטרטגיית האלוקציה שלהם לאורך זמן.",
                key="isa_managers",
            )
        with fc2:
            avail_tracks = sorted({
                t for df in (df_yearly, df_monthly) if not df.empty
                for t in df[df["manager"].isin(sel_mgr)]["track"].unique()
            }) if sel_mgr else opts["tracks"]
            sel_tracks = st.multiselect(
                "מסלול",
                options=avail_tracks,
                default=avail_tracks,
                help="בחר מסלול השקעה — כגון כללי, מנייתי. מסלול כללי מאזן בין כמה נכסים.",
                key="isa_tracks",
            )
        with fc3:
            avail_allocs = sorted({
                a for df in (df_yearly, df_monthly) if not df.empty
                for a in df[
                    df["manager"].isin(sel_mgr) & df["track"].isin(sel_tracks)
                ]["allocation_name"].unique()
            }) if sel_mgr and sel_tracks else opts["allocation_names"]
            sel_allocs = st.multiselect(
                "רכיב אלוקציה",
                options=avail_allocs,
                default=avail_allocs[:5] if len(avail_allocs) > 5 else avail_allocs,
                help='בחר רכיבי חשיפה — למשל מניות, חו"ל, מט"ח, לא-סחיר.',
                key="isa_allocs",
            )

        # Time range
        rng_c, cust_c = st.columns([3, 2])
        with rng_c:
            sel_range = st.radio(
                "טווח זמן",
                options=["הכל", "YTD", "1Y", "3Y", "5Y", "מותאם אישית"],
                index=0, horizontal=True,
                label_visibility="collapsed",
                key="isa_range",
            )
            st.caption(
                "⏱️ **טווח זמן** — YTD ו-1Y משתמשים בנתונים חודשיים בלבד. "
                "3Y/5Y/הכל משלבים חודשי + שנתי."
            )
        with cust_c:
            custom_start = None
            if sel_range == "מותאם אישית":
                from institutional_strategy_analysis.series_builder import get_time_bounds
                min_d, max_d = get_time_bounds(df_yearly, df_monthly)
                custom_start = st.date_input(
                    "מתאריך", value=min_d.date(),
                    min_value=min_d.date(), max_value=max_d.date(),
                    key="isa_custom_start",
                )

        if not sel_mgr or not sel_tracks or not sel_allocs:
            st.info("יש לבחור לפחות מנהל, מסלול ורכיב אחד.")
            return

        # ── Build display series ──────────────────────────────────────────
        filters = {"managers": sel_mgr, "tracks": sel_tracks,
                   "allocation_names": sel_allocs}

        display_df = _build_series(df_yearly, df_monthly, sel_range, custom_start, filters)

        if display_df.empty:
            if sel_range in ("YTD", "1Y") and df_monthly.empty:
                st.warning(
                    "⚠️ לא נמצאו נתונים חודשיים. "
                    "YTD ו-1Y דורשים נתונים חודשיים. "
                    "נסה 'הכל' או '3Y' לקבלת נתונים שנתיים."
                )
            else:
                st.warning("אין נתונים לסינון הנוכחי.")
            return

        # Quick stats row
        n_dates  = display_df["date"].nunique()
        n_yearly = (display_df["frequency"] == "yearly").sum()  if "frequency" in display_df.columns else 0
        n_monthly = (display_df["frequency"] == "monthly").sum() if "frequency" in display_df.columns else 0
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("נקודות זמן", n_dates)
        sc2.metric("נתונים חודשיים", n_monthly // max(1, display_df["allocation_name"].nunique()))
        sc3.metric("נתונים שנתיים",  n_yearly  // max(1, display_df["allocation_name"].nunique()))

        # ── Tabs ──────────────────────────────────────────────────────────
        t_ts, t_snap, t_delta, t_heat, t_stats, t_rank = st.tabs([
            "📈 סדרת זמן",
            "📍 Snapshot",
            "🔄 שינוי / Delta",
            "🌡️ Heatmap",
            "📊 סטטיסטיקות",
            "🏆 דירוג",
        ])

        # ── Tab 1: Time series ────────────────────────────────────────────
        with t_ts:
            from institutional_strategy_analysis.charts import build_timeseries
            fig = build_timeseries(display_df)
            _safe_plotly(fig, key="isa_ts")
            st.caption(
                "קווים מלאים = נתונים חודשיים | קווים מקווקוים = נתונים שנתיים. "
                "שנים שמכוסות על ידי נתונים חודשיים לא מוצגות כשנתיות."
            )
            col_dl, _ = st.columns([1, 5])
            with col_dl:
                st.download_button("⬇️ CSV", data=_csv(display_df),
                                   file_name="isa_timeseries.csv", mime="text/csv",
                                   key="isa_dl_ts")

        # ── Tab 2: Snapshot ───────────────────────────────────────────────
        with t_snap:
            max_d = display_df["date"].max().date()
            min_d = display_df["date"].min().date()
            snap_date = st.date_input(
                "תאריך Snapshot",
                value=max_d, min_value=min_d, max_value=max_d,
                help="מציג את הערך האחרון הידוע עד לתאריך שנבחר.",
                key="isa_snap_date",
            )
            from institutional_strategy_analysis.charts import build_snapshot
            _safe_plotly(build_snapshot(display_df, pd.Timestamp(snap_date)), key="isa_snap")

            snap_df = display_df[display_df["date"] <= pd.Timestamp(snap_date)]
            if not snap_df.empty:
                i = snap_df.groupby(["manager", "track", "allocation_name"])["date"].idxmax()
                tbl = snap_df.loc[i][["manager", "track", "allocation_name",
                                       "allocation_value", "date"]].copy()
                tbl["date"] = tbl["date"].dt.strftime("%Y-%m")
                tbl.columns = ["מנהל", "מסלול", "רכיב", "ערך (%)", "תאריך"]
                st.dataframe(tbl.sort_values("ערך (%)", ascending=False)
                               .reset_index(drop=True),
                             use_container_width=True, hide_index=True)

        # ── Tab 3: Delta ──────────────────────────────────────────────────
        with t_delta:
            min_d = display_df["date"].min().date()
            max_d = display_df["date"].max().date()
            dc1, dc2 = st.columns(2)
            with dc1:
                date_a = st.date_input("תאריך A (מוצא)",
                                       value=_clamp(max_d - timedelta(days=365), min_d, max_d),
                                       min_value=min_d, max_value=max_d,
                                       help="תאריך ההתחלה להשוואה.",
                                       key="isa_da")
            with dc2:
                date_b = st.date_input("תאריך B (יעד)", value=max_d,
                                       min_value=min_d, max_value=max_d,
                                       help="תאריך הסיום להשוואה.",
                                       key="isa_db")
            if date_a >= date_b:
                st.warning("תאריך A חייב להיות לפני B.")
            else:
                from institutional_strategy_analysis.charts import build_delta
                fig_d, delta_tbl = build_delta(display_df,
                                                pd.Timestamp(date_a),
                                                pd.Timestamp(date_b))
                _safe_plotly(fig_d, key="isa_delta")
                if not delta_tbl.empty:
                    st.dataframe(delta_tbl.reset_index(drop=True),
                                 use_container_width=True, hide_index=True)
                    col_dl2, _ = st.columns([1, 5])
                    with col_dl2:
                        st.download_button("⬇️ CSV", data=_csv(delta_tbl),
                                           file_name="isa_delta.csv", mime="text/csv",
                                           key="isa_dl_delta")

        # ── Tab 4: Heatmap ────────────────────────────────────────────────
        with t_heat:
            from institutional_strategy_analysis.charts import build_heatmap
            heat_df = display_df.copy()
            if display_df["date"].nunique() > 48:
                cutoff = display_df["date"].max() - pd.DateOffset(months=48)
                heat_df = display_df[display_df["date"] >= cutoff]
                st.caption("מוצגים 48 חודשים אחרונים. בחר 'הכל' לצפייה מלאה.")
            _safe_plotly(build_heatmap(heat_df), key="isa_heat")

        # ── Tab 5: Summary stats ──────────────────────────────────────────
        with t_stats:
            from institutional_strategy_analysis.charts import build_summary_stats
            stats = build_summary_stats(display_df)
            if stats.empty:
                st.info("אין מספיק נתונים לסטטיסטיקה.")
            else:
                st.dataframe(stats.reset_index(drop=True),
                             use_container_width=True, hide_index=True)
                col_dl3, _ = st.columns([1, 5])
                with col_dl3:
                    st.download_button("⬇️ CSV", data=_csv(stats),
                                       file_name="isa_stats.csv", mime="text/csv",
                                       key="isa_dl_stats")

        # ── Tab 6: Ranking ────────────────────────────────────────────────
        with t_rank:
            from institutional_strategy_analysis.charts import build_ranking
            if display_df["allocation_name"].nunique() > 1:
                rank_alloc = st.selectbox(
                    "רכיב לדירוג",
                    options=sorted(display_df["allocation_name"].unique()),
                    help="בחר רכיב שלפיו יוצג הדירוג החודשי.",
                    key="isa_rank_alloc",
                )
                rank_df = display_df[display_df["allocation_name"] == rank_alloc]
            else:
                rank_df = display_df

            _safe_plotly(
                build_ranking(rank_df,
                              title=f"דירוג מנהלים — {rank_df['allocation_name'].iloc[0]}"
                              if not rank_df.empty else "דירוג"),
                key="isa_rank",
            )

            # Volatility table
            if not rank_df.empty:
                vol = []
                for (mgr, trk), g in rank_df.groupby(["manager", "track"]):
                    chg = g.sort_values("date")["allocation_value"].diff().dropna()
                    vol.append({
                        "מנהל": mgr, "מסלול": trk,
                        "תנודתיות (STD)": round(chg.std(), 3) if len(chg) > 1 else float("nan"),
                        "שינוי מקסימלי": round(chg.abs().max(), 3) if not chg.empty else float("nan"),
                    })
                if vol:
                    st.caption("תנודתיות לפי מנהל:")
                    st.dataframe(
                        pd.DataFrame(vol).sort_values("תנודתיות (STD)", ascending=False)
                          .reset_index(drop=True),
                        use_container_width=True, hide_index=True,
                    )

        # ── Raw data ──────────────────────────────────────────────────────
        with st.expander("📋 נתונים גולמיים", expanded=False):
            disp = display_df.copy()
            if "date" in disp.columns:
                disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(disp.reset_index(drop=True),
                         use_container_width=True, hide_index=True)
            st.download_button("⬇️ ייצוא כל הנתונים", data=_csv(display_df),
                               file_name="isa_all.csv", mime="text/csv",
                               key="isa_dl_all")

        # ── AI Analysis (full dataset for peer comparison) ────────────────
        # Build full_df — all managers, same track(s) — for relative peer analysis
        all_filters = {
            "managers":        opts["managers"],
            "tracks":          sel_tracks,
            "allocation_names": sel_allocs,
        }
        try:
            full_df = _build_series(df_yearly, df_monthly, sel_range, custom_start, all_filters)
        except Exception:
            full_df = display_df

        ai_context = {
            "managers":         sel_mgr,
            "tracks":           sel_tracks,
            "allocation_names": sel_allocs,
            "selected_range":   sel_range,
            "date_min":         display_df["date"].min().strftime("%Y-%m") if not display_df.empty else "",
            "date_max":         display_df["date"].max().strftime("%Y-%m") if not display_df.empty else "",
        }
        _render_ai_section(display_df, full_df, ai_context, sel_mgr, sel_tracks)
