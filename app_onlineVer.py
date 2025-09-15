"""
Run with: streamlit run app.py
CSV visualizer for GovernmentFinanceMonthly with bar chart only,
year/month filters, optional Data Series filter, and an OpenAI chat assistant.
"""

import os
import re
from typing import List, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import requests

# ---------- Configuration ----------
# Set your OpenAI API key here. Change this value later as needed.
OPENAI_API_KEY = "GENERATE YOUR OPENAI KEY FROM https://platform.openai.com/account/api-keys AND PLACE IT HERE!"
# Optional: domain context shown to the model on every question
DOMAIN_CONTEXT = (
    "The dataset shown and any analysis requested are based on "
    "Singapore Government's Monthly Financial Data. Interpret series/values "
    "accordingly and keep explanations grounded to this context."
)

# Gemini configuration (hardcoded for now)
GEMINI_API_KEY = "GENERATE YOUR GEMINI KEY FROM https://console.cloud.google.com/apis/credentials AND PLACE IT HERE!"
GEMINI_MODEL = "gemini-1.5-flash"


# ---------- Setup ----------
st.set_page_config(page_title="YF Weds Demo", layout="wide")
# Inject lightweight CSS for styled Gemini responses (once)
if not st.session_state.get("css_injected"):
    st.markdown(
        """
        <style>
        .gemini-bubble {
          background: #E8F0FE;
          border: 1px solid #4285F4;
          padding: 12px 14px;
          border-radius: 10px;
          color: #202124;
          margin: 8px 0 2px 0;
          display: inline-block;
          max-width: 100%;
        }
        .gemini-bubble .gemini-label {
          font-size: 12px; font-weight: 600; color: #1a73e8;
          margin-bottom: 6px; text-transform: uppercase; letter-spacing: .3px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.css_injected = True


# ---------- Data loading ----------
def load_csv_dataframe() -> pd.DataFrame:
    candidates = [
        "GovernmentFinanceMonthly.csv",
        "GovernmentFinanceMonthly",
    ]
    path: Optional[str] = None
    for name in candidates:
        if os.path.exists(name):
            path = name
            break
    if not path:
        matches = [
            f for f in os.listdir(".")
            if f.lower().startswith("governmentfinancemonthly") and f.lower().endswith(".csv")
        ]
        path = matches[0] if matches else None

    if not path:
        st.error("Could not find 'GovernmentFinanceMonthly.csv' in this folder.")
        st.stop()

    df_local = pd.read_csv(path)

    # Attempt to parse datetime-like columns
    for col in df_local.columns:
        if df_local[col].dtype == object:
            lowered = str(col).lower()
            if any(k in lowered for k in ["date", "month", "time", "period"]):
                try:
                    df_local[col] = pd.to_datetime(df_local[col], errors="raise")
                except Exception:
                    pass
    return df_local


df = load_csv_dataframe()


# ---------- Helpers for series and month columns ----------
def find_series_column(frame: pd.DataFrame) -> Optional[str]:
    # Prefer exact match on 'Data Series' (case-insensitive)
    for c in frame.columns:
        if str(c).strip().lower() == "data series":
            return c
    # Otherwise try common alternatives
    candidates = ["series", "data_series", "dataseries", "category", "name", "label"]
    lower_map = {str(c).strip().lower(): c for c in frame.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]
    # Fallback: first non-numeric column
    cat_cols = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    return cat_cols[0] if cat_cols else None


MONTH_ABBRS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_PATTERN = re.compile(r"^(?P<year>\d{4})(?P<mon>" + "|".join(MONTH_ABBRS) + r")$")


def find_month_columns(frame: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in frame.columns:
        if MONTH_PATTERN.match(str(c)):
            cols.append(c)
    # Sort by YYYYMon chronological
    def sort_key(cname: str):
        m = MONTH_PATTERN.match(str(cname))
        if not m:
            return (9999, 13)
        y = int(m.group("year"))
        mon = MONTH_ABBRS.index(m.group("mon")) + 1
        return (y, mon)

    cols.sort(key=sort_key)
    return cols


# Predefined Data Series groups (names must match your CSV's Data Series labels)
SERIES_GROUPS: dict[str, list[str]] = {
    "L1: Cash Surplus/Deficit": [
        "Cash Surplus/Deficit",
        "Net Cash Inflow From Operating Activities",
        "Cash Receipts From Operating Activities",
        "Cash Payments For Operating Activities",
        "Net Cash Outflow From Investments In Non-Financial Assets",
        "Purchases Of Non-Financial Assets",
        "Sales Of Non-Financial Assets",
    ],
    "L2: Net Cash Inflow from Operating Activities": [
        "Net Cash Inflow From Operating Activities",
        "Cash Receipts From Operating Activities",
        "Cash Payments For Operating Activities",
    ],
    "L2: Net Cash Outflow From Investments In Non-Financial Assets": [
        "Net Cash Outflow From Investments In Non-Financial Assets",
        "Purchases Of Non-Financial Assets",
        "Sales Of Non-Financial Assets",
    ],
    "L1: Net Cash Inflow From Financing Activities": [
        "Net Cash Inflow From Financing Activities",
        "Net Incurrence Of Liabilities",
        "Domestic",
        "Foreign",
        "Net Acquisition Of Financial Assets Other Than Cash",
        "Domestic Excluding Cash",
        "Foreign Excluding Cash",
    ],
    "L2: Net Incurrence Of Liabilities": [
        "Net Incurrence Of Liabilities",
        "Domestic",
        "Foreign",
    ],
    "L2: Net Acquisition Of Financial Assets Other Than Cash": [
        "Net Acquisition Of Financial Assets Other Than Cash",
        "Domestic Excluding Cash",
        "Foreign Excluding Cash",
    ],
    # Level-only shortcuts
    "L1 only": [
        "Cash Surplus/Deficit",
        "Net Cash Inflow From Financing Activities",
    ],
    "L2 only": [
        "Net Cash Inflow From Operating Activities",
        "Net Cash Outflow From Investments In Non-Financial Assets",
        "Net Incurrence Of Liabilities",
        "Net Acquisition Of Financial Assets Other Than Cash",
    ],
    "L3 only": [
        "Cash Receipts From Operating Activities",
        "Cash Payments For Operating Activities",
        "Purchases Of Non-Financial Assets",
        "Sales Of Non-Financial Assets",
        "Domestic",
        "Foreign",
        "Domestic Excluding Cash",
        "Foreign Excluding Cash",
    ],
}


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


# ---------- Visual controls (bar chart only) ----------
st.subheader("Singapore Government Finance Monthly Data Visualizer")

series_col = find_series_column(df)
month_cols = find_month_columns(df)
if not month_cols:
    st.error("No month columns like 2025Jun, 2025May found. Please ensure columns are in 'YYYYMon' format.")
    st.dataframe(df.head(20))
    st.stop()

st.caption(f"Detected series column: {series_col if series_col else '(none)'} | Months detected: {len(month_cols)}")

# Initialize state defaults
years_all = sorted({int(MONTH_PATTERN.match(str(m)).group("year")) for m in month_cols})
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "bar"
if "include_all_years" not in st.session_state:
    st.session_state.include_all_years = True
if "selected_years" not in st.session_state:
    st.session_state.selected_years = years_all[:]
available_months_all = [m for m in month_cols if int(MONTH_PATTERN.match(str(m)).group("year")) in st.session_state.selected_years]
# Reverse month order so most recent appears on the left in charts/tables
available_months_all = list(reversed(available_months_all))
if "include_all_months" not in st.session_state:
    st.session_state.include_all_months = True
if "selected_months" not in st.session_state:
    st.session_state.selected_months = available_months_all[:]
if series_col:
    if "enable_series" not in st.session_state:
        st.session_state.enable_series = False
    if "selected_series" not in st.session_state:
        st.session_state.selected_series = sorted(df[series_col].dropna().astype(str).unique().tolist())
if "group_choice" not in st.session_state:
    st.session_state.group_choice = "(none)"

# Chart type toggle (no widget key; we sync to session_state after)
chart_type_choice = st.radio(
    "Chart Type",
    ["bar", "line"],
    index=0 if st.session_state.chart_type == "bar" else 1,
    horizontal=True,
)

# Year filter (no widget keys)
year_map = {m: int(MONTH_PATTERN.match(str(m)).group("year")) for m in month_cols}
years = sorted(set(year_map.values()))
include_all_years_choice = st.checkbox("Include all years", value=st.session_state.include_all_years)
if not include_all_years_choice:
    selected_years_choice = st.multiselect(
        "Years", options=years, default=st.session_state.selected_years or years
    )
else:
    selected_years_choice = years

# Month filter (within selected years)
available_months = [m for m in month_cols if year_map[m] in (selected_years_choice or years)]
# Reverse for display: latest months first
available_months = list(reversed(available_months))
include_all_months_choice = st.checkbox("Include all months", value=st.session_state.include_all_months)
if not include_all_months_choice:
    selected_months_choice = st.multiselect(
        "Months", options=available_months, default=st.session_state.selected_months or available_months
    )
else:
    selected_months_choice = available_months

# Data Series Group (optional)
group_names = ["(none)"] + list(SERIES_GROUPS.keys())
group_index = 0
if st.session_state.get("group_choice") in group_names:
    group_index = group_names.index(st.session_state.get("group_choice"))
group_choice = st.selectbox(
    "Data Series Group (optional)", options=group_names, index=group_index
)
st.session_state.group_choice = group_choice

# Optional Data Series filter (no widget keys)
if series_col:
    # Precompute available series and any group defaults
    series_values_all = sorted(df[series_col].dropna().astype(str).unique().tolist())
    group_series_default = None
    if group_choice != "(none)":
        lower_map = {_norm_label(v): v for v in series_values_all}
        targets = [_norm_label(x) for x in SERIES_GROUPS.get(group_choice, [])]
        group_series_default = [lower_map[t] for t in targets if t in lower_map]
        if group_series_default is None or len(group_series_default) == 0:
            st.info("No matching Data Series found for the selected group in this dataset.")

    enable_series_choice = st.checkbox(
        "Filter Data Series",
        value=(st.session_state.enable_series or (group_choice != "(none)")),
    )
    if enable_series_choice:
        series_values = series_values_all
        selected_series_choice = st.multiselect(
            "Data Series",
            options=series_values,
            default=(group_series_default if (group_series_default and group_choice != "(none)") else (st.session_state.selected_series or series_values)),
        )
    else:
        selected_series_choice = st.session_state.selected_series
else:
    enable_series_choice = False
    selected_series_choice = None

# Sync control choices back to session_state
st.session_state.chart_type = chart_type_choice
st.session_state.include_all_years = include_all_years_choice
st.session_state.selected_years = selected_years_choice
st.session_state.include_all_months = include_all_months_choice
st.session_state.selected_months = selected_months_choice
if series_col:
    st.session_state.enable_series = enable_series_choice
    st.session_state.selected_series = selected_series_choice

if not st.session_state.selected_months:
    st.info("No months selected — adjust the year/month filters.")
    st.stop()

# Optional Data Series filter
# Prepare long-form data (melt)
data = df.copy()
if series_col and st.session_state.get("enable_series"):
    sels = st.session_state.get("selected_series") or []
    if sels:
        data = data[data[series_col].astype(str).isin(sels)]

id_vars = [series_col] if series_col else []
long_df = pd.melt(
    data,
    id_vars=id_vars,
    value_vars=st.session_state.selected_months,
    var_name="Month",
    value_name="Value",
)

# Clean and order melted data
long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
long_df["Month"] = pd.Categorical(
    long_df["Month"], categories=st.session_state.selected_months, ordered=True
)

# Ascending month order for charts (oldest on left → newest on right)
def _month_sort_key(m: str):
    mm = MONTH_PATTERN.match(str(m))
    if not mm:
        return (9999, 13)
    y = int(mm.group("year"))
    mon = MONTH_ABBRS.index(mm.group("mon")) + 1
    return (y, mon)

chart_month_order = sorted(list(st.session_state.selected_months), key=_month_sort_key)

# Render bar chart
if st.session_state.chart_type == "line":
    fig = px.line(
        long_df,
        x="Month",
        y="Value",
        color=series_col if series_col else None,
        category_orders={"Month": chart_month_order},
        markers=True,
    )
else:
    fig = px.bar(
        long_df,
        x="Month",
        y="Value",
        color=series_col if series_col else None,
        category_orders={"Month": chart_month_order},
    )

# Aggregated cumulative totals per Data Series against time (reset each year)
month_to_idx = {m: i + 1 for i, m in enumerate(MONTH_ABBRS)}

def add_year_monidx(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    # Extract year and month index from Month labels like '2025Jan'
    years = []
    monidx = []
    for m in out["Month"].astype(str):
        mm = MONTH_PATTERN.match(m)
        if mm:
            years.append(int(mm.group("year")))
            monidx.append(month_to_idx.get(mm.group("mon"), None))
        else:
            years.append(None)
            monidx.append(None)
    out["Year"] = years
    out["MonIdx"] = monidx
    return out

if series_col:
    monthly = long_df.groupby(["Month", series_col], as_index=False)["Value"].sum()
    monthly["Month"] = pd.Categorical(
        monthly["Month"], categories=chart_month_order, ordered=True
    )
    monthly = add_year_monidx(monthly)
    monthly = monthly.sort_values(["Year", "MonIdx"])  # within-year order
    monthly["CumuValue"] = monthly.groupby([series_col, "Year"], sort=False)["Value"].cumsum()

    if st.session_state.chart_type == "line":
        fig_tot = px.line(
            monthly,
            x="Month",
            y="CumuValue",
            color=series_col,
            category_orders={"Month": chart_month_order},
            markers=True,
        )
    else:
        fig_tot = px.bar(
            monthly,
            x="Month",
            y="CumuValue",
            color=series_col,
            category_orders={"Month": chart_month_order},
        )
    st.markdown("### Cumulative Total per Data Series [SGD$M]")
    st.markdown("_Note: cumulative totals reset each year._")
    st.plotly_chart(fig_tot, use_container_width=True)
else:
    # No series column: cumulative total per month, reset each year
    monthly = long_df.groupby(["Month"], as_index=False)["Value"].sum()
    monthly["Month"] = pd.Categorical(
        monthly["Month"], categories=chart_month_order, ordered=True
    )
    monthly = add_year_monidx(monthly)
    monthly = monthly.sort_values(["Year", "MonIdx"])  # within-year order
    monthly["CumuValue"] = monthly.groupby(["Year"], sort=False)["Value"].cumsum()

    if st.session_state.chart_type == "line":
        fig_tot = px.line(
            monthly,
            x="Month",
            y="CumuValue",
            category_orders={"Month": chart_month_order},
            markers=True,
        )
    else:
        fig_tot = px.bar(
            monthly,
            x="Month",
            y="CumuValue",
            category_orders={"Month": chart_month_order},
        )
    st.markdown("### Cumulative Total by Month by Year")
    st.plotly_chart(fig_tot, use_container_width=True)

# Place reported value chart below the cumulative chart
st.markdown("### Reported Value per Data Series [SGD$M]")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Dataset preview"):
    st.markdown("Raw CSV (first 50 rows)")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("Current view — Cumulative (wide, resets each year)")
    try:
        cuw = monthly.copy()
        # Ensure standard column
        if "CumuValue" in cuw.columns and "CumValue" not in cuw.columns:
            cuw.rename(columns={"CumuValue": "CumValue"}, inplace=True)
        cuw["CumValue"] = pd.to_numeric(cuw.get("CumValue"), errors="coerce")
        if series_col and series_col in cuw.columns:
            idx_name = series_col
        else:
            idx_name = "Data Series"
            cuw[idx_name] = "Total"
        # Pivot cumulative; sum handles any duplicates
        wide_cu = (
            cuw.pivot_table(index=idx_name, columns="Month", values="CumValue", aggfunc="sum")
            .reindex(columns=st.session_state.selected_months)
            .reset_index()
        )
        # Ensure first column is literally named "Data Series" and months follow in order
        if idx_name != "Data Series":
            wide_cu.rename(columns={idx_name: "Data Series"}, inplace=True)
        month_cols_present = [m for m in st.session_state.selected_months if m in wide_cu.columns]
        wide_cu = wide_cu[["Data Series"] + month_cols_present]

        # Reorder rows to the requested hierarchy sequence
        try:
            desired_order = [
                "Cash Surplus/Deficit",
                "Net Cash Inflow From Operating Activities",
                "Cash Receipts From Operating Activities",
                "Cash Payments For Operating Activities",
                "Net Cash Outflow From Investments In Non-Financial Assets",
                "Purchases Of Non-Financial Assets",
                "Sales Of Non-Financial Assets",
                "Net Cash Inflow From Financing Activities",
                "Net Incurrence Of Liabilities",
                "Domestic",
                "Foreign",
                "Net Acquisition Of Financial Assets Other Than Cash",
                "Domestic Excluding Cash",
                "Foreign Excluding Cash",
            ]
            order_map = {_norm_label(n): i for i, n in enumerate(desired_order)}
            wide_cu["__ord"] = wide_cu["Data Series"].apply(lambda s: order_map.get(_norm_label(s), 10**6))
            wide_cu = wide_cu.sort_values(["__ord", "Data Series"]).drop(columns=["__ord"])  # keep unmatched at bottom
        except Exception:
            pass
        st.dataframe(wide_cu, use_container_width=True)
        st.download_button(
            "Download cumulative (wide)",
            data=wide_cu.to_csv(index=False).encode("utf-8"),
            file_name="current_view_cumulative_wide.csv",
            mime="text/csv",
        )
    except Exception:
        pass


# ---------- Chat: Explain the data ----------
st.divider()
st.subheader("Data Assistant (OpenAI)")

# Model selector for OpenAI
available_models = [
    "gpt-4.1-mini",
    "gpt-4o-mini",
]
default_model = st.session_state.get("openai_model", "gpt-4.1-mini")
if default_model not in available_models:
    default_model = "gpt-4.1-mini"
model_choice = st.selectbox("OpenAI model", options=available_models, index=available_models.index(default_model))
st.caption("Hint: 4.1‑mini = balanced accuracy/cost; 4o‑mini = cheapest/quickest.")
st.session_state.openai_model = model_choice

# Toggle: cross-check with Gemini
check_with_gemini = st.checkbox("Check with Gemini", value=st.session_state.get("check_gemini", False))
st.session_state.check_gemini = check_with_gemini

def get_openai_client() -> Optional[OpenAI]:
    key: Optional[str] = OPENAI_API_KEY.strip()
    if not key:
        return None
    return OpenAI(api_key=key)


def ask_gemini(prompt: str, model: str = GEMINI_MODEL) -> str:
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": GEMINI_API_KEY}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ],
                }
            ]
        }
        r = requests.post(url, params=params, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        # Extract first candidate text
        cand = (data.get("candidates") or [{}])[0]
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "\n".join([t for t in texts if t]).strip() or "(No response text)"
    except Exception as e:
        return f"Gemini error: {e}"


def build_data_profile(frame: pd.DataFrame,
                       series_col: Optional[str],
                       months: List[str],
                       sel_months: List[str],
                       sel_series: Optional[List[str]],
                       sel_years: Optional[List[int]] = None,
                       group_choice: Optional[str] = None,
                       view_long_df: Optional[pd.DataFrame] = None,
                       view_cumu_df: Optional[pd.DataFrame] = None) -> str:
    lines: List[str] = []
    lines.append(f"shape: {frame.shape[0]} rows x {frame.shape[1]} cols")
    lines.append(f"series_col: {series_col}")
    years_present = sorted({int(MONTH_PATTERN.match(m).group('year')) for m in months}) if months else []
    lines.append(f"years_detected: {years_present}")
    # Selected years inferred from selected months if not provided
    years_selected = sorted({int(MONTH_PATTERN.match(m).group('year')) for m in sel_months}) if sel_months else (sel_years or [])
    lines.append(f"years_selected: {years_selected}")
    lines.append(f"months_detected: {months[:6]}... total={len(months)}")
    lines.append(f"months_selected_count: {len(sel_months)}")
    # Per-year counts for selected months
    try:
        by_year_counts = {}
        for m in sel_months:
            yr = int(MONTH_PATTERN.match(m).group('year'))
            by_year_counts[yr] = by_year_counts.get(yr, 0) + 1
        lines.append(f"months_selected_by_year: {by_year_counts}")
    except Exception:
        pass
    if group_choice is not None:
        lines.append(f"group_choice: {group_choice}")
        try:
            defined = SERIES_GROUPS.get(group_choice, [])
            lines.append(f"group_members_defined_count: {len(defined)}")
            if series_col and series_col in frame.columns:
                present_vals = frame[series_col].dropna().astype(str).unique().tolist()
                present_lower = { _norm_label(v): v for v in present_vals }
                defined_lower = [_norm_label(x) for x in defined]
                present_mapped = [present_lower[d] for d in defined_lower if d in present_lower]
                missing = [defined[i] for i,d in enumerate(defined_lower) if d not in present_lower]
                if present_mapped:
                    lines.append(f"group_members_present: {present_mapped}")
                if missing:
                    lines.append(f"group_members_missing: {missing}")
        except Exception:
            pass

    if series_col and series_col in frame.columns:
        vals = frame[series_col].dropna().astype(str)
        uniq = vals.unique().tolist()
        lines.append(f"series_count: {len(uniq)}")
        if sel_series is not None:
            try:
                sel_list = [str(s) for s in sel_series]
                lines.append(f"series_selected_count: {len(sel_list)}")
                lines.append(f"series_selected: {sel_list}")
            except Exception:
                lines.append(f"series_selected_count: {len(sel_series)}")
    # Small sample
    # Build a representative sample across selected years: pick the first available month of each year
    pick_cols: List[str] = []
    try:
        for yr in years_selected:
            first_in_year = next((m for m in sel_months if int(MONTH_PATTERN.match(m).group('year')) == yr), None)
            if first_in_year:
                pick_cols.append(first_in_year)
    except Exception:
        # Fallback to the first up to 6 selected months
        pick_cols = sel_months[:6]
    sample_cols = ([series_col] if series_col else []) + pick_cols[:6]
    sample_cols = [c for c in sample_cols if c in frame.columns]
    try:
        sample_csv = frame[sample_cols].head(5).to_csv(index=False)
        lines.append("sample:\n" + sample_csv)
    except Exception:
        pass

    # Include the currently visible datasets (what the user sees)
    try:
        if view_long_df is not None:
            # Keep only relevant columns
            cols = [c for c in ["Month", series_col, "Value"] if c and c in view_long_df.columns]
            view_long_csv = view_long_df[cols].to_csv(index=False)
            lines.append("view_long:\n" + view_long_csv)
    except Exception:
        pass
    try:
        if view_cumu_df is not None:
            cu = view_cumu_df.copy()
            if "CumuValue" in cu.columns and "CumValue" not in cu.columns:
                cu.rename(columns={"CumuValue": "CumValue"}, inplace=True)
            cols = [c for c in ["Month", series_col, "CumValue"] if c and c in cu.columns]
            view_cumu_csv = cu[cols].to_csv(index=False)
            lines.append("view_cumulative:\n" + view_cumu_csv)
    except Exception:
        pass

    # Add explicit December values table for selected years, if present
    try:
        dec_cols = [m for m in sel_months if MONTH_PATTERN.match(m) and MONTH_PATTERN.match(m).group('mon') == 'Dec']
        if dec_cols:
            sub = frame.copy()
            if series_col and sel_series is not None:
                sub = sub[sub[series_col].astype(str).isin([str(s) for s in sel_series])]
            idv = [series_col] if series_col else []
            melted = pd.melt(sub, id_vars=idv, value_vars=dec_cols, var_name="Month", value_name="Value")
            melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
            if series_col:
                dec_tbl = (
                    melted.groupby(["Month", series_col], as_index=False)["Value"].sum()
                    .sort_values(["Month", series_col])
                )
            else:
                dec_tbl = (
                    melted.groupby(["Month"], as_index=False)["Value"].sum()
                    .sort_values(["Month"]) 
                )
            lines.append("december_values:\n" + dec_tbl.to_csv(index=False))
    except Exception:
        pass
    return "\n".join(lines)


def _html_escape(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    if role == "gemini":
        # Render past Gemini messages with styled bubble
        safe = _html_escape(content).replace("\n", "<br/>")
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="gemini-bubble"><div class="gemini-label">Gemini feedback</div><div>{safe}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        with st.chat_message(role):
            st.markdown(content)

user_msg = st.chat_input("Ask or command (e.g., 'see L1 only', 'relationship between election years and L1 trend', 'see 2019 to 2025', 'effect of covid years')")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Apply simple intents from the message
    def apply_intents(msg: str) -> str:
        msg_low = msg.lower()
        applied = []

        # Switch chart type
        if "line" in msg_low and "bar" not in msg_low:
            st.session_state.chart_type = "line"
            applied.append("chart=line")
        elif "bar" in msg_low:
            st.session_state.chart_type = "bar"
            applied.append("chart=bar")

        # Year(s) selection
        years_found = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", msg_low)]
        if len(years_found) >= 2:
            y1, y2 = min(years_found), max(years_found)
            st.session_state.include_all_years = False
            st.session_state.selected_years = [y for y in years if y1 <= y <= y2]
            applied.append(f"years={y1}–{y2}")
        elif len(years_found) == 1:
            y = years_found[0]
            st.session_state.include_all_years = False
            st.session_state.selected_years = [y]
            applied.append(f"years={y}")

        # Sync months with updated years if 'all months' is on
        avail_months_local = [m for m in month_cols if year_map[m] in (st.session_state.selected_years or years)]
        avail_months_local = list(reversed(avail_months_local))
        if "all months" in msg_low or st.session_state.include_all_months:
            st.session_state.include_all_months = True
            st.session_state.selected_months = avail_months_local

        # Series selection
        if series_col:
            series_values = sorted(df[series_col].dropna().astype(str).unique().tolist())
            matches = [v for v in series_values if v.lower() in msg_low]
            if "all series" in msg_low or ("show all" in msg_low and "series" in msg_low):
                st.session_state.enable_series = False
                applied.append("series=all")
            elif matches:
                st.session_state.enable_series = True
                st.session_state.selected_series = matches
                applied.append("series=" + ",".join(matches))

        # Data Series Group selection (by group name like "L1 only", etc.)
        # Try to find any defined group name mentioned in the message
        found_group = None
        for gname in SERIES_GROUPS.keys():
            if _norm_label(gname) in msg_low:
                found_group = gname
                break
        if found_group is not None:
            st.session_state.group_choice = found_group
            # When a group is chosen, auto-enable series filter and set its members if available
            if series_col:
                series_values_all = sorted(df[series_col].dropna().astype(str).unique().tolist())
                lower_map = {_norm_label(v): v for v in series_values_all}
                targets = [_norm_label(x) for x in SERIES_GROUPS.get(found_group, [])]
                mapped = [lower_map[t] for t in targets if t in lower_map]
                if mapped:
                    st.session_state.enable_series = True
                    st.session_state.selected_series = mapped
            applied.append(f"group={found_group}")

        return "; ".join(applied)

    changes = apply_intents(user_msg)
    ack = f"Applied: {changes}" if changes else None

    client = get_openai_client()
    if client is None:
        assistant_text = "Please set OPENAI_API_KEY at the top of app.py to enable the assistant."
    else:
        profile = build_data_profile(
            df,
            series_col,
            month_cols,
            st.session_state.selected_months,
            st.session_state.get("selected_series") if st.session_state.get("enable_series") else None,
            st.session_state.get("selected_years"),
            st.session_state.get("group_choice"),
            long_df,
            locals().get("monthly"),
        )
        instructions = (
            "You are a helpful data analyst. "
            + DOMAIN_CONTEXT + " "
            "Use ONLY the provided data profile (view_long/view_cumulative) as the source of truth for values. "
            "If a Data Series Group or explicit series selection is provided, restrict analysis to those series. "
            "When listing values across months or series, prefer a concise Markdown table instead of bullets. "
            "Keep tables compact (≤12 rows, ≤6 columns) and add a brief 1–2 sentence summary after the table. "
            "Avoid long bullet lists and avoid speculating beyond the selected data."
        )
        try:
            resp = client.responses.create(
                model=st.session_state.get("openai_model", "gpt-4.1-mini"),
                instructions=instructions,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"DATA PROFILE\n{profile}\n\nQUESTION\n{user_msg}"}
                        ],
                    }
                ],
            )
            assistant_text = getattr(resp, "output_text", None)
            if not assistant_text:
                chunks: List[str] = []
                for out in getattr(resp, "output", []) or []:
                    if getattr(out, "type", None) == "message":
                        for b in getattr(out, "content", []) or []:
                            if getattr(b, "type", None) == "output_text":
                                chunks.append(getattr(b, "text", ""))
                assistant_text = " ".join(chunks) or "(No response text)"
        except Exception as e:
            assistant_text = f"Error from OpenAI: {e}"

    final_text = (ack + "\n\n" if ack else "") + (assistant_text or "")
    st.session_state.chat.append(("assistant", final_text))
    with st.chat_message("assistant"):
        st.markdown(final_text)

    # Optional cross-check with Gemini
    if st.session_state.get("check_gemini"):
        gemini_prompt = (
            "You are a second analyst. "
            + DOMAIN_CONTEXT + "\n\n"
            "Formatting constraints: Do NOT use any Markdown emphasis (no bold, italics, underline), "
            "and avoid code blocks. Provide plain text only.\n"
            "Given the data profile below and the user's question, review the provided answer; "
            "point out any inaccuracies or missing considerations, and suggest concrete ways to improve "
            "the answer's clarity, structure, and usefulness. When helpful, propose a brief improved answer. "
            "Keep it concise and polite.\n\n"
            f"DATA PROFILE\n{profile}\n\nQUESTION\n{user_msg}\n\nANSWER\n{assistant_text}"
        )
        gemini_text = ask_gemini(gemini_prompt)
        # Save as a dedicated 'gemini' role so history renders with bubble
        st.session_state.chat.append(("gemini", gemini_text))
        safe = _html_escape(gemini_text).replace("\n", "<br/>")
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="gemini-bubble"><div class="gemini-label">Gemini feedback</div><div>{safe}</div></div>',
                unsafe_allow_html=True,
            )

    # After answering, rerun so the chart reflects applied filters immediately
    if changes:
        st.rerun()
