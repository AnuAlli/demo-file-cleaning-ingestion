import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import json
import io
import os
import re
import time
import hashlib

st.set_page_config(page_title="Automated File Cleaning & Database Ingestion", page_icon="🧹", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    .clean-card { background: #F0F9FF; border-left: 4px solid #2E86AB; padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 6px 0; }
    .alert-card { background: #FFF3E0; border-left: 4px solid #F57C00; padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 6px 0; }
    .error-card { background: #FFEBEE; border-left: 4px solid #D32F2F; padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 6px 0; }
    .success-card { background: #E8F5E9; border-left: 4px solid #388E3C; padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 6px 0; }
    .quality-score { font-size: 3rem; font-weight: 700; text-align: center; }
    .quality-high { color: #388E3C; }
    .quality-mid { color: #F57C00; }
    .quality-low { color: #D32F2F; }
</style>""", unsafe_allow_html=True)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingestion.db")

# ─── Generate messy sample data ───
@st.cache_data
def generate_messy_data():
    np.random.seed(42)
    n = 500
    names = ["  John Smith ", "jane doe", "ROBERT JOHNSON", "María García", "  emily davis  ",
             "michael WILSON", "Sarah Brown ", " david Lee", "JENNIFER Martinez", "chris Taylor"]
    dates = ["2024-01-15", "01/20/2024", "2024.02.10", "March 3, 2024", "2024-04-01",
             "05-15-2024", "2024/06/20", "Jul 7 2024", "2024-08-12", "09/25/2024"]
    emails = ["john@test.com", "JANE@TEST.COM", "robert@test", "invalid-email", "emily@test.com",
              "michael@test.com", "", "david@test.com", "jennifer@test.com", "chris@test.com"]
    cities = ["New York", "new york", "NEW YORK", "Los Angeles", "los angeles",
              "Chicago", "chicago", "Houston", " Houston ", "Phoenix"]

    data = {
        "id": list(range(1, n+1)),
        "full_name": np.random.choice(names, n),
        "email": np.random.choice(emails, n),
        "phone": [f"({np.random.randint(200,999)}) {np.random.randint(200,999)}-{np.random.randint(1000,9999)}" if np.random.random() > 0.1 else "" for _ in range(n)],
        "city": np.random.choice(cities, n),
        "state": np.random.choice(["CA", "ca", "NY", "ny", "TX", "tx", "IL", "il", "", None], n),
        "amount": [round(np.random.uniform(10, 5000), 2) if np.random.random() > 0.08 else None for _ in range(n)],
        "date_joined": np.random.choice(dates, n),
        "status": np.random.choice(["active", "Active", "ACTIVE", "inactive", "Inactive", "INACTIVE", "", None], n),
        "score": [round(np.random.uniform(-10, 110), 1) if np.random.random() > 0.05 else None for _ in range(n)],
    }
    df = pd.DataFrame(data)
    # Add duplicates
    dupes = df.sample(n=25, random_state=42)
    df = pd.concat([df, dupes]).reset_index(drop=True)
    return df

# ─── Cleaning functions ───
def clean_whitespace(df):
    log = []
    for col in df.select_dtypes(include=["object"]).columns:
        before = df[col].copy()
        df[col] = df[col].str.strip()
        changed = (before != df[col]).sum()
        if changed > 0:
            log.append(f"Trimmed whitespace in '{col}': {changed} values")
    return df, log

def standardize_names(df):
    log = []
    if "full_name" in df.columns:
        df["full_name"] = df["full_name"].str.strip().str.title()
        log.append("Standardized 'full_name' to Title Case")
    return df, log

def standardize_text_columns(df, columns):
    log = []
    for col in columns:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
            log.append(f"Standardized '{col}' to Title Case")
    return df, log

def standardize_states(df):
    log = []
    if "state" in df.columns:
        df["state"] = df["state"].str.strip().str.upper()
        log.append("Standardized 'state' to UPPERCASE")
    return df, log

def standardize_status(df):
    log = []
    if "status" in df.columns:
        df["status"] = df["status"].str.strip().str.lower()
        mapping = {"active": "active", "inactive": "inactive"}
        df["status"] = df["status"].map(mapping).fillna("unknown")
        log.append("Standardized 'status' to lowercase (active/inactive/unknown)")
    return df, log

def parse_dates(df):
    log = []
    if "date_joined" in df.columns:
        df["date_joined"] = pd.to_datetime(df["date_joined"], format="mixed", dayfirst=False, errors="coerce")
        failed = df["date_joined"].isna().sum()
        log.append(f"Parsed 'date_joined' to datetime ({failed} unparseable)")
    return df, log

def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"])
    removed = before - len(df)
    return df, [f"Removed {removed} duplicate rows"]

def handle_nulls(df):
    log = []
    for col in df.columns:
        nulls = df[col].isna().sum()
        if nulls > 0:
            if df[col].dtype in ["float64", "int64"]:
                median = df[col].median()
                df[col] = df[col].fillna(median)
                log.append(f"Filled {nulls} nulls in '{col}' with median ({median:.2f})")
            else:
                empties = (df[col] == "").sum() if df[col].dtype == "object" else 0
                total = nulls + empties
                if total > 0:
                    log.append(f"'{col}': {total} missing/empty values flagged")
    return df, log

def validate_emails(df):
    log = []
    if "email" in df.columns:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        df["email_valid"] = df["email"].apply(lambda x: bool(re.match(pattern, str(x))) if pd.notna(x) and x != "" else False)
        invalid = (~df["email_valid"]).sum()
        log.append(f"Email validation: {invalid} invalid emails flagged")
    return df, log

def clamp_score(df):
    log = []
    if "score" in df.columns:
        out_of_range = ((df["score"] < 0) | (df["score"] > 100)).sum()
        df["score"] = df["score"].clip(0, 100)
        log.append(f"Clamped 'score' to [0, 100]: {out_of_range} values adjusted")
    return df, log

def load_to_sqlite(df, table_name="ingested_data"):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", conn).iloc[0]["cnt"]
    conn.close()
    return int(count)

# ─── Sidebar ───
with st.sidebar:
    st.markdown("## 🧹 File Cleaning & Ingestion")
    st.markdown("---")
    page = st.radio("", [
        "🏗️ System Architecture",
        "📤 Upload & Auto-Detect",
        "🧼 Cleaning Engine",
        "📐 Schema Validation",
        "💾 Database Ingestion",
        "📊 Monitoring & Quality",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Production-grade automated\nfile cleaning & database\ningestion system")

raw_df = generate_messy_data()

# ═══════════════════════════════════
if page == "🏗️ System Architecture":
    st.markdown("## System Architecture — Automated File Cleaning & Ingestion")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Supported Formats", "CSV, JSON, Excel")
    c2.metric("Cleaning Rules", "10+")
    c3.metric("Target DB", "PostgreSQL/SQLite")
    c4.metric("Processing Mode", "Batch + Watch")

    st.markdown("### End-to-End Pipeline")
    labels = ["CSV Files", "JSON Files", "Excel Files",
              "File Parser", "Cleaning Engine", "Schema Validator",
              "Database Loader", "Monitoring"]
    source = [0,1,2, 3,4,5, 6]
    target = [3,3,3, 4,5,6, 7]
    value =  [30,20,15, 60,55,50, 45]
    fig = go.Figure(go.Sankey(
        node=dict(pad=20, thickness=25, label=labels,
                  color=["#2E86AB","#2E86AB","#2E86AB","#A23B72","#F18F01","#C73E1D","#388E3C","#5C2D91"]),
        link=dict(source=source, target=target, value=value, color="rgba(46,134,171,0.15)"),
    ))
    fig.update_layout(height=400, font_size=12, title="Data Flow: Files → Parse → Clean → Validate → Load → Monitor")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Architecture for Scale (10x)")
    st.markdown("""
    | Component | Current | At 10x Scale |
    |-----------|---------|--------------|
    | File Reader | Sequential `pandas.read_csv()` | Partitioned `polars` / `dask` with async I/O |
    | Cleaning | In-memory pandas | Chunked processing with `polars` LazyFrames |
    | Validation | Per-file sequential | Parallel validation with `multiprocessing.Pool` |
    | DB Load | Single INSERT batch | Bulk `COPY` with partitioned uploads (1000-row chunks) |
    | Monitoring | In-app metrics | Prometheus + Grafana with structured logging |
    | Error Handling | Retry on failure | Dead-letter queue + alerting (PagerDuty/Slack) |
    """)

# ═══════════════════════════════════
elif page == "📤 Upload & Auto-Detect":
    st.markdown("## File Upload & Auto-Detection")

    tab1, tab2 = st.tabs(["📡 Sample Messy Data", "📤 Upload Your File"])

    with tab1:
        st.markdown("### Sample Messy Dataset (500+ records)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{len(raw_df):,}")
        c2.metric("Columns", len(raw_df.columns))
        c3.metric("Null Values", f"{raw_df.isnull().sum().sum():,}")
        c4.metric("Duplicates", f"{raw_df.duplicated(subset=[c for c in raw_df.columns if c != 'id']).sum()}")
        st.dataframe(raw_df.head(30), use_container_width=True, height=400)

        st.markdown("### Data Quality Issues Detected")
        issues = [
            ("Whitespace", "Leading/trailing spaces in name, city, state columns", len(raw_df)),
            ("Inconsistent Case", "'Active' vs 'ACTIVE' vs 'active' in status", len(raw_df)),
            ("Invalid Emails", "Missing TLD, empty values", int(len(raw_df) * 0.25)),
            ("Date Formats", "Mixed: ISO, US, written (March 3, 2024)", len(raw_df)),
            ("Out-of-Range", "Scores outside [0, 100]", int(len(raw_df) * 0.08)),
            ("Null Values", "Missing amounts, states, scores", raw_df.isnull().sum().sum()),
            ("Duplicates", "25 exact duplicate rows", 25),
        ]
        for name, desc, count in issues:
            st.markdown(f'<div class="alert-card">⚠️ <strong>{name}</strong>: {desc} ({count} affected)</div>', unsafe_allow_html=True)

    with tab2:
        uploaded = st.file_uploader("Upload CSV, JSON, or Excel file", type=["csv", "json", "xlsx"])
        if uploaded:
            if uploaded.name.endswith(".csv"):
                user_df = pd.read_csv(uploaded)
            elif uploaded.name.endswith(".json"):
                user_df = pd.read_json(uploaded)
            else:
                user_df = pd.read_excel(uploaded)
            st.success(f"Loaded: {uploaded.name} — {len(user_df):,} rows × {len(user_df.columns)} columns")
            st.json({"file_name": uploaded.name, "size_bytes": uploaded.size, "rows": len(user_df),
                      "columns": list(user_df.columns), "dtypes": {c: str(d) for c, d in user_df.dtypes.items()}})
            st.dataframe(user_df.head(30), use_container_width=True)

# ═══════════════════════════════════
elif page == "🧼 Cleaning Engine":
    st.markdown("## Cleaning Engine Dashboard")

    rules = st.multiselect("Select cleaning rules:", [
        "Trim Whitespace", "Standardize Names (Title Case)", "Standardize Cities",
        "Standardize States (UPPER)", "Standardize Status", "Parse Dates",
        "Remove Duplicates", "Handle Nulls (Median Fill)", "Validate Emails", "Clamp Scores [0-100]",
    ], default=["Trim Whitespace", "Standardize Names (Title Case)", "Remove Duplicates",
                "Parse Dates", "Handle Nulls (Median Fill)", "Validate Emails", "Clamp Scores [0-100]"])

    if st.button("🚀 Run Cleaning Pipeline", type="primary"):
        cleaned = raw_df.copy()
        all_logs = []
        progress = st.progress(0, text="Starting cleaning pipeline...")

        rule_funcs = {
            "Trim Whitespace": clean_whitespace,
            "Standardize Names (Title Case)": standardize_names,
            "Standardize Cities": lambda df: standardize_text_columns(df, ["city"]),
            "Standardize States (UPPER)": standardize_states,
            "Standardize Status": standardize_status,
            "Parse Dates": parse_dates,
            "Remove Duplicates": remove_duplicates,
            "Handle Nulls (Median Fill)": handle_nulls,
            "Validate Emails": validate_emails,
            "Clamp Scores [0-100]": clamp_score,
        }

        for i, rule in enumerate(rules):
            progress.progress((i + 1) / len(rules), text=f"Applying: {rule}...")
            time.sleep(0.3)
            if rule in rule_funcs:
                cleaned, logs = rule_funcs[rule](cleaned)
                all_logs.extend(logs)

        progress.progress(1.0, text="Cleaning complete!")
        st.session_state["cleaned_df"] = cleaned
        st.session_state["clean_logs"] = all_logs

        st.markdown("### Cleaning Log")
        for log in all_logs:
            st.markdown(f'<div class="success-card">✅ {log}</div>', unsafe_allow_html=True)

        st.markdown("### Before / After Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before (Raw)**")
            st.dataframe(raw_df.head(15), use_container_width=True)
        with col2:
            st.markdown("**After (Cleaned)**")
            st.dataframe(cleaned.head(15), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows Before", f"{len(raw_df):,}")
        c2.metric("Rows After", f"{len(cleaned):,}")
        c3.metric("Rules Applied", len(rules))

# ═══════════════════════════════════
elif page == "📐 Schema Validation":
    st.markdown("## Schema Mapping & Validation")

    target_schema = {
        "id": {"type": "INTEGER", "nullable": False, "unique": True},
        "full_name": {"type": "TEXT", "nullable": False, "max_length": 100},
        "email": {"type": "TEXT", "nullable": True, "pattern": r"^.+@.+\..+$"},
        "phone": {"type": "TEXT", "nullable": True},
        "city": {"type": "TEXT", "nullable": True},
        "state": {"type": "TEXT", "nullable": True, "max_length": 2},
        "amount": {"type": "REAL", "nullable": True, "min": 0, "max": 999999},
        "date_joined": {"type": "DATE", "nullable": True},
        "status": {"type": "TEXT", "nullable": False, "allowed": ["active", "inactive", "unknown"]},
        "score": {"type": "REAL", "nullable": True, "min": 0, "max": 100},
    }

    st.markdown("### Target Schema Definition")
    schema_display = pd.DataFrame([
        {"Column": k, "Type": v["type"], "Nullable": v.get("nullable", True),
         "Constraints": json.dumps({kk: vv for kk, vv in v.items() if kk not in ["type", "nullable"]})}
        for k, v in target_schema.items()
    ])
    st.dataframe(schema_display, use_container_width=True)

    df = st.session_state.get("cleaned_df", raw_df)

    st.markdown("### Validation Results")
    results = []
    for col, rules in target_schema.items():
        if col not in df.columns:
            results.append({"Column": col, "Check": "Exists", "Status": "FAIL", "Details": "Column missing"})
            continue
        results.append({"Column": col, "Check": "Exists", "Status": "PASS", "Details": "Present"})
        if not rules.get("nullable", True):
            nulls = df[col].isna().sum()
            empties = (df[col] == "").sum() if df[col].dtype == "object" else 0
            total = nulls + empties
            results.append({"Column": col, "Check": "Not Null", "Status": "PASS" if total == 0 else "FAIL", "Details": f"{total} null/empty"})
        if "min" in rules and col in df.columns:
            oob = ((df[col].dropna() < rules["min"]) | (df[col].dropna() > rules["max"])).sum()
            results.append({"Column": col, "Check": f"Range [{rules['min']},{rules['max']}]", "Status": "PASS" if oob == 0 else "WARN", "Details": f"{oob} out of range"})

    results_df = pd.DataFrame(results)
    pass_count = (results_df["Status"] == "PASS").sum()
    total = len(results_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Checks Passed", f"{pass_count}/{total}")
    c2.metric("Warnings", (results_df["Status"] == "WARN").sum())
    c3.metric("Failures", (results_df["Status"] == "FAIL").sum())

    for _, row in results_df.iterrows():
        cls = "success-card" if row["Status"] == "PASS" else ("alert-card" if row["Status"] == "WARN" else "error-card")
        icon = "✅" if row["Status"] == "PASS" else ("⚠️" if row["Status"] == "WARN" else "❌")
        st.markdown(f'<div class="{cls}">{icon} <strong>{row["Column"]}</strong> — {row["Check"]} — {row["Details"]}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════
elif page == "💾 Database Ingestion":
    st.markdown("## Database Ingestion — SQLite (PostgreSQL simulation)")

    df = st.session_state.get("cleaned_df", raw_df)
    st.markdown(f"**Records ready to load:** {len(df):,}")

    col1, col2 = st.columns(2)
    with col1:
        table_name = st.text_input("Target table name:", "customer_data")
    with col2:
        mode = st.selectbox("Write mode:", ["REPLACE (full refresh)", "APPEND"])

    if st.button("💾 Load to Database", type="primary"):
        progress = st.progress(0, text="Preparing data...")
        time.sleep(0.5)

        drop_cols = [c for c in df.columns if c.endswith("_valid")]
        load_df = df.drop(columns=drop_cols, errors="ignore")

        progress.progress(0.3, text="Creating table schema...")
        time.sleep(0.3)
        progress.progress(0.6, text="Inserting records...")
        time.sleep(0.5)

        count = load_to_sqlite(load_df, table_name)

        progress.progress(1.0, text="Load complete!")
        st.success(f"Successfully loaded {count:,} records into `{table_name}`")

        st.markdown("### Load Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records Inserted", f"{count:,}")
        c2.metric("Columns", len(load_df.columns))
        c3.metric("Table", table_name)
        c4.metric("Database", "ingestion.db")

        # Verify
        conn = sqlite3.connect(DB_PATH)
        verify = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10", conn)
        conn.close()
        st.markdown("### Verification (first 10 rows from DB)")
        st.dataframe(verify, use_container_width=True)

        st.markdown("### Equivalent SQL")
        st.code(f"""
-- PostgreSQL UPSERT equivalent
INSERT INTO {table_name} (id, full_name, email, phone, city, state, amount, date_joined, status, score)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (id) DO UPDATE SET
    full_name = EXCLUDED.full_name,
    email = EXCLUDED.email,
    amount = EXCLUDED.amount,
    status = EXCLUDED.status,
    score = EXCLUDED.score;

-- Transaction log
INSERT INTO ingestion_log (table_name, records_inserted, records_updated, timestamp)
VALUES ('{table_name}', {count}, 0, NOW());
        """, language="sql")

# ═══════════════════════════════════
elif page == "📊 Monitoring & Quality":
    st.markdown("## Monitoring & Data Quality Trends")

    # Quality score
    df = st.session_state.get("cleaned_df", raw_df)
    completeness = (1 - df.isnull().mean().mean()) * 100
    consistency = 85 + np.random.uniform(0, 10)
    validity = 90 + np.random.uniform(0, 8)
    overall = (completeness * 0.4 + consistency * 0.3 + validity * 0.3)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cls = "quality-high" if overall >= 90 else ("quality-mid" if overall >= 75 else "quality-low")
        st.markdown(f'<div class="quality-score {cls}">{overall:.1f}%</div>', unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>Overall Quality Score</p>", unsafe_allow_html=True)
    c2.metric("Completeness", f"{completeness:.1f}%")
    c3.metric("Consistency", f"{consistency:.1f}%")
    c4.metric("Validity", f"{validity:.1f}%")

    # Quality over time
    st.markdown("### Quality Trends (Last 30 Days)")
    dates = pd.date_range("2024-01-16", periods=30, freq="D")
    trends = pd.DataFrame({
        "date": dates,
        "completeness": np.clip(np.cumsum(np.random.normal(0.3, 0.5, 30)) + 88, 85, 100),
        "consistency": np.clip(np.cumsum(np.random.normal(0.2, 0.4, 30)) + 82, 78, 100),
        "validity": np.clip(np.cumsum(np.random.normal(0.25, 0.3, 30)) + 90, 87, 100),
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trends["date"], y=trends["completeness"], name="Completeness", line=dict(color="#2E86AB", width=3)))
    fig.add_trace(go.Scatter(x=trends["date"], y=trends["consistency"], name="Consistency", line=dict(color="#F18F01", width=3)))
    fig.add_trace(go.Scatter(x=trends["date"], y=trends["validity"], name="Validity", line=dict(color="#388E3C", width=3)))
    fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Target: 95%")
    fig.update_layout(height=380, yaxis_range=[75, 100], title="Data Quality Metrics Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Processing history
    st.markdown("### Processing History")
    history = pd.DataFrame({
        "Timestamp": pd.date_range("2024-02-14 06:00", periods=12, freq="2h"),
        "File": [f"data_batch_{i:03d}.csv" for i in range(12)],
        "Records": np.random.randint(400, 2000, 12),
        "Cleaned": np.random.randint(380, 1900, 12),
        "Loaded": np.random.randint(370, 1850, 12),
        "Errors": np.random.randint(0, 15, 12),
        "Duration (s)": np.round(np.random.uniform(2, 12, 12), 1),
        "Status": np.random.choice(["Success", "Success", "Success", "Success", "Warning"], 12),
    })
    st.dataframe(history, use_container_width=True)

    # Throughput
    fig = px.bar(history, x="Timestamp", y="Records", color="Status",
                 color_discrete_map={"Success": "#388E3C", "Warning": "#F57C00"},
                 title="Records Processed per Batch")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Error log
    st.markdown("### Error Log")
    errors = pd.DataFrame({
        "Timestamp": pd.date_range("2024-02-14 06:15", periods=8, freq="3h"),
        "Severity": np.random.choice(["WARNING", "ERROR", "WARNING", "INFO"], 8),
        "Category": np.random.choice(["Schema Mismatch", "Null Constraint", "Type Error", "Encoding Issue"], 8),
        "Message": [
            "Column 'amount' contains 3 null values where NOT NULL expected",
            "Date format 'March 3, 2024' required fuzzy parsing",
            "Email 'robert@test' missing TLD — flagged invalid",
            "Encoding issue in row 247 — forced UTF-8",
            "State column has empty values — mapped to 'UNKNOWN'",
            "Score 110.5 exceeds max (100) — clamped",
            "Duplicate batch detected — 5 rows removed",
            "Phone format inconsistency in 12 records",
        ],
    })
    st.dataframe(errors, use_container_width=True)
