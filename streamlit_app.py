# streamlit_app.py
import os
import io
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Try optional deps
try:
    import xgboost as xgb  # noqa

    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import tensorflow as tf  # noqa

    TF_OK = True
except Exception as e:
    TF_OK = False
    print(f"TensorFlow not available: {e}")

# Import your classes from the module file you created
from healthcare_pipeline import (
    HealthcareDataProcessor,
    FeatureEngineer,
    ReadmissionPredictor,
    ClinicalInsightsGenerator,
    DiseaseProgressionTracker,
    ModelVisualizer,
    ReportGenerator,
)

# Page configuration
st.set_page_config(
    page_title="RWEsearch – Healthcare Analytics", page_icon="🏥", layout="wide"
)

# Main title
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto:wght@400;500;700&display=swap');
  :root {
    --accent: #14b8a6; /* teal-500 */
    --accent-600: #0d9488;
    --text: #0f172a; /* slate-900 */
    --muted: #475569; /* slate-600 */
    --card-bg: #ffffff;
    --card-border: #e2e8f0; /* slate-200 */
  }
  html, body, [class*="css"] { font-family: 'Inter', 'Roboto', system-ui, -apple-system, Segoe UI, sans-serif; }
  .app-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.25rem 0 0.75rem 0; border-bottom: 1px solid var(--card-border);
    margin-bottom: 0.75rem;
  }
  .brand {
    display:flex; gap:.75rem; align-items:center; color: var(--text);
  }
  .brand .logo {
    width:40px; height:40px; display:grid; place-items:center; color:white;
    background: linear-gradient(135deg, var(--accent), #22c55e);
    border-radius: 12px; font-weight:700;
  }
  .brand h1 { font-size: 1.25rem; margin: 0; letter-spacing: .2px; }
  .brand p { margin: 0; color: var(--muted); font-size: .9rem; }
  .pill {
    background: rgba(20,184,166,.08); color: var(--accent-600);
    padding: .35rem .6rem; border-radius: 999px; font-size: .8rem; border:1px solid rgba(20,184,166,.25);
  }
  .card { background: var(--card-bg); border:1px solid var(--card-border); border-radius: 14px; padding: 1rem; box-shadow: 0 1px 3px rgba(15,23,42,.04);
          transition: box-shadow .2s ease, transform .2s ease; }
  .card:hover { box-shadow: 0 6px 16px rgba(2,6,23,.08); }
  .card h3 { margin: 0 0 .5rem 0; font-size: 1rem; color: var(--muted); font-weight:600; }
  .metric { font-size: 1.4rem; font-weight:700; color: var(--text); }
  .subtle { color: var(--muted); font-size:.9rem; }
  .section-title { margin: .25rem 0 .5rem 0; font-size: 1.05rem; color: var(--text); }
  .divider { height:1px; background: var(--card-border); margin: .5rem 0 1rem 0; }
</style>
<div class="app-header">
  <div class="brand">
    <div class="logo">RWE</div>
    <div>
      <h1>RWEsearch Hackathon – Healthcare Analytics</h1>
      <p>Minimal, professional dashboard for real‑world evidence exploration</p>
    </div>
  </div>
  <span class="pill">use local files option for better perfomance </span>
</div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("⚙️ Setup & Configuration")
st.sidebar.markdown("---")
st.sidebar.caption(" Upload files or provide local paths")

use_uploads = st.sidebar.radio(
    "Data Source", [" Upload CSVs", " Use local file paths"], index=1
)

# Default file names (DE-SynPUF)
default_files = {
    "beneficiaries_2008": "data/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv",
    "beneficiaries_2009": "data/DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv",
    "beneficiaries_2010": "data/DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv",
    "inpatient": "data/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv",
    "outpatient": "data/DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv",
}

uploads = {}
paths = {}

if use_uploads == " Upload CSVs":
    st.sidebar.markdown(
        """
    <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px;border: 1px solid #e2e8f0;">
        <h4 style="color: black; margin: 0 0 0.5rem 0;"> Upload CSV Files</h4>
    </div>
    """,
        unsafe_allow_html=True,
    )

    uploads["beneficiaries_2008"] = st.sidebar.file_uploader(
        " 2008 Beneficiary Summary", type=["csv"]
    )
    uploads["beneficiaries_2009"] = st.sidebar.file_uploader(
        "2009 Beneficiary Summary", type=["csv"]
    )
    uploads["beneficiaries_2010"] = st.sidebar.file_uploader(
        " 2010 Beneficiary Summary", type=["csv"]
    )
    uploads["inpatient"] = st.sidebar.file_uploader(
        "Inpatient Claims (2008–2010)", type=["csv"]
    )
    uploads["outpatient"] = st.sidebar.file_uploader(
        " Outpatient Claims (2008–2010)", type=["csv"]
    )
else:
    st.sidebar.markdown(
        """
    <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: black; margin: 0 0 0.5rem 0;"> Local File Paths</h4>
    </div>
    """,
        unsafe_allow_html=True,
    )

    for key, default in default_files.items():
        display_name = key.replace("_", " ").title()
        if "beneficiaries" in key:
            display_name = f" {display_name}"
        elif "inpatient" in key:
            display_name = f" {display_name}"
        elif "outpatient" in key:
            display_name = f" {display_name}"

        paths[key] = st.sidebar.text_input(display_name, value=default)

st.sidebar.markdown("---")
st.sidebar.subheader(" Model Configuration")

target_choice = st.sidebar.selectbox(
    " Target Horizon", ["READMIT_30", "READMIT_60", "READMIT_90"], index=0
)
run_dl = st.sidebar.checkbox(
    " Train Deep Learning (TF)",
    value=False,
    help="Requires TensorFlow; auto-skipped if unavailable.",
)
seed = st.sidebar.number_input(" Random Seed", 0, 999999, 42)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
<div style="text-align: center; color: #475569; font-size: 0.8rem;">
   Healthcare Analytics Platform<br>Built for Hackathon Demos
</div>
""",
    unsafe_allow_html=True,
)


# ---------- Utilities for cleaned claims/beneficiaries ----------
def get_clean_claims(processor: "HealthcareDataProcessor"):
    ip = None
    op = None
    if processor.inpatient_data is not None:
        ip = processor.inpatient_data.copy()
        if "CLM_ADMSN_DT" in ip.columns:
            ip["ADMIT_DATE"] = pd.to_datetime(
                ip["CLM_ADMSN_DT"], format="%Y%m%d", errors="coerce"
            )
        if "CLM_THRU_DT" in ip.columns:
            ip["DISCHARGE_DATE"] = pd.to_datetime(
                ip["CLM_THRU_DT"], format="%Y%m%d", errors="coerce"
            )
        if "CLM_PMT_AMT" in ip.columns:
            ip["CLM_PMT_AMT"] = pd.to_numeric(ip["CLM_PMT_AMT"], errors="coerce")
        ip["CLAIM_TYPE"] = "Inpatient"
    if processor.outpatient_data is not None:
        op = processor.outpatient_data.copy()
        if "CLM_FROM_DT" in op.columns:
            op["ADMIT_DATE"] = pd.to_datetime(
                op["CLM_FROM_DT"], format="%Y%m%d", errors="coerce"
            )
        if "CLM_THRU_DT" in op.columns:
            op["DISCHARGE_DATE"] = pd.to_datetime(
                op["CLM_THRU_DT"], format="%Y%m%d", errors="coerce"
            )
        if "CLM_PMT_AMT" in op.columns:
            op["CLM_PMT_AMT"] = pd.to_numeric(op["CLM_PMT_AMT"], errors="coerce")
        op["CLAIM_TYPE"] = "Outpatient"
    if ip is None and op is None:
        return None
    if ip is None:
        return op
    if op is None:
        return ip
    return pd.concat([ip, op], ignore_index=True)


def diagnosis_columns(df: pd.DataFrame):
    return [c for c in df.columns if "ICD9_DGNS_CD" in c or "ICD9_PRCDR_CD" in c]


# ------------- Helper: Load Data Either From Uploads or Paths -------------
def _read_csv_streamlit(file_or_path: str | io.BytesIO) -> pd.DataFrame:
    if isinstance(file_or_path, str):
        return pd.read_csv(file_or_path)
    else:
        return pd.read_csv(file_or_path)


def _write_temp_upload(uploaded_file, name_hint):
    # Save uploaded file to a temp path so your pipeline can read it by filename
    if uploaded_file is None:
        return None
    temp_path = os.path.join(
        st.session_state.get("_tmpdir", "."), f"_tmp_{name_hint}.csv"
    )
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_path


# ------------- Cache & Pipeline Assembly -------------
@st.cache_resource(show_spinner=False)
def build_objects(file_mode: str, uploads, paths_dict, seed=42):
    np.random.seed(seed)

    # Create processor and point it to actual filenames on disk
    processor = HealthcareDataProcessor()

    # We’ll write uploaded files to temp and then call your existing loader
    file_map = {}

    if file_mode == "Upload CSVs":
        # Create a temp dir per session
        if "_tmpdir" not in st.session_state:
            st.session_state["_tmpdir"] = os.path.join(os.getcwd(), ".st_tmp")
            os.makedirs(st.session_state["_tmpdir"], exist_ok=True)

        # Beneficiary files
        b08 = _write_temp_upload(uploads["beneficiaries_2008"], "bene2008")
        b09 = _write_temp_upload(uploads["beneficiaries_2009"], "bene2009")
        b10 = _write_temp_upload(uploads["beneficiaries_2010"], "bene2010")
        ip = _write_temp_upload(uploads["inpatient"], "inpatient")
        op = _write_temp_upload(uploads["outpatient"], "outpatient")

        file_map = {2008: b08, 2009: b09, 2010: b10}

        # Override processor load to use these explicit files
        def load_beneficiary_override(years=[2008, 2009, 2010]):
            dfs = []
            for y in years:
                fname = file_map.get(y)
                if not fname:
                    continue
                df = pd.read_csv(fname)
                df["YEAR"] = y
                dfs.append(df)
            if dfs:
                processor.beneficiary_data = pd.concat(dfs, ignore_index=True)
            return processor.beneficiary_data

        def load_claims_override():
            if ip and os.path.exists(ip):
                processor.inpatient_data = pd.read_csv(ip)
            if op and os.path.exists(op):
                processor.outpatient_data = pd.read_csv(op)

        processor.load_beneficiary_data = load_beneficiary_override
        processor.load_claims_data = load_claims_override

    else:
        # Local paths mode: redefine loader to use provided paths
        file_map = {
            2008: paths_dict["beneficiaries_2008"],
            2009: paths_dict["beneficiaries_2009"],
            2010: paths_dict["beneficiaries_2010"],
        }

        def load_beneficiary_override(years=[2008, 2009, 2010]):
            dfs = []
            for y in years:
                fname = file_map.get(y)
                if not fname or not os.path.exists(fname):
                    continue
                df = pd.read_csv(fname)
                df["YEAR"] = y
                dfs.append(df)
            if dfs:
                processor.beneficiary_data = pd.concat(dfs, ignore_index=True)
            return processor.beneficiary_data

        def load_claims_override():
            ip = paths_dict["inpatient"]
            op = paths_dict["outpatient"]
            if ip and os.path.exists(ip):
                processor.inpatient_data = pd.read_csv(ip)
            if op and os.path.exists(op):
                processor.outpatient_data = pd.read_csv(op)

        processor.load_beneficiary_data = load_beneficiary_override
        processor.load_claims_data = load_claims_override

    # Load
    processor.load_beneficiary_data()
    processor.load_claims_data()
    if processor.inpatient_data is not None:
        processor.create_readmission_labels()

    # Feature engineering
    engineer = FeatureEngineer(processor)
    engineer.create_patient_features()
    engineer.create_diagnosis_features()

    # Predictor + Insights + Visuals + Tracker + Reporter
    predictor = ReadmissionPredictor(engineer)
    insights = ClinicalInsightsGenerator(predictor)
    visualizer = ModelVisualizer(predictor)
    tracker = DiseaseProgressionTracker(processor)
    reporter = ReportGenerator(processor, predictor, insights, visualizer)

    return processor, engineer, predictor, insights, visualizer, tracker, reporter


tabs = st.tabs(
    [
        "RWE Dashboard",
        "Upload & Prepare",
        "Data Overview",
        "Train Models",
        "Model Performance",
        "Feature Importance",
        "Costs & Recommendations",
        "Disease Progression",
        "Patient Explorer",
        "Model Management",
        "Batch Predictions",
        "Executive Report",
    ]
)

with tabs[0]:
    st.subheader("RWE Dashboard")
    if "processor" in st.session_state:
        processor = st.session_state["processor"]
        engineer = (
            st.session_state["engineer"] if "engineer" in st.session_state else None
        )

        claims = get_clean_claims(processor)
        bene = processor.beneficiary_data

        if claims is None or bene is None:
            st.info("Please build the dataset in Upload & Prepare.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    '<div class="card"><h3>Unique Patients</h3><div class="metric">{:,}</div></div>'.format(
                        bene["DESYNPUF_ID"].nunique()
                    ),
                    unsafe_allow_html=True,
                )
            with c2:
                total_amt = claims["CLM_PMT_AMT"].fillna(0).sum()
                st.markdown(
                    '<div class="card"><h3>Total Reimbursed</h3><div class="metric">${:,.0f}</div></div>'.format(
                        total_amt
                    ),
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    '<div class="card"><h3>Inpatient Claims</h3><div class="metric">{:,}</div></div>'.format(
                        (claims["CLAIM_TYPE"] == "Inpatient").sum()
                    ),
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    '<div class="card"><h3>Outpatient Claims</h3><div class="metric">{:,}</div></div>'.format(
                        (claims["CLAIM_TYPE"] == "Outpatient").sum()
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # Filters
            st.markdown(
                "<div class='section-title'>Filters</div>", unsafe_allow_html=True
            )
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                year_opts = (
                    sorted(bene["YEAR"].dropna().unique().tolist())
                    if "YEAR" in bene.columns
                    else []
                )
                sel_years = st.multiselect("Year", year_opts, default=year_opts)
            with f2:
                claim_type = st.multiselect(
                    "Claim Type",
                    ["Inpatient", "Outpatient"],
                    default=["Inpatient", "Outpatient"],
                )
            with f3:
                min_age = int(bene["AGE"].min()) if "AGE" in bene.columns else 0
                max_age = int(bene["AGE"].max()) if "AGE" in bene.columns else 120
                age_range = (
                    st.slider("Age", min_age, max_age, (min_age, max_age))
                    if "AGE" in bene.columns
                    else (0, 120)
                )
            with f4:
                diag_query = st.text_input("Search Diagnosis (ICD9)")

            # Apply filters
            merged = (
                claims.merge(
                    (
                        bene[["DESYNPUF_ID", "AGE", "YEAR"]]
                        if "AGE" in bene.columns
                        else bene[["DESYNPUF_ID", "YEAR"]]
                    ),
                    on="DESYNPUF_ID",
                    how="left",
                )
                if "DESYNPUF_ID" in claims.columns
                else claims.copy()
            )
            if sel_years:
                merged = (
                    merged[merged["YEAR"].isin(sel_years)]
                    if "YEAR" in merged.columns
                    else merged
                )
            if claim_type:
                merged = merged[merged["CLAIM_TYPE"].isin(claim_type)]
            if "AGE" in merged.columns:
                merged = merged[
                    (merged["AGE"] >= age_range[0]) & (merged["AGE"] <= age_range[1])
                ]

            if diag_query:
                dcols = [c for c in diagnosis_columns(merged)]
                if dcols:
                    mask = (
                        merged[dcols]
                        .astype(str)
                        .apply(
                            lambda r: any(diag_query in str(v) for v in r.values),
                            axis=1,
                        )
                    )
                    merged = merged[mask]

            # Charts row: Reimbursements over time and by type
            left, right = st.columns((2, 1))
            with left:
                if "ADMIT_DATE" in merged.columns and "CLM_PMT_AMT" in merged.columns:
                    ts = merged.dropna(subset=["ADMIT_DATE"]).copy()
                    ts["month"] = ts["ADMIT_DATE"].dt.to_period("M").dt.to_timestamp()
                    agg = (
                        ts.groupby(["month", "CLAIM_TYPE"])["CLM_PMT_AMT"]
                        .sum()
                        .reset_index()
                    )
                    line = (
                        alt.Chart(agg)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("month:T", title="Month"),
                            y=alt.Y("CLM_PMT_AMT:Q", title="Total Reimbursed"),
                            color="CLAIM_TYPE:N",
                            tooltip=["month:T", "CLAIM_TYPE:N", "CLM_PMT_AMT:Q"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(line, use_container_width=True)
            with right:
                if "CLAIM_TYPE" in merged.columns and "CLM_PMT_AMT" in merged.columns:
                    by_type = (
                        merged.groupby("CLAIM_TYPE")["CLM_PMT_AMT"].sum().reset_index()
                    )
                    pie = (
                        alt.Chart(by_type)
                        .mark_arc(innerRadius=50)
                        .encode(
                            theta="CLM_PMT_AMT:Q",
                            color="CLAIM_TYPE:N",
                            tooltip=["CLAIM_TYPE", "CLM_PMT_AMT"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(pie, use_container_width=True)

            # Demographics and chronic conditions
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                if "BENE_SEX_IDENT_CD" in bene.columns:
                    gdf = (
                        bene.groupby("BENE_SEX_IDENT_CD")
                        .size()
                        .reset_index(name="count")
                    )
                    bar = (
                        alt.Chart(gdf)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                        .encode(
                            x=alt.X("BENE_SEX_IDENT_CD:N", title="Sex"),
                            y=alt.Y("count:Q", title="Patients"),
                            tooltip=["BENE_SEX_IDENT_CD", "count"],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(bar, use_container_width=True)
            with dcol2:
                cond_cols = [c for c in bene.columns if c.startswith("SP_")]
                if cond_cols:
                    counts = (
                        bene[cond_cols]
                        .apply(lambda s: (s == 1).sum())
                        .sort_values(ascending=False)
                        .head(10)
                    )
                    cdf = counts.reset_index().rename(
                        columns={"index": "Condition", 0: "count"}
                    )
                    cdf["Condition"] = cdf["Condition"].str.replace(
                        "SP_", "", regex=False
                    )
                    barc = (
                        alt.Chart(cdf)
                        .mark_bar(
                            cornerRadiusTopLeft=4,
                            cornerRadiusTopRight=4,
                            color="#14b8a6",
                        )
                        .encode(
                            x=alt.X("count:Q", title="Patients with Condition"),
                            y=alt.Y("Condition:N", sort="-x"),
                            tooltip=["Condition", "count"],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(barc, use_container_width=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # Interactive table
            st.markdown(
                "<div class='section-title'>Filtered Claims</div>",
                unsafe_allow_html=True,
            )
            display_cols = [
                c
                for c in [
                    "DESYNPUF_ID",
                    "CLAIM_TYPE",
                    "ADMIT_DATE",
                    "DISCHARGE_DATE",
                    "CLM_PMT_AMT",
                    "CLM_UTLZTN_DAY_CNT",
                ]
                if c in merged.columns
            ]
            st.dataframe(
                merged[display_cols]
                .sort_values(
                    by=(
                        display_cols[2]
                        if "ADMIT_DATE" in display_cols
                        else display_cols[0]
                    )
                )
                .head(1000),
                use_container_width=True,
            )
    else:
        st.info("Use Upload & Prepare to build the dataset.")

# Upload & Prepare
with tabs[1]:
    st.subheader(" Upload & Prepare")
    st.write(
        "Upload CSV files or provide local file paths, then build your dataset for analysis"
    )

    # File validation
    files_ready = False
    paths_ready = False

    if use_uploads == " Upload CSVs":
        st.write("** Upload CSV Files**")

        # Check if files are uploaded
        files_ready = all(
            uploads.get(key) is not None
            for key in [
                "beneficiaries_2008",
                "beneficiaries_2009",
                "beneficiaries_2010",
                "inpatient",
                "outpatient",
            ]
        )

        if files_ready:
            st.success(" All Required Files Uploaded Successfully!")

            # Show file info
            st.subheader("📋 Uploaded Files Summary")
            col1, col2 = st.columns(2)

            with col1:
                for i, (key, file) in enumerate(uploads.items()):
                    if file and i % 2 == 0:
                        st.write(f"**{key.replace('_', ' ').title()}**: {file.name}")

            with col2:
                for i, (key, file) in enumerate(uploads.items()):
                    if file and i % 2 == 1:
                        st.write(f"**{key.replace('_', ' ').title()}**: {file.name}")
        else:
            st.warning("⚠️ Please Upload All Required CSV Files")
            st.write(
                "Required files: Beneficiary summaries (2008-2010), Inpatient claims, Outpatient claims"
            )

            # Show what's missing
            missing_files = []
            for key in [
                "beneficiaries_2008",
                "beneficiaries_2009",
                "beneficiaries_2010",
                "inpatient",
                "outpatient",
            ]:
                if uploads.get(key) is None:
                    missing_files.append(key.replace("_", " ").title())

            if missing_files:
                st.write("**Missing files:**")
                for file in missing_files:
                    st.write(f"❌ {file}")
    else:
        st.write("** Local File Paths**")

        # Check if local paths are provided
        paths_ready = all(
            paths.get(key) and paths[key].strip()
            for key in [
                "beneficiaries_2008",
                "beneficiaries_2009",
                "beneficiaries_2010",
                "inpatient",
                "outpatient",
            ]
        )

        if paths_ready:
            st.success(" All Required File Paths Provided!")

            # Show file paths
            st.subheader(" Local File Paths")
            col1, col2 = st.columns(2)

            with col1:
                for i, (key, path) in enumerate(paths.items()):
                    if path and path.strip() and i % 2 == 0:
                        st.write(f"**{key.replace('_', ' ').title()}**: {path}")

            with col2:
                for i, (key, path) in enumerate(paths.items()):
                    if path and path.strip() and i % 2 == 1:
                        st.write(f"**{key.replace('_', ' ').title()}**: {path}")
        else:
            st.warning(" Please Provide All Required File Paths")
            st.write(
                "Required files: Beneficiary summaries (2008-2010), Inpatient claims, Outpatient claims"
            )

            # Show what's missing
            missing_paths = []
            for key in [
                "beneficiaries_2008",
                "beneficiaries_2009",
                "beneficiaries_2010",
                "inpatient",
                "outpatient",
            ]:
                if not paths.get(key) or not paths[key].strip():
                    missing_paths.append(key.replace("_", " ").title())

            if missing_paths:
                st.write("**Missing file paths:**")
                for path in missing_paths:
                    st.write(f"❌ {path}")

    # Build dataset button
    if (use_uploads == " Upload CSVs" and files_ready) or (
        use_uploads == " Use local file paths" and paths_ready
    ):
        st.success(" Ready to Build Your Dataset!")
        st.write(
            "All required files are ready. Click the button below to process your data and prepare for machine learning analysis."
        )

        if st.button("Build Dataset", type="primary"):
            with st.spinner(" Building dataset..."):
                try:
                    (
                        processor,
                        engineer,
                        predictor,
                        insights,
                        visualizer,
                        tracker,
                        reporter,
                    ) = build_objects(use_uploads, uploads, paths, seed)
                    st.session_state["processor"] = processor
                    st.session_state["engineer"] = engineer
                    st.session_state["predictor"] = predictor
                    st.session_state["insights"] = insights
                    st.session_state["visualizer"] = visualizer
                    st.session_state["tracker"] = tracker
                    st.session_state["reporter"] = reporter

                    st.success(" Dataset Built Successfully!")
                    st.write(
                        "Your healthcare data is now ready for analysis. Navigate to the Train Models tab to start building predictive models."
                    )

                    # Show next steps
                    st.subheader(" Next Steps:")
                    st.write("1. Go to Train Models tab")
                    st.write(
                        "2. Select your target variable (30/60/90-day readmission)"
                    )
                    st.write("3. Train machine learning models")
                    st.write("4. Analyze model performance and insights")

                except Exception as e:
                    st.error(f"❌ Error Building Dataset: {str(e)}")

                    if use_uploads == " Upload CSVs":
                        st.info("💡 Please check your uploaded files and try again.")
                    else:
                        st.info("💡 Please check your file paths and try again.")
    else:
        st.markdown(
            """
        <div style="text-align: center; margin: 2rem 0;">
            <div style="background: rgba(255, 255, 255, 0.1); padding: 2rem; border-radius: 20px; border: 2px dashed #667eea;">
                <h3 style="margin: 0; color: #667eea;"> Files Not Ready</h3>
                <p style="margin: 0.5rem 0 0 0; color: #7f8c8d;">Please upload all required CSV files or provide valid file paths to continue.</p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.info(
        "Tip: If TensorFlow/XGBoost aren’t installed, the app will still run using classical models."
    )

# 2) Data Overview
with tabs[2]:
    st.subheader("Data Overview")
    if "processor" in st.session_state:
        processor = st.session_state["processor"]
        engineer = st.session_state["engineer"]
    else:
        st.info("Please run **Build Dataset** first in the previous tab.")

    if "processor" in st.session_state and processor.beneficiary_data is not None:
        # Basic metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Unique Patients",
                value=int(processor.beneficiary_data["DESYNPUF_ID"].nunique()),
            )
        with c2:
            st.metric(
                "Inpatient Admissions",
                value=int(
                    len(processor.inpatient_data)
                    if processor.inpatient_data is not None
                    else 0
                ),
            )
        with c3:
            st.metric(
                "Outpatient Claims",
                value=int(
                    len(processor.outpatient_data)
                    if processor.outpatient_data is not None
                    else 0
                ),
            )

        # Readmission counts
        st.subheader(" Readmission Statistics")
        if "predictor" in st.session_state and st.session_state["predictor"].results:
            predictor = st.session_state["predictor"]

            # Calculate readmission counts for each period
            readmission_counts = {}
            for period in ["READMIT_30", "READMIT_60", "READMIT_90"]:
                if period in predictor.results:
                    # Check if results contain actual model metrics or just flags
                    if (
                        isinstance(predictor.results[period], dict)
                        and "_models_exist" in predictor.results[period]
                    ):
                        continue  # Skip if only flags are present

                    # Get the best performing model for each period
                    best_model = None
                    best_auc = 0
                    for model_name, results in predictor.results[period].items():
                        if (
                            isinstance(results, dict)
                            and "auc" in results
                            and results["auc"] > best_auc
                        ):
                            best_auc = results["auc"]
                            best_model = results

                    if best_model and "y_pred" in best_model:
                        # Count actual readmissions (y_test) and predicted readmissions (y_pred)
                        actual_readmissions = int(best_model["y_test"].sum())
                        predicted_readmissions = int(best_model["y_pred"].sum())
                        readmission_counts[period] = {
                            "actual": actual_readmissions,
                            "predicted": predicted_readmissions,
                            "total_patients": len(best_model["y_test"]),
                        }

            if readmission_counts:
                # Display readmission metrics in columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    if "READMIT_30" in readmission_counts:
                        counts = readmission_counts["READMIT_30"]
                        st.metric(
                            "30-Day Readmissions",
                            value=counts["actual"],
                            delta=f"Predicted: {counts['predicted']}",
                            help=f"Actual: {counts['actual']}, Predicted: {counts['predicted']} out of {counts['total_patients']} patients",
                        )
                    else:
                        st.metric("30-Day Readmissions", value="N/A")

                with col2:
                    if "READMIT_60" in readmission_counts:
                        counts = readmission_counts["READMIT_60"]
                        st.metric(
                            "60-Day Readmissions",
                            value=counts["actual"],
                            delta=f"Predicted: {counts['predicted']}",
                            help=f"Actual: {counts['actual']}, Predicted: {counts['predicted']} out of {counts['total_patients']} patients",
                        )
                    else:
                        st.metric("60-Day Readmissions", value="N/A")

                with col3:
                    if "READMIT_90" in readmission_counts:
                        counts = readmission_counts["READMIT_90"]
                        st.metric(
                            "90-Day Readmissions",
                            value=counts["actual"],
                            delta=f"Predicted: {counts['predicted']}",
                            help=f"Actual: {counts['actual']}, Predicted: {counts['predicted']} out of {counts['total_patients']} patients",
                        )
                    else:
                        st.metric("90-Day Readmissions", value="N/A")

                # Add readmission rates
                st.subheader(" Readmission Rates")
                rates_data = []
                for period, counts in readmission_counts.items():
                    period_name = period.replace("READMIT_", "").replace("_", " ")
                    actual_rate = (counts["actual"] / counts["total_patients"]) * 100
                    predicted_rate = (
                        counts["predicted"] / counts["total_patients"]
                    ) * 100
                    rates_data.append(
                        {
                            "Period": period_name,
                            "Actual Rate (%)": f"{actual_rate:.2f}%",
                            "Predicted Rate (%)": f"{predicted_rate:.2f}%",
                            "Total Patients": counts["total_patients"],
                            "Actual Count": counts["actual"],
                            "Predicted Count": counts["predicted"],
                        }
                    )

                if rates_data:
                    rates_df = pd.DataFrame(rates_data)
                    st.dataframe(rates_df, use_container_width=True)
            else:
                st.info("Train models first to see readmission statistics.")
        else:
            st.info("Train models first to see readmission statistics.")

        st.markdown("**Sample: Beneficiary Features**")
        st.dataframe(engineer.feature_matrix.head(20), use_container_width=True)
    else:
        st.info("Upload & prepare data first.")

# 3) Train Models
with tabs[3]:
    st.subheader("Train Prediction Models")

    # Information about model loading improvements
    with st.expander("ℹ️ About Model Loading", expanded=False):
        st.markdown(
            """
        **🚀 Improved Model Loading:**
        - The system now automatically loads existing trained models
        - Performance metrics are generated from saved models without retraining
        - Cost analysis and insights work directly with loaded models
        - Only retrain if you want to update the models with new data
        
        **💡 How it works:**
        1. On first load, the system searches for saved models
        2. Existing models are loaded and performance metrics calculated
        3. Results, insights, and cost analysis are available immediately
        4. Use "Retrain" only if you need to update models
        """
        )

    if "predictor" in st.session_state:
        predictor = st.session_state["predictor"]

        # Auto-load existing models on page load
        if "models_loaded" not in st.session_state:
            st.session_state["models_loaded"] = False

        if not st.session_state["models_loaded"]:
            with st.spinner("🔍 Loading existing models..."):
                loaded_count = 0
                for period in ["READMIT_30", "READMIT_60", "READMIT_90"]:
                    if predictor.load_existing_models(period):
                        loaded_count += 1
                st.session_state["models_loaded"] = True
                if loaded_count > 0:
                    st.success(
                        f"✅ Loaded existing models for {loaded_count} time periods"
                    )
                else:
                    st.info("No existing models found - you can train new ones below")

        # Check if all models are already loaded and functional
        all_loaded = all(
            predictor.models_already_loaded(period)
            for period in ["READMIT_30", "READMIT_60", "READMIT_90"]
        )

        if all_loaded:
            st.success(
                "🎉 All models are loaded and ready! Performance metrics available below."
            )
            st.info(
                "💡 Models are already trained. You can view results below or retrain if needed."
            )

        # Training controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if all_loaded:
                go = st.button("🔄 Retrain All (30/60/90)", type="secondary")
            else:
                go = st.button("🚀 Train All (30/60/90)", type="primary")
        with col2:
            if st.button("📊 Refresh Results", type="secondary"):
                st.rerun()
        with col3:
            if st.button("🧠 Generate Insights", type="secondary"):
                if all_loaded:
                    with st.spinner("Generating insights from loaded models..."):
                        for period in ["READMIT_30", "READMIT_60", "READMIT_90"]:
                            insights = predictor.generate_insights_from_loaded_models(
                                period
                            )
                            if insights:
                                st.write(f"**{period} Insights Generated**")
                else:
                    st.warning("Please load or train models first")

        if go:
            with st.spinner("Training classical models..."):
                for period in ["READMIT_30", "READMIT_60", "READMIT_90"]:
                    predictor.train_ensemble(period)

            if run_dl and TF_OK:
                with st.spinner("Training deep learning model (30-day)..."):
                    predictor.train_deep_learning_model("READMIT_30")
            elif run_dl and not TF_OK:
                st.warning("TensorFlow not available—skipping deep learning.")
            st.success(" Training complete! Models saved automatically.")

        # Display comprehensive results
        if predictor.results:
            st.subheader("📈 Model Performance Results")

            for period in ["READMIT_30", "READMIT_60", "READMIT_90"]:
                if period in predictor.results:
                    # Check if models exist but need retraining
                    if predictor.results[period].get("_needs_retraining"):
                        st.warning(
                            f"⚠️ **{period}**: Models exist but need retraining to show performance metrics"
                        )
                        st.write(
                            f"Found existing models for {period}. Click 'Train All' to retrain and get results."
                        )
                        continue

                    # Check if we have actual results
                    valid_results = {
                        k: v
                        for k, v in predictor.results[period].items()
                        if k not in ["_models_exist", "_needs_retraining"]
                        and isinstance(v, dict)
                        and "auc" in v
                    }

                    if not valid_results:
                        st.info(
                            f"📊 **{period}**: No performance metrics available yet"
                        )
                        continue

                    st.markdown(f"### 🎯 {period} Results")

                    # Show summary banner
                    best_model = max(
                        valid_results.items(), key=lambda x: x[1].get("auc", 0)
                    )
                    st.success(
                        f"🏆 Best Model: **{best_model[0]}** (AUC: {best_model[1]['auc']:.3f})"
                    )

                    # Get performance summary
                    summary_df = predictor.get_model_performance_summary(period)
                    if summary_df is not None:
                        st.dataframe(summary_df, use_container_width=True)

                        # Show detailed metrics for each model
                        for model_name, results in valid_results.items():
                            with st.expander(f"📋 {model_name} - Detailed Metrics"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Accuracy", f"{results['accuracy']:.3f}")
                                    st.metric("AUC", f"{results['auc']:.3f}")
                                with col2:
                                    st.metric(
                                        "F1 Score", f"{results.get('f1_score', 0):.3f}"
                                    )
                                    st.metric(
                                        "Recall", f"{results.get('recall', 0):.3f}"
                                    )
                                with col3:
                                    st.metric(
                                        "Precision",
                                        f"{results.get('precision', 0):.3f}",
                                    )
                                    if "cv_mean" in results:
                                        st.metric(
                                            "CV AUC",
                                            f"{results['cv_mean']:.3f} ± {results.get('cv_std', 0):.3f}",
                                        )

                    st.markdown("---")  # Separator between periods
        else:
            st.info("🚀 Train models first to see results.")
    else:
        st.info("Build dataset first.")

# 4) ROC & AUC
with tabs[4]:
    st.subheader("ROC Curves & AUC")
    if "visualizer" in st.session_state and st.session_state["predictor"].results:
        visualizer = st.session_state["visualizer"]

        # Inline plotting versions (so they render inside Streamlit)
        periods = ["READMIT_30", "READMIT_60", "READMIT_90"]
        fig, axes = plt.subplots(
            1, len(periods), figsize=(18, 6), sharey=True, sharex=True
        )
        fig.suptitle("ROC Curves", fontsize=16)

        for idx, period in enumerate(periods):
            ax = axes[idx]
            ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
            results = st.session_state["predictor"].results.get(period, {})

            # Check if results contain actual model metrics or just flags
            if isinstance(results, dict) and "_models_exist" in results:
                ax.set_title(f"{period} - Models Exist (Need Retraining)")
                continue

            # Check if results is a valid dictionary with model metrics
            if not isinstance(results, dict) or not results:
                ax.set_title(f"{period} - No Data")
                continue

            for name, result in results.items():
                if isinstance(result, dict) and "y_pred_proba" in result:
                    from sklearn.metrics import roc_curve, roc_auc_score

                    fpr, tpr, _ = roc_curve(result["y_test"], result["y_pred_proba"])
                    auc = result.get(
                        "auc", roc_auc_score(result["y_test"], result["y_pred_proba"])
                    )
                    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
            ax.set_title(period)
            ax.set_xlabel("FPR")
            if idx == 0:
                ax.set_ylabel("TPR")
            ax.legend()
            ax.grid(True)

        st.pyplot(fig)
    else:
        st.info("Train models first.")

# 5) Feature Importance
with tabs[5]:
    st.subheader("Random Forest Feature Importances")
    top_n = st.slider("Top features", 5, 30, 15, 1)
    if "predictor" in st.session_state and st.session_state["predictor"].results:
        predictor = st.session_state["predictor"]
        feature_names = predictor.feature_names

        periods = ["READMIT_30", "READMIT_60", "READMIT_90"]
        fig, axes = plt.subplots(1, len(periods), figsize=(20, 8))
        fig.suptitle(f"Top {top_n} Importances", fontsize=16)

        for idx, period in enumerate(periods):
            ax = axes[idx]
            res = predictor.results.get(period, {})

            # Check if results contain actual model metrics or just flags
            if isinstance(res, dict) and "_models_exist" in res:
                ax.set_title(f"{period} - Models Exist (Need Retraining)")
                continue

            # Check if Random Forest model exists and has required data
            if "Random Forest" not in res or "model" not in res["Random Forest"]:
                ax.set_title(f"{period} - No RF Model")
                continue

            rf_model = res["Random Forest"]["model"]
            model_importances = getattr(rf_model, "feature_importances_", None)
            if model_importances is None or len(model_importances) == 0:
                ax.set_title(f"{period} - No importance data")
                continue
            names = (
                feature_names
                if feature_names and len(feature_names) == len(model_importances)
                else [f"Feature {i+1}" for i in range(len(model_importances))]
            )
            importances = pd.Series(model_importances, index=names).nlargest(top_n)
            sns.barplot(x=importances.values, y=importances.index, ax=ax)
            ax.set_title(period)
            ax.set_xlabel("Importance")
        st.pyplot(fig)
    else:
        st.info("Train models first.")

# 6) Costs & Recommendations
with tabs[6]:
    st.subheader("Cost Impact & Recommendations")
    if "insights" in st.session_state and st.session_state["predictor"].results:
        insights = st.session_state["insights"]
        with st.spinner("Estimating cost impact..."):
            cost_analysis = insights.estimate_treatment_costs()

        st.markdown("**Potential Savings (by horizon)**")
        if cost_analysis:
            rows = []
            for k, v in cost_analysis.items():
                rows.append(
                    {
                        "Horizon": k,
                        "Potential Savings (USD)": round(
                            v.get("potential_savings", 0), 2
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("**Recommendations (by risk tier)**")
        recs = insights.generate_treatment_recommendations()
        st.json(recs, expanded=False)
    else:
        st.info("Train models first.")

# 7) Disease Progression
with tabs[7]:
    st.subheader("Disease Progression & Comorbidities")
    if "tracker" in st.session_state:
        tracker = st.session_state["tracker"]
        prog = tracker.analyze_condition_progression()
        if prog is not None and not prog.empty:
            st.write("Prevalence (%) by Year")
            st.dataframe(prog.round(2), use_container_width=True)

        comorb = tracker.identify_comorbidity_patterns()
        if comorb:
            st.write("Top Comorbidity Pairs (counts)")
            top_pairs = sorted(comorb.items(), key=lambda kv: kv[1], reverse=True)[:10]
            st.dataframe(
                pd.DataFrame(top_pairs, columns=["Pair", "Count"]),
                use_container_width=True,
            )
    else:
        st.info("Build dataset first.")

# 8) Patient Explorer
with tabs[8]:
    st.subheader("Patient Explorer")
    if (
        "engineer" in st.session_state
        and hasattr(st.session_state["engineer"], "feature_matrix")
        and st.session_state["engineer"].feature_matrix is not None
    ):
        engineer = st.session_state["engineer"]
        df = engineer.feature_matrix.copy()
        id_list = (
            df["DESYNPUF_ID"].astype(str).tolist()[:1000]
        )  # limit to keep UI snappy
        pick = st.selectbox("Select Patient ID", id_list)
        if pick:
            row = df[df["DESYNPUF_ID"].astype(str) == pick]
            st.write("**Patient Feature Vector**")
            st.dataframe(row.T, use_container_width=True)
    else:
        st.info("Build dataset first.")

# 9) Model Management
with tabs[9]:
    st.subheader(" Model Management")

    if "predictor" in st.session_state:
        predictor = st.session_state["predictor"]

        # Show saved models
        st.subheader(" Saved Models")
        saved_models = predictor.list_saved_models()

        if saved_models:
            st.success(f" Found {len(saved_models)} saved models")

            # Display model files
            for model_file in saved_models:
                st.write(f"• `{model_file}`")

            # Model loading section
            st.subheader("Load & Use Saved Models")

            col1, col2 = st.columns(2)
            with col1:
                selected_model = st.selectbox("Select Model", saved_models)

            with col2:
                if st.button("Load Model Info", type="secondary"):
                    st.info(f"Selected: {selected_model}")
                    # Here you could add model metadata display

        else:
            st.info(
                "No saved models found. Train models first to save them automatically."
            )

        # Model performance export
        if st.session_state["predictor"].results:
            st.subheader("Export Model Performance")

            if st.button("Export Performance to CSV", type="primary"):
                for period in ["READMIT_30", "READMIT_60", "READMIT_90"]:
                    if period in predictor.results:
                        summary_df = predictor.get_model_performance_summary(period)
                        if summary_df is not None:
                            csv_data = summary_df.to_csv(index=False)
                            st.download_button(
                                label=f"Download {period} Performance",
                                data=csv_data,
                                file_name=f"model_performance_{period}.csv",
                                mime="text/csv",
                            )
        else:
            st.info("Train models first to see performance metrics.")
    else:
        st.info("Build dataset first.")

# 10) Batch Predictions
with tabs[10]:
    st.subheader(" Batch Predictions")

    if "predictor" in st.session_state:
        predictor = st.session_state["predictor"]

        # Check if we have saved models
        saved_models = predictor.list_saved_models()

        if saved_models:
            st.success(f"Found {len(saved_models)} saved models for predictions")

            # Model selection
            st.subheader(" Select Model & Data")

            col1, col2 = st.columns(2)
            with col1:
                selected_model = st.selectbox("Select Model File", saved_models)
                target_period = st.selectbox(
                    "Target Period", ["READMIT_30", "READMIT_60", "READMIT_90"]
                )

            with col2:
                # File upload for new data
                new_data_file = st.file_uploader("Upload New Data (CSV)", type=["csv"])

                if new_data_file:
                    st.success(f" Data uploaded: {new_data_file.name}")

            # Guidance: required columns and template
            st.markdown("---")
            st.markdown(
                "**Required feature columns**: These must match the training features exactly (order doesn't matter)."
            )
            # Ensure feature names are available. If empty, derive via prepare_data without training
            req_features = (
                predictor.feature_names if hasattr(predictor, "feature_names") else []
            )
            if (not req_features) and hasattr(predictor, "prepare_data"):
                try:
                    _ = predictor.prepare_data(target_col=target_period)
                    req_features = predictor.feature_names
                except Exception:
                    req_features = []
            if req_features:
                st.code(",".join(req_features), language="text")
                template_csv = pd.DataFrame(columns=req_features).to_csv(index=False)
                st.download_button(
                    label="Download CSV Template",
                    data=template_csv,
                    file_name="batch_prediction_template.csv",
                    mime="text/csv",
                )
                # Optional: generate sample data quickly for demo
                with st.expander("Generate sample data for a quick demo"):
                    n_rows = st.slider("Rows", 10, 200, 50, 10)
                    seed_val = st.number_input("Seed", 0, 999999, 42)
                    if st.button("⚡ Generate Sample Data"):
                        rng = np.random.default_rng(int(seed_val))
                        sample = pd.DataFrame(
                            rng.normal(
                                loc=0.0, scale=1.0, size=(n_rows, len(req_features))
                            ),
                            columns=req_features,
                        )
                        for col in sample.columns:
                            lower = col.lower()
                            if any(
                                k in lower
                                for k in [
                                    "age",
                                    "num",
                                    "count",
                                    "los",
                                    "cost",
                                    "pmt",
                                    "days",
                                    "total",
                                ]
                            ):
                                sample[col] = sample[col].abs()
                        st.session_state["batch_sample_df"] = sample
                    if "batch_sample_df" in st.session_state:
                        st.dataframe(
                            st.session_state["batch_sample_df"].head(20),
                            use_container_width=True,
                        )
                        st.download_button(
                            label=" Download Sample CSV",
                            data=st.session_state["batch_sample_df"].to_csv(
                                index=False
                            ),
                            file_name="sample_batch_data.csv",
                            mime="text/csv",
                        )
            else:
                st.info(
                    "Feature list is not available yet. Train models in 'Train Models' to populate feature names."
                )

            # If uploaded, pre-validate columns before prediction
            if "new_data_file" in locals() and new_data_file is not None:
                try:
                    preview_df = pd.read_csv(new_data_file)
                    if req_features:
                        missing = sorted(
                            list(set(req_features) - set(preview_df.columns))
                        )
                        extra = sorted(
                            list(set(preview_df.columns) - set(req_features))
                        )
                        if missing:
                            st.warning(
                                f"Missing required columns ({len(missing)}): {', '.join(missing[:30])}{' ...' if len(missing)>30 else ''}"
                            )
                        else:
                            st.success("All required columns are present.")
                        if extra:
                            with st.expander("Columns in your file that are not used"):
                                st.write(extra)
                    with st.expander("Preview uploaded data (first 20 rows)"):
                        st.dataframe(preview_df.head(20), use_container_width=True)
                except Exception as _e:
                    st.error("Could not read the uploaded CSV for validation.")

            # Make predictions
            use_sample = st.checkbox(
                "Use generated sample data instead of upload", value=False
            )
            if st.button("Make Predictions", type="primary") and (
                (new_data_file is not None)
                or (use_sample and "batch_sample_df" in st.session_state)
            ):
                try:
                    # Load new data
                    if use_sample and "batch_sample_df" in st.session_state:
                        new_data = st.session_state["batch_sample_df"].copy()
                    else:
                        new_data = pd.read_csv(new_data_file)
                    st.write(
                        f"Loaded {len(new_data)} records with {len(new_data.columns)} features"
                    )

                    # Find corresponding scaler
                    scaler_files = [
                        f for f in saved_models if "scaler" in f and target_period in f
                    ]

                    if scaler_files:
                        scaler_file = scaler_files[0]
                        predictions, message = predictor.predict_with_saved_model(
                            selected_model, scaler_file, new_data
                        )

                        if predictions is not None:
                            st.success("Predictions completed!")

                            # Display results
                            results_df = pd.DataFrame(
                                {
                                    "Patient_ID": range(len(predictions)),
                                    "Readmission_Probability": predictions,
                                    "Risk_Level": [
                                        (
                                            "High"
                                            if p > 0.7
                                            else "Medium" if p > 0.3 else "Low"
                                        )
                                        for p in predictions
                                    ],
                                }
                            )

                            st.subheader("Prediction Results")
                            st.dataframe(results_df, use_container_width=True)

                            # Download results
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                label=" Download Predictions",
                                data=csv_data,
                                file_name=f"readmission_predictions_{target_period}.csv",
                                mime="text/csv",
                            )
                        else:
                            st.error(f"❌ Prediction failed: {message}")
                    else:
                        st.error("❌ No matching scaler found for the selected period")

                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
        else:
            st.info("No saved models found. Train models first to enable predictions.")
    else:
        st.info("Build dataset first.")

# 11) Report
with tabs[11]:
    st.subheader("Generate & Download Report")
    if "reporter" in st.session_state and st.session_state["predictor"].results:
        reporter = st.session_state["reporter"]
        if st.button("Generate Executive Summary (.txt)", type="primary"):
            with st.spinner("Generating report..."):
                reporter.generate_executive_summary("healthcare_analytics_report.txt")
            st.success("Report generated!")

        # Offer download if exists
        if os.path.exists("healthcare_analytics_report.txt"):
            with open("healthcare_analytics_report.txt", "rb") as f:
                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name="healthcare_analytics_report.txt",
                    mime="text/plain",
                )
    else:
        st.info("Train models first.")
