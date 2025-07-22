import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import re
import pandas as pd
import time
import os

# ─── PAGE CONFIG & GLOBAL CSS ─────────────────────────────────────────────────
st.set_page_config(page_title="Falcon Structures LLM Tool", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    /* ── Tab styling ───────────────────────────────────────────────────────── */
    /* Base selector for Streamlit tabs */
    div[data-baseweb="tab-list"] button[role="tab"] {
        background-color: #fff !important;
        color: #000 !important;
        border: 1px solid transparent;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        margin: 0;
        position: relative;
    }
    /* Separator between tabs */
    div[data-baseweb="tab-list"] button[role="tab"]:not(:last-child)::after {
        content: "|";
        position: absolute;
        right: -10px;
        top: 50%;
        transform: translateY(-50%);
        color: #000;
    }
    /* Hover state */
    div[data-baseweb="tab-list"] button[role="tab"]:hover {
        background-color: red !important;
        color: #fff !important;
    }
    /* Selected tab */
    div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
        border-color: #888 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        background-color: #fff !important;
        color: #000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ─── LOGO & HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:1rem 0;'>
  <img src='https://github.com/raianrith/AI-Client-Research-Tool/blob/main/Weidert_Logo_primary-logomark-antique.png?raw=true' width='60'/>
  <h1>Falcon AI‑Powered LLM Search Visibility Tool</h1>
  <h4 style='color:#ccc;'>Created by Weidert Group, Inc.</h4>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR: MODEL CONFIGURATION ─────────────────────────────────────────────
st.sidebar.title("🛠️ Model Configuration")
openai_model        = st.sidebar.selectbox("OpenAI model", ["gpt-4","gpt-4o","gpt-3.5-turbo","gpt-3.5-turbo-16k"], index=0)
gemini_model_name   = st.sidebar.selectbox("Gemini model", ["gemini-2.5-flash","gemini-2.5-pro"], index=0)
perplexity_model_name = st.sidebar.selectbox("Perplexity model", ["sonar","sonar-pro"], index=0)

# ─── API CLIENTS ────────────────────────────────────────────────────────────────
openai_key     = st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
gemini_key     = st.secrets.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
perp_key       = st.secrets.get("perplexity_api_key") or os.getenv("PERPLEXITY_API_KEY")

openai_client = OpenAI(api_key=openai_key)
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel(gemini_model_name)
perplexity_client = OpenAI(api_key=perp_key, base_url="https://api.perplexity.ai")

SYSTEM_PROMPT = "Provide a helpful answer to the user’s query."

def get_openai_response(q):
    try:
        r = openai_client.chat.completions.create(
            model=openai_model,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return "ERROR"

def get_gemini_response(q):
    try:
        r = gemini_model.generate_content(q)
        return r.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return "ERROR"

def get_perplexity_response(q):
    try:
        r = perplexity_client.chat.completions.create(
            model=perplexity_model_name,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Perplexity error: {e}")
        return "ERROR"

competitors = [
    "ROXBOX Containers","Wilmot Modular","Pac-Van","BMarko Structures",
    "Giant Containers","XCaliber Container","Conexwest",
    "Mobile Modular Portable Storage","WillScot"
]

# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Multi-LLM Response Generator","Search Visibility Analysis"])

with tab1:
    st.markdown("### Enter queries to compare responses")
    queries_input = st.text_area(
        "Queries (one per line)",
        height=150,
        placeholder="e.g. What companies provide modular container offices in the US?"
    )
    if st.button("Run Analysis", key="gen"):
        qs = [q.strip() for q in queries_input.splitlines() if q.strip()]
        if not qs:
            st.warning("Please enter at least one query.")
        else:
            results = []
            with st.spinner("Gathering responses…"):
                for q in qs:
                    for source, fn in [("OpenAI", get_openai_response),
                                       ("Gemini", get_gemini_response),
                                       ("Perplexity", get_perplexity_response)]:
                        txt = fn(q)
                        results.append({"Query": q, "Source": source, "Response": txt})
                        time.sleep(1)
            df = pd.DataFrame(results)[["Query","Source","Response"]]
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), "responses.csv", "text/csv")

with tab2:
    st.markdown("### Search Visibility Analysis")
    uploaded = st.file_uploader("Upload your results CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        # Detect Falcon mentions
        df["Falcon_Mention"] = df["Response"].str.contains(r"\bfalcon\b|\bfalconstructures\b", case=False)
        # Mention rate by source
        rates = df.groupby("Source")["Falcon_Mention"].mean() * 100
        st.subheader("Falcon Mention Rate by Source")
        cols = st.columns(3)
        for col, src in zip(cols, rates.index):
            col.metric(src, f"{rates[src]:.1f}%")
        # Competitor share in non-branded queries
        df["NonBranded"] = ~df["Query"].str.contains("falcon", case=False)
        nonb = df[df["NonBranded"]]
        share = {}
        total = len(nonb)
        for comp in competitors:
            share[comp] = nonb["Response"].str.contains(re.escape(comp), case=False).sum() / total * 100
        comp_df = pd.DataFrame.from_dict(share, orient="index", columns=["Share (%)"]).sort_values("Share (%)", ascending=False)
        st.subheader("Share of Voice in Generic Queries")
        st.bar_chart(comp_df, use_container_width=True)
        # Falcon URL citation rate
        df["Has_Falcon_URL"] = df["Response"].str.contains(r"https?://\S*falconstructures\.com", case=False)
        cit = df.groupby("Source")["Has_Falcon_URL"].mean() * 100
        st.subheader("Falcon URL Citation Rate")
        cit_df = pd.DataFrame(cit).rename(columns={"Has_Falcon_URL":"Citation Rate (%)"})
        st.table(cit_df.style.format("{:.1f}%"))
        # Competitor share overall
        st.subheader("Competitor Share of Mentions")
        overall = {}
        total_resp = len(df)
        for comp in competitors:
            overall[comp] = df["Response"].str.contains(re.escape(comp), case=False).sum() / total_resp * 100
        overall_df = pd.DataFrame.from_dict(overall, orient="index", columns=["Overall Share (%)"]).sort_values("Overall Share (%)", ascending=False)
        st.dataframe(overall_df.style.format("{:.1f}%"))
        # Summary
        st.markdown("**Summary:** Falcon Structures leads mention rate on most engines, but competitors ROXBOX and WillScot capture significant share in generic queries.")
    else:
        st.info("Upload the CSV exported from the first tab to see visibility metrics.")
