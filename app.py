import os
import re
import time

import google.generativeai as genai
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from openai import OpenAI

load_dotenv()

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# ─── PAGE CONFIG (unchanged) ───────────────────────────────────────────────────
st.set_page_config(page_title="Falcon Structures LLM Tool", layout="wide")

# ─── MINIMAL THEME / CSS (UI only) ────────────────────────────────────────────
st.markdown(
    """
<style>
/* Minimal, light theme */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

:root{
  --bg:#f7f8fa;
  --surface:#ffffff;
  --text:#0f172a;
  --muted:#64748b;
  --stroke:#e5e7eb;
  --accent:#2563eb; /* blue-600 */
}

html, body, .stApp { color: var(--text); background: var(--bg); font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
.block-container { max-width: 1180px; padding-top: 0.75rem; }

/* Header (simple + centered) */
.hero {
  margin: 6px 0 18px 0;
  padding: 16px 18px;
  border-radius: 12px;
  background: var(--surface);
  border: 1px solid var(--stroke);
}
.hero .row { display:flex; align-items:center; gap:12px; justify-content:center; flex-wrap:wrap; text-align:center; }
.hero img { width: 52px; height: 52px; border-radius: 10px; }
.hero h1 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: .2px; }
.hero .subtitle { margin-top: 2px; color: var(--muted); font-size: 14px; }

/* Sidebar */
[data-testid="stSidebar"] > div {
  background: var(--surface);
  border-right: 1px solid var(--stroke);
}
.sidebar-title { font-weight: 600; }

/* Tabs: simple underline style */
div[data-baseweb="tab-list"] {
  display:flex !important; justify-content:center !important;
  gap: 6px; margin-bottom: 8px;
}
div[data-baseweb="tab-list"] button[role="tab"]{
  background: transparent !important;
  color: var(--muted) !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 12px !important;
}
div[data-baseweb="tab-list"] button[role="tab"]:hover{
  color: var(--text) !important;
  border-bottom-color: var(--stroke) !important;
}
div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"]{
  color: var(--text) !important;
  border-bottom-color: var(--accent) !important;
}

/* Cards/sections */
.section {
  background: var(--surface);
  border: 1px solid var(--stroke);
  border-radius: 12px;
  padding: 16px 16px;
  margin: 10px 0 18px 0;
}

/* Inputs */
.stTextArea textarea, .stTextInput input, .stSelectbox [data-baseweb="select"] > div {
  background: #fff !important;
  border-color: var(--stroke) !important;
  color: var(--text) !important;
  border-radius: 10px !important;
}
.stFileUploader {
  background: #fff;
  border: 1px dashed var(--stroke); border-radius: 12px; padding: 10px;
}

/* Buttons & downloads (subtle) */
div.stButton > button, .stDownloadButton button {
  background: var(--accent);
  color:#fff; border: 1px solid var(--accent); border-radius: 10px;
  padding: 0.55rem 0.9rem; font-weight: 600; letter-spacing:.2px;
}
div.stButton > button:hover, .stDownloadButton button:hover{ filter: brightness(1.05); }
div.stButton > button { margin: .4rem auto .2rem auto; display:block; }

/* Metrics */
[data-testid="stMetric"]{
  background: var(--surface);
  border: 1px solid var(--stroke);
  border-radius: 12px;
  padding: 10px 12px;
}

/* Tables */
[data-testid="stTable"] table, .stDataFrame { color: var(--text); }
caption, .stCaption { color: var(--muted) !important; }

/* Hide default chrome */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# Optional: make seaborn/pyplot minimal
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.edgecolor": "#e5e7eb",
    "axes.labelcolor": "#0f172a",
    "text.color": "#0f172a",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "grid.color": "#f1f5f9"
})

# ─── HEADER (content unchanged, just styled) ───────────────────────────────────
st.markdown(
    """
<div class="hero">
  <div class="row">
    <img src="https://github.com/raianrith/AI-Client-Research-Tool/blob/main/Weidert_Logo_primary-logomark-antique.png?raw=true" />
    <div>
      <h1>Falcon AI-Powered LLM Search Visibility Tool</h1>
      <div class="subtitle">Created by Weidert Group, Inc.</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─── SIDEBAR (unchanged functionality) ─────────────────────────────────────────
st.sidebar.markdown("<div class='sidebar-title'>🛠️ Model Configuration</div>", unsafe_allow_html=True)
openai_model = st.sidebar.selectbox(
    "OpenAI model", ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"], index=0
)
gemini_model_name = st.sidebar.selectbox("Gemini model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
perplexity_model_name = st.sidebar.selectbox("Perplexity model", ["sonar", "sonar-pro"], index=0)

# ─── API CLIENTS (unchanged) ──────────────────────────────────────────────────
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("openai_api_key")
gemini_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("gemini_api_key")
perp_key = os.getenv("PERPLEXITY_API_KEY") or st.secrets.get("perplexity_api_key")

openai_client = OpenAI(api_key=openai_key)
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel(gemini_model_name)
perplexity_client = OpenAI(api_key=perp_key, base_url="https://api.perplexity.ai")

SYSTEM_PROMPT = "Provide a helpful answer to the user’s query."

def get_openai_response(q):
    try:
        r = openai_client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
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
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Perplexity error: {e}")
        return "ERROR"

# ─── TABS (structure/logic unchanged) ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["Multi-LLM Response Generator", "Search Visibility Analysis", "Time Series Analysis"]
)

with tab1:
    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.markdown(
        '<h5 style="text-align:center; margin-bottom:0.75rem; color:#64748b">'
        "Enter queries to generate responses from OpenAI (Chat GPT), Gemini, & Perplexity."
        "</h5>",
        unsafe_allow_html=True,
    )
    queries_input = st.text_area(
        "Queries (one per line)",
        height=200,
        placeholder="e.g. What companies provide modular container offices in the US?",
    )
    _, center, _ = st.columns([1, 2, 1])
    with center:
        run = st.button("Run Analysis", key="run")

    if run:
        qs = [q.strip() for q in queries_input.splitlines() if q.strip()]
        if not qs:
            st.warning("Please enter at least one query.")
        else:
            results = []
            with st.spinner("Gathering responses…"):
                for q in qs:
                    for source, fn in [
                        ("OpenAI", get_openai_response),
                        ("Gemini", get_gemini_response),
                        ("Perplexity", get_perplexity_response),
                    ]:
                        txt = fn(q)
                        results.append({"Query": q, "Source": source, "Response": txt})
                        time.sleep(1)

            df = pd.DataFrame(results)[["Query", "Source", "Response"]]
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), "responses.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.markdown("### Search Visibility Analysis")
    uploaded = st.file_uploader("Upload your results CSV", type="csv")

    if uploaded:
        df_main = pd.read_csv(uploaded)

        competitors = [
            "ROXBOX", "Wilmot", "Pac-Van", "BMarko", "Giant", "XCaliber",
            "Conexwest", "Mobile Modular", "WillScot",
        ]
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(c) for c in competitors) + r")\b", flags=re.IGNORECASE
        )

        def extract_competitors(text):
            matches = pattern.findall(text or "")
            found = []
            for m in matches:
                for comp in competitors:
                    if m.lower() == comp.lower():
                        found.append(comp)
            return ", ".join(sorted(set(found)))

        from datetime import datetime

        df_main["Date"] = datetime.today().date()
        df_main["Competitors Mentioned"] = df_main["Response"].apply(extract_competitors)
        df_main["Branded Query"] = (
            df_main["Query"].str.contains("falcon", case=False, na=False).map({True: "Y", False: "N"})
        )
        df_main["Falcon Mentioned"] = (
            df_main["Response"].str.contains("falcon", case=False, na=False).map({True: "Y", False: "N"})
        )
        df_main["Sources Cited"] = (
            df_main["Response"].str.findall(r"(https?://\S+)").apply(lambda lst: ", ".join(lst) if lst else "")
        )
        df_main["Response Word-Count"] = df_main["Response"].astype(str).str.split().str.len()
        df_main["Query Number"] = pd.factorize(df_main["Query"])[0] + 1
        df_main = df_main[
            ["Date","Query Number","Query","Source","Response","Response Word-Count",
             "Branded Query","Falcon Mentioned","Competitors Mentioned","Sources Cited"]
        ]

        st.subheader("🧹 Cleaned Dataset")
        st.dataframe(df_main, use_container_width=True, height=420)
        st.download_button(
            "Download Cleaned CSV", df_main.to_csv(index=False), "cleaned_responses.csv", "text/csv"
        )

        st.divider()

        st.subheader("📊 Mention Rates")
        overall_rate = (
            df_main.groupby("Source")["Falcon Mentioned"]
            .apply(lambda x: (x == "Y").mean() * 100)
            .round(1)
        )
        cols = st.columns(len(overall_rate))
        for col, src in zip(cols, overall_rate.index):
            col.metric(f"{src} Mentions Falcon", f"{overall_rate[src]}%")

        st.caption("Breakdown of Falcon mentions for branded vs. non-branded queries by source.")
        mention_rate = (
            df_main.groupby(["Source", "Branded Query"])["Falcon Mentioned"]
            .apply(lambda x: (x == "Y").mean() * 100)
            .reset_index(name="Mention Rate (%)")
        )
        pivot = (
            mention_rate.pivot(index="Source", columns="Branded Query", values="Mention Rate (%)")
            .rename(columns={"Y": "Branded (%)", "N": "Non-Branded (%)"})
            .round(1)
        )
        st.dataframe(pivot.reset_index(), use_container_width=True)

        st.divider()

        # Falcon URL citation detection
        df_main["Falcon URL Cited"] = df_main["Response"].str.contains(
            r"https?://(?:www\.)?falconstructures\.com", case=False, regex=True, na=False
        )

        st.subheader("🔗 Falcon URL Citation Rate")
        st.caption("How often each source linked to Falcon’s website.")
        cit_rate = df_main.groupby("Source")["Falcon URL Cited"].mean().mul(100).round(1)
        st.bar_chart(cit_rate, height=210, use_container_width=True)

        st.divider()

        df_main["sentiment_score"] = (
            df_main["Response"]
            .fillna("")
            .apply(lambda t: ((sia.polarity_scores(t)["compound"] + 1) / 2) * 9 + 1)
        )
        sentiment_df = df_main.groupby("Source")["sentiment_score"].mean().round(1)
        st.subheader("💬 Average Sentiment per LLM")
        st.bar_chart(sentiment_df, height=210, use_container_width=True)

        st.divider()

        mask = (
            (df_main["Falcon Mentioned"] == "N")
            & df_main["Competitors Mentioned"].notna()
            & (df_main["Competitors Mentioned"].str.strip() != "")
        )
        gaps = df_main[mask][["Source", "Query", "Response", "Competitors Mentioned"]]
        st.subheader("⚠️ Competitor-Only Gaps (No Falcon Mention)")
        st.dataframe(gaps.reset_index(drop=True), use_container_width=True)

        st.divider()

        def compute_brand_share(df, query_type="Y"):
            df_subset = df[df["Branded Query"] == query_type].copy()

            brand_mentions = []
            for _, row in df_subset.iterrows():
                brands = []
                if row["Falcon Mentioned"] == "Y":
                    brands.append("Falcon Structures")
                competitors = row["Competitors Mentioned"]
                if pd.notna(competitors) and competitors.strip():
                    brands += [x.strip() for x in competitors.split(",")]

                for brand in brands:
                    brand_mentions.append({"Brand": brand, "Source": row["Source"]})

            mention_df = pd.DataFrame(brand_mentions)
            if mention_df.empty:
                return pd.DataFrame(columns=["Brand", "Overall Share (%)"])

            overall = (
                mention_df["Brand"]
                .value_counts(normalize=True)
                .mul(100)
                .round(1)
                .rename("Overall Share (%)")
            )
            by_source = mention_df.groupby(["Brand", "Source"]).size().unstack(fill_value=0)
            by_source = by_source.div(by_source.sum(axis=0), axis=1).mul(100).round(1)

            full = (
                pd.concat([overall, by_source], axis=1)
                .fillna(0)
                .reset_index()
                .rename(columns={"index": "Brand"})
            )
            return full.sort_values("Overall Share (%)", ascending=False)

        st.subheader("🏷️ Brand Share — Non-Branded Queries")
        nonbranded_df = compute_brand_share(df_main, query_type="N")
        st.dataframe(nonbranded_df, use_container_width=True)

        st.divider()

        st.subheader("📈 Response Word Count Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(data=df_main, x="Source", y="Response Word-Count", ax=ax, palette="pastel")
        ax.set_title("Distribution of Response Lengths by Source")
        st.pyplot(fig)

        st.subheader("📄 Daily Summary by Source for Google Sheet")
        st.caption("Use this table to export summary metrics per source per run.")

        st.divider()

        import datetime
        today = datetime.date.today()

        df_main["Branded Query"] = (
            df_main["Query"].str.contains("falcon", case=False, na=False).map({True: "Y", False: "N"})
        )
        df_main["Falcon Mentioned"] = (
            df_main["Response"].str.contains("falcon", case=False, na=False).map({True: "Y", False: "N"})
        )
        df_main["Falcon URL Cited"] = df_main["Response"].str.contains(
            r"https?://(?:www\.)?falconstructures\.com", case=False, na=False
        )

        mention_rates = (
            df_main.groupby(["Source", "Branded Query"])["Falcon Mentioned"]
            .apply(lambda x: (x == "Y").mean() * 100)
            .unstack()
            .round(1)
            .rename(columns={"Y": "Branded Mention Rate (%)", "N": "Non-Branded Mention Rate (%)"})
        )

        citation_rates = (
            df_main.groupby(["Source", "Branded Query"])["Falcon URL Cited"]
            .mean()
            .mul(100)
            .unstack()
            .round(1)
            .rename(
                columns={
                    True: "Branded URL Citation Rate (%)",
                    False: "Non-Branded URL Citation Rate (%)",
                }
            )
        )

        def compute_brand_share_simple(df, brand_filter):
            mentions = []
            for source in df["Source"].unique():
                sub_df = df[(df["Source"] == source) & brand_filter]
                total = len(sub_df)
                falcon_mentions = (
                    sub_df["Response"].str.contains("falcon", case=False, na=False).sum()
                )
                share = (falcon_mentions / total) * 100 if total > 0 else 0
                mentions.append((source, round(share, 1)))
            return dict(mentions)

        branded_share = compute_brand_share_simple(df_main, df_main["Branded Query"] == "Y")
        nonbranded_share = compute_brand_share_simple(df_main, df_main["Branded Query"] == "N")

        brand_share_df = pd.DataFrame(
            {
                "Falcon Brand Share (Branded)": pd.Series(branded_share),
                "Falcon Brand Share (Non-Branded)": pd.Series(nonbranded_share),
            }
        )

        summary_df = mention_rates.join(citation_rates, how="outer").join(brand_share_df, how="outer")
        summary_df["Date"] = today
        summary_df.reset_index(inplace=True)

        st.subheader("📊 Daily Summary Metrics by Source")
        st.dataframe(summary_df, use_container_width=True)

        st.download_button(
            label="Download Daily Summary CSV",
            data=summary_df.to_csv(index=False),
            file_name=f"Falcon_LLM_Summary_{today}.csv",
            mime="text/csv",
        )

    else:
        st.info("Please upload the raw CSV to begin analysis.")

    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.markdown("### 📈 Time Series Analysis")
    st.caption("Track changes in key search visibility metrics over time.")

    json_key = st.file_uploader("Upload your Google Sheets service account key (.json)", type="json")

    if json_key is not None:
        import tempfile
        import gspread
        import nltk
        import pandas as pd
        from gspread_dataframe import get_as_dataframe
        from nltk.sentiment import SentimentIntensityAnalyzer
        from oauth2client.service_account import ServiceAccountCredentials

        nltk.download("vader_lexicon", quiet=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            tmp_file.write(json_key.read())
            tmp_file_path = tmp_file.name

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(tmp_file_path, scope)
        client = gspread.authorize(creds)

        st.divider()

        sheet = client.open("Falcon_Search_Visibility_Data").sheet1
        df_main = get_as_dataframe(sheet).dropna(how="all")
        df_main = df_main.dropna(axis=1, how="all")

        df_main["Date"] = pd.to_datetime(df_main["Date"])

        sia = SentimentIntensityAnalyzer()

        mention_rates_ts = (
            df_main.groupby(["Date", "Source", "Branded Query"])["Falcon Mentioned"]
            .apply(lambda x: (x == "Y").mean() * 100)
            .reset_index(name="Falcon Mention Rate")
        )

        brand_share_ts = (
            df_main.groupby(["Date", "Source"])
            .apply(lambda g: (g["Falcon Mentioned"] == "Y").sum() / len(g) * 100)
            .reset_index(name="Falcon Brand Share")
        )

        df_main["Falcon URL Cited"] = df_main["Sources Cited"].str.contains(
            "falconstructures.com", na=False, case=False
        )
        citation_rate_ts = (
            df_main.groupby(["Date", "Source"])["Falcon URL Cited"]
            .mean()
            .mul(100)
            .reset_index(name="Citation Rate")
        )

        df_main["Response Word-Count"] = df_main["Response"].astype(str).str.split().str.len()
        word_count_ts = (
            df_main.groupby(["Date", "Source"])["Response Word-Count"]
            .mean()
            .reset_index(name="Avg Word Count")
        )

        df_main["sentiment_score"] = (
            df_main["Response"]
            .fillna("")
            .apply(lambda t: ((sia.polarity_scores(t)["compound"] + 1) / 2) * 9 + 1)
        )
        sentiment_ts = (
            df_main.groupby(["Date", "Source"])["sentiment_score"]
            .mean()
            .reset_index(name="Avg Sentiment")
        )

        st.subheader("Falcon Mention Rate (Branded & Non-Branded)")
        for bq in ["Y", "N"]:
            label = "Branded Queries" if bq == "Y" else "Non-Branded Queries"
            st.markdown(f"**{label}**")
            subset = mention_rates_ts[mention_rates_ts["Branded Query"] == bq].pivot(
                index="Date", columns="Source", values="Falcon Mention Rate"
            )
            st.line_chart(subset, height=240, use_container_width=True)

        st.divider()

        st.subheader("Falcon Brand Share Over Time")
        share_pivot = brand_share_ts.pivot(index="Date", columns="Source", values="Falcon Brand Share")
        st.line_chart(share_pivot, height=240, use_container_width=True)

        st.divider()

        st.subheader("Falcon URL Citation Rate Over Time")
        cite_pivot = citation_rate_ts.pivot(index="Date", columns="Source", values="Citation Rate")
        st.line_chart(cite_pivot, height=240, use_container_width=True)

        st.divider()

        st.subheader("Average Response Word Count")
        wc_pivot = word_count_ts.pivot(index="Date", columns="Source", values="Avg Word Count")
        st.line_chart(wc_pivot, height=240, use_container_width=True)

        st.divider()

        st.subheader("Average Sentiment Score (1-10 Scale)")
        sent_pivot = sentiment_ts.pivot(index="Date", columns="Source", values="Avg Sentiment")
        st.line_chart(sent_pivot, height=240, use_container_width=True)

    else:
        st.warning("⬆️ Please upload your service account `.json` file to begin.")

    st.markdown('</div>', unsafe_allow_html=True)
