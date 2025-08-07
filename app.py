import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import re
import pandas as pd
import time
import os
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ─── PAGE CONFIG & GLOBAL CSS ─────────────────────────────────────────────────
st.set_page_config(page_title="Falcon Structures LLM Tool", layout="wide")

# Custom CSS
st.markdown("""
<style>
div[data-baseweb="tab-list"] {
    display: flex !important;
    justify-content: center !important;
}
div[data-baseweb="tab-list"] button[role="tab"] {
    background-color: #fff !important;
    color: #000 !important;
    border: 1px solid transparent;
    border-radius: 4px 4px 0 0;
    padding: 0.5rem 1rem;
    margin: 0;
    position: relative;
}
div[data-baseweb="tab-list"] button[role="tab"]:not(:last-child)::after {
    content: "|";
    position: absolute;
    right: -10px;
    top: 50%;
    transform: translateY(-50%);
    color: #000;
}
div[data-baseweb="tab-list"] button[role="tab"]:hover {
    background-color: red !important;
    color: #fff !important;
}
div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
    border-color: #888 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    background-color: #fff !important;
    color: #000 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
  <style>
    div.stButton > button {
      margin: 0 auto;
      display: block;
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

# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Multi-LLM Response Generator","Search Visibility Analysis"])

with tab1:
    st.markdown(
        '<h5 style="text-align:center; margin-bottom:1rem; color:#a9a9a9">'
        'Enter queries to generate responses from OpenAI (Chat GPT), Gemini, & Perplexity.'
        '</h5>',
        unsafe_allow_html=True
    )
    queries_input = st.text_area(
        "Queries (one per line)",
        height=200,
        placeholder="e.g. What companies provide modular container offices in the US?"
    )
    left, center, right = st.columns([1, 2, 1])
    with center:
        run = st.button("Run Analysis", key="run")

    openai_model        = st.sidebar.selectbox("OpenAI model", ["gpt-4","gpt-4o","gpt-3.5-turbo","gpt-3.5-turbo-16k"], index=0)
    gemini_model_name   = st.sidebar.selectbox("Gemini model", ["gemini-2.5-flash","gemini-2.5-pro"], index=0)
    perplexity_model_name = st.sidebar.selectbox("Perplexity model", ["sonar","sonar-pro"], index=0)

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
                        ("Perplexity", get_perplexity_response)
                    ]:
                        txt = fn(q)
                        results.append({"Query": q, "Source": source, "Response": txt})
                        time.sleep(1)

            df = pd.DataFrame(results)[["Query","Source","Response"]]
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                "responses.csv",
                "text/csv"
            )

with tab2:
    st.markdown("### Search Visibility Analysis")
    uploaded = st.file_uploader("Upload your results CSV", type="csv")

    if uploaded:
        df_main = pd.read_csv(uploaded)

        competitors = ["ROXBOX", "Wilmot", "Pac‑Van", "BMarko", "Giant", "XCaliber", "Conexwest", "Mobile Modular", "WillScot"]
        pattern = re.compile(r'\b(' + '|'.join(re.escape(c) for c in competitors) + r')\b', flags=re.IGNORECASE)

        def extract_competitors(text):
            matches = pattern.findall(text or "")
            found = []
            for m in matches:
                for comp in competitors:
                    if m.lower() == comp.lower():
                        found.append(comp)
            return ", ".join(sorted(set(found)))

        df_main["Competitors Mentioned"] = df_main["Response"].apply(extract_competitors)
        df_main['Branded Query'] = df_main['Query'].str.contains('falcon', case=False, na=False).map({True: 'Y', False: 'N'})
        df_main['Falcon Mentioned'] = df_main['Response'].str.contains('falcon', case=False, na=False).map({True: 'Y', False: 'N'})
        df_main['Sources Cited'] = df_main['Response'].str.findall(r'(https?://\S+)').apply(lambda lst: ', '.join(lst) if lst else '')
        df_main['Response Word-Count'] = df_main['Response'].astype(str).str.split().str.len()
        df_main['Query Number'] = pd.factorize(df_main['Query'])[0] + 1
        df_main = df_main[["Query Number", "Query", "Source", "Response", "Response Word-Count", "Branded Query", "Falcon Mentioned", "Competitors Mentioned", "Sources Cited"]]

        st.subheader("🧹 Cleaned Dataset")
        st.caption("Shows the enriched LLM response dataset with extracted brand mentions, citation links, and other metadata.")
        st.dataframe(df_main, use_container_width=True, height=400)
        st.download_button("Download Cleaned CSV", df_main.to_csv(index=False), "cleaned_responses.csv", "text/csv")

        st.subheader("📊 Mention Rates")
        st.caption("Percentage of responses from each LLM that mention Falcon at least once.")
        overall_rate = df_main.groupby('Source')['Falcon Mentioned'].apply(lambda x: (x == 'Y').mean() * 100).round(1)
        cols = st.columns(len(overall_rate))
        for col, src in zip(cols, overall_rate.index):
            col.metric(f"{src} Mentions Falcon", f"{overall_rate[src]}%")

        mention_rate = df_main.groupby(['Source', 'Branded Query'])['Falcon Mentioned'].apply(lambda x: (x == 'Y').mean() * 100).reset_index(name='Mention Rate (%)')
        pivot = mention_rate.pivot(index='Source', columns='Branded Query', values='Mention Rate (%)').rename(columns={'Y': 'Branded (%)', 'N': 'Non‑Branded (%)'}).round(1)
        st.subheader("📈 Branded vs Non-Branded Mention Rates")
        st.caption("Breakdown of Falcon mention rate in branded vs non-branded queries across LLMs.")
        st.dataframe(pivot.reset_index())

        df_main['Falcon URL Cited'] = df_main['Response'].str.contains(r"https?://(?:www\.)?falconstructures\.com", case=False, regex=True, na=False)
        cit_rate = df_main.groupby("Source")["Falcon URL Cited"].mean().mul(100).round(1).reset_index()
        
        st.subheader("🔗 Falcon URL Citation Rate")
        st.caption("Percentage of responses from each LLM that include a link to falconstructures.com.")
        
        # Smaller, borderless chart
        fig, ax = plt.subplots(figsize=(4.5, 2.8))
        sns.barplot(data=cit_rate, x="Source", y="Falcon URL Cited", palette="Set2", ax=ax)
        
        # Remove top and right border lines (spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Annotate values
        for index, row in cit_rate.iterrows():
            ax.text(index, row["Falcon URL Cited"] + 1, f"{row['Falcon URL Cited']:.1f}%", ha='center')
        
        ax.set_ylabel("Citation Rate (%)")
        ax.set_xlabel("")
        fig.tight_layout()
        
        st.pyplot(fig)

        df_main['sentiment_score'] = df_main['Response'].fillna('').apply(lambda t: ((sia.polarity_scores(t)['compound'] + 1) / 2) * 9 + 1)
        sentiment_df = df_main.groupby("Source")["sentiment_score"].mean().round(1).reset_index()
        st.subheader("💬 Average Sentiment per LLM")
        st.caption("Average sentiment score of responses from each LLM (1 = negative, 10 = positive).")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        sns.barplot(data=sentiment_df, x="Source", y="sentiment_score", palette="Set1", ax=ax2)
        for index, row in sentiment_df.iterrows():
            ax2.text(index, row["sentiment_score"] + 0.1, f"{row['sentiment_score']:.1f}", ha='center')
        ax2.set_ylabel("Avg Sentiment (1–10)")
        st.pyplot(fig2)

        mask = (df_main['Falcon Mentioned'] == 'N') & df_main['Competitors Mentioned'].notna() & (df_main['Competitors Mentioned'].str.strip() != '')
        gaps = df_main[mask][["Source", "Query", "Response", "Competitors Mentioned"]]
        st.subheader("⚠️ Competitor-Only Gaps (No Falcon Mention)")
        st.caption("Cases where one or more competitors are mentioned but Falcon is not.")
        st.dataframe(gaps.reset_index(drop=True), use_container_width=True)

        # ─── New Metric: Word Count Distribution ─────────────────────────────────
        st.subheader("📏 Response Length Distribution")
        st.caption("Histogram showing how long LLM responses are across sources.")
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        sns.histplot(data=df_main, x="Response Word-Count", hue="Source", multiple="stack", palette="pastel", bins=20, ax=ax3)
        ax3.set_xlabel("Word Count")
        ax3.set_ylabel("Number of Responses")
        st.pyplot(fig3)

    else:
        st.info("Please upload the raw CSV to begin analysis.")
