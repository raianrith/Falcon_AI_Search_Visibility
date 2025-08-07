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

# Tab 1 remains unchanged
# (...code for tab1 here...)

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
        fig, ax = plt.subplots()
        sns.barplot(data=cit_rate, x="Source", y="Falcon URL Cited", palette="Set2", ax=ax)
        for index, row in cit_rate.iterrows():
            ax.text(index, row["Falcon URL Cited"] + 1, f"{row['Falcon URL Cited']:.1f}%", ha='center')
        ax.set_ylabel("Citation Rate (%)")
        st.pyplot(fig)

        df_main['sentiment_score'] = df_main['Response'].fillna('').apply(lambda t: ((sia.polarity_scores(t)['compound'] + 1) / 2) * 9 + 1)
        sentiment_df = df_main.groupby("Source")["sentiment_score"].mean().round(1).reset_index()
        st.subheader("💬 Average Sentiment per LLM")
        st.caption("Average sentiment score of responses from each LLM (1 = negative, 10 = positive).")
        fig2, ax2 = plt.subplots()
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
        fig3, ax3 = plt.subplots()
        sns.histplot(data=df_main, x="Response Word-Count", hue="Source", multiple="stack", palette="pastel", bins=20, ax=ax3)
        ax3.set_xlabel("Word Count")
        ax3.set_ylabel("Number of Responses")
        st.pyplot(fig3)

    else:
        st.info("Please upload the raw CSV to begin analysis.")
