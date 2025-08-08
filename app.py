import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import re
import pandas as pd
import time
import os
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ─── PAGE CONFIG & GLOBAL CSS ─────────────────────────────────────────────────
st.set_page_config(page_title="Falcon Structures LLM Tool", layout="wide")

# Custom CSS
st.markdown("""
<style>
/* Center the tabs */
div[data-baseweb="tab-list"] {
    display: flex !important;
    justify-content: center !important;
}

/* (Your existing tab styles follow…) */
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

# ─── TABS ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Multi-LLM Response Generator", "Search Visibility Analysis", "Time Series Analysis"])

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

        from datetime import datetime
        df_main['Date'] = datetime.today().date()
        df_main["Competitors Mentioned"] = df_main["Response"].apply(extract_competitors)
        df_main['Branded Query'] = df_main['Query'].str.contains('falcon', case=False, na=False).map({True: 'Y', False: 'N'})
        df_main['Falcon Mentioned'] = df_main['Response'].str.contains('falcon', case=False, na=False).map({True: 'Y', False: 'N'})
        df_main['Sources Cited'] = df_main['Response'].str.findall(r'(https?://\S+)').apply(lambda lst: ', '.join(lst) if lst else '')
        df_main['Response Word-Count'] = df_main['Response'].astype(str).str.split().str.len()
        df_main['Query Number'] = pd.factorize(df_main['Query'])[0] + 1
        df_main = df_main[["Date", "Query Number", "Query", "Source", "Response", "Response Word-Count", "Branded Query", "Falcon Mentioned", "Competitors Mentioned", "Sources Cited"]]
        
        st.subheader("🧹 Cleaned Dataset")
        st.dataframe(df_main, use_container_width=True, height=400)
        st.download_button("Download Cleaned CSV", df_main.to_csv(index=False), "cleaned_responses.csv", "text/csv")

        st.divider()
        
        st.subheader("📊 Mention Rates")
        overall_rate = df_main.groupby('Source')['Falcon Mentioned'].apply(lambda x: (x == 'Y').mean() * 100).round(1)
        cols = st.columns(len(overall_rate))
        for col, src in zip(cols, overall_rate.index):
            col.metric(f"{src} Mentions Falcon", f"{overall_rate[src]}%")

        st.caption("Breakdown of Falcon mentions for branded vs. non-branded queries by source.")
        mention_rate = df_main.groupby(['Source', 'Branded Query'])['Falcon Mentioned'].apply(lambda x: (x == 'Y').mean() * 100).reset_index(name='Mention Rate (%)')
        pivot = mention_rate.pivot(index='Source', columns='Branded Query', values='Mention Rate (%)').rename(columns={'Y': 'Branded (%)', 'N': 'Non‑Branded (%)'}).round(1)
        st.dataframe(pivot.reset_index())
        
        st.divider()
        
        # Falcon URL citation detection (corrected regex)
        df_main['Falcon URL Cited'] = df_main['Response'].str.contains(r"https?://(?:www\.)?falconstructures\.com", case=False, regex=True, na=False)
        
        # Citation rate chart
        cit_rate = df_main.groupby("Source")["Falcon URL Cited"].mean().mul(100).round(1)
        st.subheader("🔗 Falcon URL Citation Rate")
        st.caption("Shows how often each source included a link to Falcon’s website in their response.")
        st.bar_chart(cit_rate)

        st.divider()
        
        df_main['sentiment_score'] = df_main['Response'].fillna('').apply(lambda t: ((sia.polarity_scores(t)['compound'] + 1) / 2) * 9 + 1)
        sentiment_df = df_main.groupby("Source")["sentiment_score"].mean().round(1)
        st.subheader("💬 Average Sentiment per LLM")
        st.caption("Calculated sentiment score (1-10 scale) based on tone of responses mentioning Falcon.")
        st.bar_chart(sentiment_df)

        st.divider()
        
        mask = (df_main['Falcon Mentioned'] == 'N') & df_main['Competitors Mentioned'].notna() & (df_main['Competitors Mentioned'].str.strip() != '')
        gaps = df_main[mask][["Source", "Query", "Response", "Competitors Mentioned"]]
        st.subheader("⚠️ Competitor-Only Gaps (No Falcon Mention)")
        st.caption("Cases where one or more competitors are mentioned but Falcon is not.")
        st.dataframe(gaps.reset_index(drop=True), use_container_width=True)

        st.divider()
        
        def compute_brand_share(df, query_type='Y'):
            df_subset = df[df['Branded Query'] == query_type].copy()
        
            brand_mentions = []
        
            for idx, row in df_subset.iterrows():
                brands = []
                if row['Falcon Mentioned'] == 'Y':
                    brands.append("Falcon Structures")
                competitors = row['Competitors Mentioned']
                if pd.notna(competitors) and competitors.strip():
                    brands += [x.strip() for x in competitors.split(",")]
        
                for brand in brands:
                    brand_mentions.append({"Brand": brand, "Source": row["Source"]})
        
            mention_df = pd.DataFrame(brand_mentions)
            overall = mention_df["Brand"].value_counts(normalize=True).mul(100).round(1).rename("Overall Share (%)")
            by_source = mention_df.groupby(["Brand", "Source"]).size().unstack(fill_value=0)
            by_source = by_source.div(by_source.sum(axis=0), axis=1).mul(100).round(1)
            
            full = pd.concat([overall, by_source], axis=1).fillna(0).reset_index().rename(columns={"index": "Brand"})
            return full.sort_values("Overall Share (%)", ascending=False)
        

        
        st.subheader("🏷️ Brand Share — Non‑Branded Queries")
        st.caption("Of all responses to Non‑Branded queries (generic, no “Falcon”), what percentage of brand mentions go to each company?")
        
        # Non-Branded
        nonbranded_df = compute_brand_share(df_main, query_type='N')
        st.dataframe(nonbranded_df, use_container_width=True)

        
        st.divider()
        
        import seaborn as sns
        st.subheader("📈 Response Word Count Distribution")
        st.caption("Visualizes how long the responses are across sources.")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(data=df_main, x="Source", y="Response Word-Count", ax=ax, palette="pastel")
        ax.set_title("Distribution of Response Lengths by Source")
        st.pyplot(fig)

        st.subheader("📄 Daily Summary by Source for Google Sheet")
        st.caption("Use this table to export summary metrics per source per run.")

        st.divider()

        
        import datetime
        
        # Add current date
        today = datetime.date.today()
        
        # Step 1: Branded and Non-Branded flags
        df_main['Branded Query'] = df_main['Query'].str.contains('falcon', case=False, na=False).map({True: 'Y', False: 'N'})
        df_main['Falcon Mentioned'] = df_main['Response'].str.contains('falcon', case=False, na=False).map({True: 'Y', False: 'N'})
        df_main['Falcon URL Cited'] = df_main['Response'].str.contains(r"https?://(?:www\.)?falconstructures\.com", case=False, na=False)
        
        # Step 2: Mention Rate by Source & Query Type
        mention_rates = (
            df_main.groupby(["Source", "Branded Query"])["Falcon Mentioned"]
            .apply(lambda x: (x == 'Y').mean() * 100)
            .unstack()
            .round(1)
            .rename(columns={'Y': 'Branded Mention Rate (%)', 'N': 'Non-Branded Mention Rate (%)'})
        )
        
        # Step 3: URL Citation Rate by Source & Query Type
        citation_rates = (
            df_main.groupby(["Source", "Branded Query"])["Falcon URL Cited"]
            .mean()
            .mul(100)
            .unstack()
            .round(1)
            .rename(columns={True: 'Branded URL Citation Rate (%)', False: 'Non-Branded URL Citation Rate (%)'})
        )
        
        # Step 4: Falcon Brand Share
        def compute_brand_share(df, brand_filter):
            mentions = []
            for source in df['Source'].unique():
                sub_df = df[(df['Source'] == source) & brand_filter]
                total = len(sub_df)
                falcon_mentions = sub_df['Response'].str.contains("falcon", case=False, na=False).sum()
                share = (falcon_mentions / total) * 100 if total > 0 else 0
                mentions.append((source, round(share, 1)))
            return dict(mentions)
        
        branded_share = compute_brand_share(df_main, df_main["Branded Query"] == 'Y')
        nonbranded_share = compute_brand_share(df_main, df_main["Branded Query"] == 'N')
        
        brand_share_df = pd.DataFrame({
            "Falcon Brand Share (Branded)": pd.Series(branded_share),
            "Falcon Brand Share (Non-Branded)": pd.Series(nonbranded_share)
        })
        
        # Step 5: Merge all together
        summary_df = mention_rates.join(citation_rates, how='outer').join(brand_share_df, how='outer')
        summary_df["Date"] = today
        summary_df.reset_index(inplace=True)
        
        # Show in app
        st.subheader("📊 Daily Summary Metrics by Source")
        st.dataframe(summary_df, use_container_width=True)
        
        # Optional: Save to CSV
        st.download_button(
            label="📥 Download Daily Summary CSV",
            data=summary_df.to_csv(index=False),
            file_name=f"Falcon_LLM_Summary_{today}.csv",
            mime="text/csv"
        )
    
    else:
        st.info("Please upload the raw CSV to begin analysis.")
# ─── TAB: TIME SERIES ANALYSIS ──────────────────────────────────────────────


with tab3:
    st.markdown("### 📈 Time Series Analysis")
    st.caption("Track changes in key search visibility metrics over time.")

    # Upload service account key
    json_key = st.file_uploader("Upload your Google Sheets service account key (.json)", type="json")

    if json_key is not None:
        import gspread
        import tempfile
        import pandas as pd
        from oauth2client.service_account import ServiceAccountCredentials
        from gspread_dataframe import get_as_dataframe
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk
        nltk.download('vader_lexicon', quiet=True)

        # Save uploaded key to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            tmp_file.write(json_key.read())
            tmp_file_path = tmp_file.name

        # Authenticate
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(tmp_file_path, scope)
        client = gspread.authorize(creds)

        # Open the Google Sheet and read data
        sheet = client.open("Falcon_Search_Visibility_Data").sheet1
        df_main = get_as_dataframe(sheet).dropna(how='all')
        df_main = df_main.dropna(axis=1, how='all')  # drop empty columns too

        # Ensure Date column is datetime
        df_main['Date'] = pd.to_datetime(df_main['Date'])

        # Setup Sentiment Analyzer
        sia = SentimentIntensityAnalyzer()

        # Falcon Mention Rate by Source & Branded
        mention_rates_ts = (
            df_main.groupby(["Date", "Source", "Branded Query"])["Falcon Mentioned"]
            .apply(lambda x: (x == 'Y').mean() * 100)
            .reset_index(name="Falcon Mention Rate")
        )

        # _________ TEST 
        # Brand Share — Non-Branded Queries Time Series
        st.subheader("📊 Brand Share Over Time (Non-Branded Queries)")
        
        # Step 1: Filter only non-branded queries
        non_branded = df_main[df_main["Branded Query"] == "N"].copy()
        
        # Step 2: Extract brand from the 'Response' using a custom list
        brands = [
            "Falcon Structures", "WillScot", "Conexwest", "BMarko",
            "Giant", "Mobile Modular", "ROXBOX", "XCaliber"
        ]
        
        # Step 3: Assign which brand is mentioned in each row
        def get_brand(text):
            for b in brands:
                if pd.notnull(text) and b.lower() in text.lower():
                    return b
            return None
        
        non_branded["Brand Mentioned"] = non_branded["Response"].apply(get_brand)
        
        # Step 4: Compute overall brand share (regardless of source)
        overall_brand_share = (
            non_branded.groupby(["Date", "Brand Mentioned"])
            .size().div(non_branded.groupby("Date").size(), axis=0).mul(100)
            .reset_index(name="Overall Share (%)")
        )
        
        # Step 5: Compute brand share by source
        brand_share_source = (
            non_branded.groupby(["Date", "Source", "Brand Mentioned"])
            .size().div(non_branded.groupby(["Date", "Source"]).size(), axis=0).mul(100)
            .reset_index(name="Brand Share (%)")
        )
        
        # 🎯 Plot 1: Overall Share by Brand (Time Series)
        overall_pivot = overall_brand_share.pivot(index="Date", columns="Brand Mentioned", values="Overall Share (%)")
        st.line_chart(overall_pivot, use_container_width=True, height=300)
        
        # 🎯 Plots 2-4: Share by Brand & Source (Gemini, OpenAI, Perplexity)
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
        
        for i, src in enumerate(["Gemini", "OpenAI", "Perplexity"]):
            sub = brand_share_source[brand_share_source["Source"] == src]
            pivot = sub.pivot(index="Date", columns="Brand Mentioned", values="Brand Share (%)")
            axs[i].set_title(f"{src} Brand Share")
            pivot.plot(ax=axs[i])
            axs[i].legend().set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)

        
        # _________ TEST 

        # Falcon URL Citation Rate
        df_main["Falcon URL Cited"] = df_main["Sources Cited"].str.contains("falconstructures.com", na=False, case=False)
        citation_rate_ts = (
            df_main.groupby(["Date", "Source"])["Falcon URL Cited"]
            .mean()
            .mul(100)
            .reset_index(name="Citation Rate")
        )

        # Response Word Count
        df_main['Response Word-Count'] = df_main['Response'].astype(str).str.split().str.len()
        word_count_ts = df_main.groupby(["Date", "Source"])["Response Word-Count"].mean().reset_index(name="Avg Word Count")

        # Sentiment Score (rescaled to 1-10)
        df_main['sentiment_score'] = df_main['Response'].fillna('').apply(
            lambda t: ((sia.polarity_scores(t)['compound'] + 1) / 2) * 9 + 1
        )
        sentiment_ts = df_main.groupby(["Date", "Source"])["sentiment_score"].mean().reset_index(name="Avg Sentiment")

        ### PLOTS

        st.subheader("Falcon Mention Rate (Branded & Non-Branded)")
        for bq in ['Y', 'N']:
            label = "Branded Queries" if bq == 'Y' else "Non-Branded Queries"
            st.markdown(f"**{label}**")
            subset = mention_rates_ts[mention_rates_ts["Branded Query"] == bq].pivot(index="Date", columns="Source", values="Falcon Mention Rate")
            st.line_chart(subset, height=250, use_container_width=True)

        st.subheader("Falcon Brand Share Over Time")
        # Falcon Brand Share Over Time (Only Falcon — from previous 'brand_share_source')
        falcon_share_ts = brand_share_source[brand_share_source["Brand Mentioned"] == "Falcon Structures"]
        share_pivot = falcon_share_ts.pivot(index="Date", columns="Source", values="Brand Share (%)")
        st.line_chart(share_pivot, height=250, use_container_width=True)


        st.subheader("Falcon URL Citation Rate Over Time")
        cite_pivot = citation_rate_ts.pivot(index="Date", columns="Source", values="Citation Rate")
        st.line_chart(cite_pivot, height=250, use_container_width=True)

        st.subheader("Average Response Word Count")
        wc_pivot = word_count_ts.pivot(index="Date", columns="Source", values="Avg Word Count")
        st.line_chart(wc_pivot, height=250, use_container_width=True)

        st.subheader("Average Sentiment Score (1‑10 Scale)")
        sent_pivot = sentiment_ts.pivot(index="Date", columns="Source", values="Avg Sentiment")
        st.line_chart(sent_pivot, height=250, use_container_width=True)

    else:
        st.warning("⬆️ Please upload your service account `.json` file to begin.")



