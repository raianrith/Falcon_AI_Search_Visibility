import os
import re
import time
import json
import tempfile
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st

# Plotting
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NLP
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Providers
from openai import OpenAI
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_yaml_config(uploaded_file=None, default_path=None):
    """Load YAML config from uploaded file or a default path/env var. Always returns a dict."""
    try:
        import yaml
    except ImportError:
        st.error("Missing dependency: pyyaml. Add it to requirements.txt")
        return {}

    try:
        if uploaded_file is not None:
            data = yaml.safe_load(uploaded_file.getvalue())
            return data or {}
        path = default_path or os.getenv("CLIENT_CONFIG_PATH", "client.config.yaml")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                return data or {}
        st.warning(f"Config not found at {path}; using baked-in defaults.")
        return {}
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {}

    if uploaded_file is not None:
        return yaml.safe_load(uploaded_file.getvalue())

    path = default_path or os.getenv("CLIENT_CONFIG_PATH", "client.config.yaml")
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    st.warning(f"Config not found at {path}; using baked-in defaults.")
    return {}

# ──────────────────────────────────────────────────────────────────────────────
# NLTK SETUP
# ──────────────────────────────────────────────────────────────────────────────
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    # Some newer NLTK builds expose punkt_tab; ignore if missing
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")

sia = SentimentIntensityAnalyzer()

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Search Visibility Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR: LOAD CONFIG & OVERRIDES
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🧭 Client Configuration")
config_upload = st.sidebar.file_uploader("Upload client.config.yaml (optional)", type=["yaml", "yml"]) 
cfg = load_yaml_config(config_upload) or {}

# Safe defaults if keys missing
palette = cfg.get("palette", {})
PRIMARY = palette.get("primary", "#667eea")
SECONDARY = palette.get("secondary", "#764ba2")
HOVER = palette.get("hover", "#e53935")
TEXT_DEFAULT = palette.get("text_default", "#000000")
TEXT_ON_PRIMARY = palette.get("text_on_primary", "#ffffff")

BRAND = st.sidebar.text_input("Brand name", value=cfg.get("brand_name", "Your Brand"))
BRAND_DOMAIN = st.sidebar.text_input("Brand domain", value=cfg.get("brand_domain", "example.com"))
LOGO_URL = st.sidebar.text_input("Logo URL", value=cfg.get("logo_url", ""))
HEADER_TITLE = cfg.get("header_title", f"{BRAND} AI‑Powered LLM Search Visibility Tool")
HEADER_SUBTITLE = cfg.get("header_subtitle", "Enhanced Analytics & Competitive Intelligence")
HEADER_BYLINE_HTML = cfg.get("header_byline_html", "")

COMPETITORS = st.sidebar.text_area(
    "Competitors (one per line)",
    value="\n".join(cfg.get("competitors", []))
).splitlines()
COMPETITORS = [c.strip() for c in COMPETITORS if c.strip()]

QUERY_TEMPLATES = cfg.get("query_templates", {})
THRESHOLDS = cfg.get("thresholds", {
    "mention_rate_warn": 50,
    "first_third_warn": 30,
    "positive_context_warn": 60,
    "nonbranded_mention_warn": 20,
    "competitor_density_warn": 3,
})

models_cfg = cfg.get("models", {})

# ──────────────────────────────────────────────────────────────────────────────
# CSS (uses palette)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Tabs */
div[data-baseweb="tab-list"] {{
  display: flex !important; justify-content: center !important;
}}

div[data-baseweb="tab-list"] button[role="tab"] {{
  background-color: #fff !important; color: {TEXT_DEFAULT} !important;
  border: 1px solid transparent; border-radius: 4px 4px 0 0;
  padding: 0.5rem 1rem; margin: 0; position: relative;
}}

div[data-baseweb="tab-list"] button[role="tab"]:not(:last-child)::after {{
  content: "|"; position: absolute; right: -10px; top: 50%; transform: translateY(-50%);
  color: {TEXT_DEFAULT};
}}

div[data-baseweb="tab-list"] button[role="tab"]:hover {{
  background-color: {HOVER} !important; color: #fff !important;
}}

div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {{
  border-color: #888 !important; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  background-color: #fff !important; color: {TEXT_DEFAULT} !important;
}}

/* Center buttons */
div.stButton > button {{ margin: 0 auto; display: block; }}

/* Metric cards */
.metric-card {{
  background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
  padding: 1rem; border-radius: 10px; color: {TEXT_ON_PRIMARY};
  text-align: center; margin: 0.5rem 0;
}}
.metric-value {{ font-size: 2rem; font-weight: bold; }}
.metric-label {{ font-size: 0.9rem; opacity: 0.9; }}

/* Template cards */
.template-card {{ border: 1px solid #e1e5e9; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; background: #f8f9fa; }}
.template-title {{ font-weight: bold; color: #2c3e50; margin-bottom: 0.5rem; }}

.position-indicator {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 15px; font-size: 0.8rem; font-weight: bold; }}
.position-first {{ background: #28a745; color: white; }}
.position-middle {{ background: #ffc107; color: black; }}
.position-last {{ background: #dc3545; color: white; }}
.position-none {{ background: #6c757d; color: white; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
header_logo_html = f"<img src='{LOGO_URL}' width='60'/>" if LOGO_URL else ""
st.markdown(f"""
<div style='text-align:center; padding:1rem 0;'>
  {header_logo_html}
  <h1>{HEADER_TITLE}</h1>
  <h4 style='color:#aaa;'>{HEADER_SUBTITLE}</h4>
  <p style='color:#999; font-size:0.9rem;'>{HEADER_BYLINE_HTML}</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🛠️ Model Configuration")
openai_model = st.sidebar.selectbox("OpenAI model", [
    models_cfg.get("openai", "gpt-4o"), "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
], index=0)

gemini_model_name = st.sidebar.selectbox("Gemini model", [
    models_cfg.get("gemini", "gemini-2.5-flash"), "gemini-2.5-pro"
], index=0)

perplexity_model_name = st.sidebar.selectbox("Perplexity model", [
    models_cfg.get("perplexity", "sonar"), "sonar-pro"
], index=0)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Advanced Settings")
max_workers = st.sidebar.slider("Parallel Processing Workers", 3, 12, 6)
delay_between_requests = st.sidebar.slider("Delay Between Requests (seconds)", 0.0, 2.0, 0.1)

st.sidebar.subheader("📦 Batch Processing")
batch_size = st.sidebar.number_input("Batch Size", 1, 50, 10)
enable_pause_resume = st.sidebar.checkbox("Enable Pause/Resume", value=True)

# API keys (secrets first, then env)
openai_key = st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
gemini_key = st.secrets.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
perp_key = st.secrets.get("perplexity_api_key") or os.getenv("PERPLEXITY_API_KEY")

openai_client = OpenAI(api_key=openai_key) if openai_key else None
genai.configure(api_key=gemini_key) if gemini_key else None
gemini_model = genai.GenerativeModel(gemini_model_name) if gemini_key else None
perplexity_client = OpenAI(api_key=perp_key, base_url="https://api.perplexity.ai") if perp_key else None

SYSTEM_PROMPT = "Provide a helpful answer to the user's query."

# ──────────────────────────────────────────────────────────────────────────────
# PROVIDER WRAPPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_openai_response(q:str):
    if openai_client is None:
        return "ERROR: OpenAI key not configured"
    try:
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)
        r = openai_client.chat.completions.create(
            model=openai_model,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def get_gemini_response(q:str):
    if gemini_model is None:
        return "ERROR: Gemini key not configured"
    try:
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)
        r = gemini_model.generate_content(q)
        return r.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        return f"ERROR: {e}"


def get_perplexity_response(q:str):
    if perplexity_client is None:
        return "ERROR: Perplexity key not configured"
    try:
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)
        r = perplexity_client.chat.completions.create(
            model=perplexity_model_name,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def safe_sentence_tokenize(text):
    try:
        return nltk.sent_tokenize(str(text))
    except Exception:
        sentences = re.split(r'[.!?]+', str(text))
        return [s.strip() for s in sentences if s.strip()]


def analyze_position(text, brand:str):
    if not text or pd.isna(text):
        return "Not Mentioned", 0, "N/A"
    sentences = safe_sentence_tokenize(str(text))
    if not sentences:
        return "Not Mentioned", 0, "N/A"
    total = len(sentences)
    for i, s in enumerate(sentences):
        if brand.lower() in s.lower():
            pct = (i + 1) / total
            if pct <= 0.33:
                bucket = "First Third"
            elif pct <= 0.66:
                bucket = "Middle Third"
            else:
                bucket = "Last Third"
            return bucket, i + 1, f"{pct:.1%}"
    return "Not Mentioned", 0, "N/A"


def analyze_context(text, brand:str):
    if not text or pd.isna(text) or brand.lower() not in str(text).lower():
        return "Not Mentioned", 0, []
    sentences = safe_sentence_tokenize(str(text))
    contexts = []
    for s in sentences:
        if brand.lower() in s.lower():
            sent = sia.polarity_scores(s)['compound']
            ctx = "Positive" if sent >= 0.1 else ("Negative" if sent <= -0.1 else "Neutral")
            contexts.append({'sentence': s, 'sentiment': sent, 'context': ctx})
    if contexts:
        avg = float(np.mean([c['sentiment'] for c in contexts]))
        # Pick the first mention's context as summary label
        return contexts[0]['context'], avg, contexts
    return "Neutral", 0, []


def extract_competitors_detailed(text:str):
    if not text or pd.isna(text):
        return [], {}
    # Build regex from COMPETITORS; allow simple word boundaries
    pattern = re.compile(r"\b(" + "|".join(re.escape(c) for c in COMPETITORS) + r")\b", re.IGNORECASE)
    matches = pattern.finditer(str(text))
    found, positions = [], {}
    sentences = safe_sentence_tokenize(str(text))
    for m in matches:
        comp = m.group(1)
        # Normalize to canonical case from list
        for c in COMPETITORS:
            if comp.lower() == c.lower():
                comp = c
                break
        if comp not in found:
            found.append(comp)
            # find sentence index
            for i, s in enumerate(sentences):
                if comp.lower() in s.lower():
                    positions[comp] = i + 1
                    break
    return found, positions

# ──────────────────────────────────────────────────────────────────────────────
# BATCH / PARALLEL
# ──────────────────────────────────────────────────────────────────────────────
class BatchState:
    def __init__(self):
        if 'batch_state' not in st.session_state:
            st.session_state.batch_state = {
                'is_running': False, 'is_paused': False,
                'current_batch': 0, 'total_batches': 0,
                'results': [], 'queries': []
            }
    def pause(self): st.session_state.batch_state['is_paused'] = True
    def resume(self): st.session_state.batch_state['is_paused'] = False
    def stop(self):
        st.session_state.batch_state.update({'is_running': False, 'is_paused': False})

batch_state = BatchState()


def get_response_with_source(tup):
    source, func, query = tup
    t0 = time.time()
    try:
        resp = func(query)
        dt = round(time.time() - t0, 2)
        return {"Query": query, "Source": source, "Response": resp, "Response_Time": dt, "Timestamp": datetime.now().isoformat()}
    except Exception as e:
        dt = round(time.time() - t0, 2)
        return {"Query": query, "Source": source, "Response": f"ERROR: {e}", "Response_Time": dt, "Timestamp": datetime.now().isoformat()}


def process_queries_parallel(queries):
    tasks = []
    for q in queries:
        tasks.extend([
            ("OpenAI", get_openai_response, q),
            ("Gemini", get_gemini_response, q),
            ("Perplexity", get_perplexity_response, q),
        ])
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(get_response_with_source, tasks))

# ──────────────────────────────────────────────────────────────────────────────
# PROMPT VARIANTS
# ──────────────────────────────────────────────────────────────────────────────

def generate_prompt_suggestions(base_query: str, brand: str):
    return {
        "Brand-Focused": [
            f"What are the benefits of {base_query} from {brand}?",
            f"How does {brand} handle {base_query}?",
            f"{brand} solutions for {base_query}",
        ],
        "Comparison-Focused": [
            f"Compare top companies for {base_query}",
            f"Best alternatives for {base_query} including {brand}",
            f"{brand} vs competitors for {base_query}",
        ],
        "Problem-Solution": [
            f"How to solve {base_query} challenges",
            f"What's the best approach to {base_query}",
            f"Professional solutions for {base_query}",
        ],
        "Location-Specific": [
            f"{base_query} companies in the United States",
            f"Local providers of {base_query}",
            f"Regional specialists in {base_query}",
        ],
    }

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Multi-LLM Response Generator",
    "Search Visibility Analysis",
    "Competitor Comparison",
    "Executive Dashboard",
    "Time Series Analysis",
])

# TAB 1
with tab1:
    st.markdown(
        '<h5 style="text-align:center; margin-bottom:1rem; color:#a9a9a9">'
        'Generate and analyze responses from OpenAI, Gemini, & Perplexity with enhanced analytics'
        '</h5>',
        unsafe_allow_html=True,
    )

    # Query Templates
    with st.expander("📋 Query Templates", expanded=False):
        if QUERY_TEMPLATES:
            category = st.selectbox("Choose Category:", list(QUERY_TEMPLATES.keys()))
            cols = st.columns(2)
            for i, template in enumerate(QUERY_TEMPLATES[category]):
                col = cols[i % 2]
                with col:
                    st.markdown(f"""
                    <div class="template-card"> <div class="template-title">Query {i+1}</div>
                    <div>{template}</div> </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Use Template {i+1}", key=f"tpl_{category}_{i}"):
                        st.session_state.template_query = template
        else:
            st.info("No query templates found in config. Add them under query_templates.")

    # A/B Suggestions
    with st.expander("🧪 A/B Testing & Prompt Engineering", expanded=False):
        ab_base = st.text_input("Base query for A/B testing:", placeholder="e.g., modular office solutions")
        if ab_base:
            sugg = generate_prompt_suggestions(ab_base, BRAND)
            for cat, prompts in sugg.items():
                st.markdown(f"**{cat} Variations:**")
                for p in prompts:
                    st.markdown(f"• {p}")
                st.markdown("")

    # First-visit hints
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
        st.info("💡 Try a few example queries — or use templates from the expander above.")

    # Input
    initial_value = st.session_state.get('template_query', '')
    queries_input = st.text_area("Queries (one per line)", value=initial_value, height=200)

    c1, c2, c3 = st.columns([1,2,1])
    with c1:
        if st.button("🔍 Run Analysis", type="primary"):
            st.session_state.run_triggered = True
    with c2:
        if enable_pause_resume and st.session_state.batch_state['is_running']:
            if st.session_state.batch_state['is_paused']:
                if st.button("▶️ Resume"): batch_state.resume()
            else:
                if st.button("⏸️ Pause"): batch_state.pause()
    with c3:
        if st.session_state.batch_state['is_running']:
            if st.button("⏹️ Stop"): batch_state.stop()

    # Process
    if st.session_state.get('run_triggered', False):
        qs = [q.strip() for q in queries_input.splitlines() if q.strip()]
        if not qs:
            st.warning("Please enter at least one query.")
        else:
            with st.spinner("Gathering responses and running analytics…"):
                t0 = time.time()
                results = process_queries_parallel(qs)
                t1 = time.time()
                st.success(f"✅ Completed {len(results)} API calls in {t1 - t0:.1f} seconds!")

            df = pd.DataFrame(results)
            # Analytics using BRAND
            df['Brand_Position'], df['Brand_Sentence_Num'], df['Brand_Position_Pct'] = zip(*df['Response'].apply(lambda x: analyze_position(x, BRAND)))
            df['Context_Type'], df['Context_Sentiment'], df['Context_Details'] = zip(*df['Response'].apply(lambda x: analyze_context(x, BRAND)))
            comp_data = df['Response'].apply(extract_competitors_detailed)
            df['Competitors_Found'] = [c[0] for c in comp_data]
            df['Competitor_Positions'] = [c[1] for c in comp_data]

            st.subheader("📊 Enhanced Analysis Results")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                mention_rate = (df['Response'].str.contains(BRAND, case=False, na=False).sum() / len(df) * 100)
                st.metric(f"{BRAND} Mention Rate", f"{mention_rate:.1f}%")
            with col2:
                st.metric("Avg Response Time", f"{df['Response_Time'].mean():.1f}s")
            with col3:
                first_third = (df['Brand_Position'] == 'First Third').sum() / len(df) * 100
                st.metric("First Third Mentions", f"{first_third:.1f}%")
            with col4:
                positive = (df['Context_Type'] == 'Positive').sum() / len(df) * 100
                st.metric("Positive Context", f"{positive:.1f}%")

            display_df = df[['Query', 'Source', 'Response', 'Response_Time', 'Brand_Position', 'Context_Type', 'Competitors_Found']].copy()
            st.dataframe(display_df, use_container_width=True, height=400)
            st.download_button("📥 Download Enhanced Results", df.to_csv(index=False), "enhanced_responses.csv", "text/csv")
        st.session_state.run_triggered = False

# TAB 2: Search Visibility Analysis
with tab2:
    st.markdown("### 🔍 Search Visibility Analysis")
    uploaded = st.file_uploader("Upload your results CSV", type="csv", key="visibility_upload")
    if uploaded:
        df_main = pd.read_csv(uploaded)
        df_main['Date'] = pd.to_datetime(df_main.get('Date', datetime.today().date()))

        # BRAND analytics
        df_main['Brand_Position'], df_main['Brand_Sentence_Num'], df_main['Brand_Position_Pct'] = zip(*df_main['Response'].apply(lambda x: analyze_position(x, BRAND)))
        df_main['Context_Type'], df_main['Context_Sentiment'], df_main['Context_Details'] = zip(*df_main['Response'].apply(lambda x: analyze_context(x, BRAND)))
        comp_data = df_main['Response'].apply(extract_competitors_detailed)
        df_main['Competitors_Found'] = [', '.join(c[0]) for c in comp_data]
        df_main['Competitor_Positions'] = [c[1] for c in comp_data]

        df_main['Branded_Query'] = df_main['Query'].astype(str).str.contains(BRAND, case=False, na=False).map({True:'Y', False:'N'})
        df_main['Brand_Mentioned'] = df_main['Response'].astype(str).str.contains(BRAND, case=False, na=False).map({True:'Y', False:'N'})
        df_main['Sources_Cited'] = df_main['Response'].astype(str).str.findall(r'(https?://\S+)').apply(lambda lst: ', '.join(lst) if lst else '')
        df_main['Response_Word_Count'] = df_main['Response'].astype(str).str.split().str.len()
        df_main['Query_Number'] = pd.factorize(df_main['Query'])[0] + 1
        df_main['Brand_URL_Cited'] = df_main['Sources_Cited'].str.contains(BRAND_DOMAIN, case=False, na=False)

        st.subheader("📍 Position Analysis")
        c1, c2 = st.columns(2)
        with c1:
            cnt = df_main['Brand_Position'].value_counts()
            fig = px.pie(values=cnt.values, names=cnt.index, title=f"{BRAND} Mention Position Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            by_src = df_main.groupby(['Source','Brand_Position']).size().unstack(fill_value=0)
            fig = px.bar(by_src, title="Position Distribution by Source", barmode='stack')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("💭 Context Analysis")
        c1, c2 = st.columns(2)
        with c1:
            ctx = df_main['Context_Type'].value_counts()
            fig = px.bar(x=ctx.index, y=ctx.values, title="Context Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ctx_src = df_main.groupby(['Source','Context_Type']).size().unstack(fill_value=0)
            fig = px.bar(ctx_src, title="Context Distribution by Source", barmode='group')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧹 Enhanced Dataset")
        cols = ["Date","Query_Number","Query","Source","Response","Response_Word_Count","Branded_Query","Brand_Mentioned","Brand_Position","Context_Type","Context_Sentiment","Competitors_Found","Sources_Cited","Brand_URL_Cited"]
        st.dataframe(df_main[cols], use_container_width=True, height=400)
        st.download_button("📥 Download Enhanced Analysis", df_main[cols].to_csv(index=False), "enhanced_visibility_analysis.csv", "text/csv")

        st.divider()
        st.subheader("📊 Traditional Mention Rates")
        overall = df_main.groupby('Source')['Brand_Mentioned'].apply(lambda x: (x=='Y').mean()*100).round(1)
        cols = st.columns(len(overall))
        for col, src in zip(cols, overall.index):
            col.metric(f"{src} mentions {BRAND}", f"{overall[src]}%")

# TAB 3: Competitor Comparison
with tab3:
    st.markdown("### 🏆 Competitor Comparison")
    selected = st.multiselect("Select Competitors to Compare:", [BRAND] + COMPETITORS, default=[BRAND] + COMPETITORS[:3])
    comparison_queries = st.text_area("Comparison Queries (one per line):", height=150)
    if st.button("🔍 Run Competitor Analysis"):
        if comparison_queries.strip():
            queries = [q.strip() for q in comparison_queries.splitlines() if q.strip()]
            with st.spinner("Running competitor comparison analysis…"):
                dfc = pd.DataFrame(process_queries_parallel(queries))
            comp_stats = {}
            for comp in selected:
                mentions = dfc['Response'].astype(str).str.contains(comp, case=False, na=False)
                positions = dfc['Response'].apply(lambda x: analyze_position(x, comp))
                contexts = dfc['Response'].apply(lambda x: analyze_context(x, comp))
                comp_stats[comp] = {
                    'Mention Rate (%)': (mentions.sum()/len(dfc)*100) if len(dfc) else 0.0,
                    'Avg Position': (sum([p[1] for p in positions if p[1]>0]) / max(sum([1 for p in positions if p[1]>0]),1)),
                    'Positive Context (%)': (sum([1 for c in contexts if c[0]=='Positive'])/len(dfc)*100) if len(dfc) else 0.0,
                    'First Third (%)': (sum([1 for p in positions if p[0]=='First Third'])/len(dfc)*100) if len(dfc) else 0.0,
                }
            table = pd.DataFrame(comp_stats).T.round(1)
            st.dataframe(table, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(x=table.index, y=table['Mention Rate (%)'], title="Mention Rate Comparison", labels={'x':'Competitor','y':'Mention Rate (%)'})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(x=table.index, y=table['First Third (%)'], title="First Third Position Rate", labels={'x':'Competitor','y':'First Third (%)'})
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("📄 Side-by-Side Response Analysis")
            for i, q in enumerate(queries):
                with st.expander(f"Query {i+1}: {q}"):
                    rows = dfc[dfc['Query'] == q]
                    cols = st.columns(len(rows))
                    for col, (_, r) in zip(cols, rows.iterrows()):
                        with col:
                            st.markdown(f"**{r['Source']}**")
                            txt = str(r['Response'])
                            for comp in selected:
                                if comp.lower() in txt.lower():
                                    txt = re.sub(fr"(?i){re.escape(comp)}", f"**{comp}**", txt)
                            preview = (txt[:500] + "…") if len(txt) > 500 else txt
                            st.markdown(preview)
                            # quick badges
                            for comp in selected:
                                if comp.lower() in str(r['Response']).lower():
                                    pos = analyze_position(r['Response'], comp)
                                    ctx = analyze_context(r['Response'], comp)
                                    klass = "position-first" if pos[0]=="First Third" else ("position-middle" if pos[0]=="Middle Third" else ("position-last" if pos[0]=="Last Third" else "position-none"))
                                    st.markdown(f"<span class='position-indicator {klass}'>{comp}: {pos[0]} | {ctx[0]}</span>", unsafe_allow_html=True)

# TAB 4: Executive Dashboard
with tab4:
    st.markdown("### 📈 Executive Dashboard")
    st.markdown("*Comprehensive overview of search visibility performance*")
    dash_file = st.file_uploader("Upload analysis results for dashboard", type="csv", key="dashboard_upload")
    if dash_file:
        df = pd.read_csv(dash_file)
        if 'Response' in df.columns:
            df['Date'] = pd.to_datetime(df.get('Date', datetime.today().date()))
            df['Brand_Mentioned'] = df['Response'].astype(str).str.contains(BRAND, case=False, na=False)
            df['Branded_Query'] = df.get('Query', "").astype(str).str.contains(BRAND, case=False, na=False)
            pos_data = df['Response'].apply(lambda x: analyze_position(x, BRAND))
            ctx_data = df['Response'].apply(lambda x: analyze_context(x, BRAND))
            comp_data = df['Response'].apply(extract_competitors_detailed)
            df['Position_Category'] = [p[0] for p in pos_data]
            df['Context_Type'] = [c[0] for c in ctx_data]
            df['Context_Sentiment'] = [c[1] for c in ctx_data]
            df['Competitors_Count'] = [len(c[0]) for c in comp_data]
            df['Brand_URL_Cited'] = df.get('Sources_Cited', "").astype(str).str.contains(BRAND_DOMAIN, case=False, na=False)

            st.subheader("🎯 Key Performance Indicators")
            c1,c2,c3,c4,c5 = st.columns(5)
            total = len(df)
            mention_rate = (df['Brand_Mentioned'].sum()/total*100) if total else 0
            first_rate = (sum([1 for p in pos_data if p[0]=='First Third'])/total*100) if total else 0
            pos_rate = (sum([1 for c in ctx_data if c[0]=='Positive'])/total*100) if total else 0
            avg_comp = df['Competitors_Count'].mean() if total else 0
            nonbrand_rate = (df[~df['Branded_Query']]['Brand_Mentioned'].mean()*100) if (~df['Branded_Query']).any() else 0

            with c1: st.markdown(f"<div class='metric-card'><div class='metric-value'>{mention_rate:.1f}%</div><div class='metric-label'>Overall Mention Rate</div></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='metric-card'><div class='metric-value'>{first_rate:.1f}%</div><div class='metric-label'>First Third Position</div></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='metric-card'><div class='metric-value'>{pos_rate:.1f}%</div><div class='metric-label'>Positive Context</div></div>", unsafe_allow_html=True)
            with c4: st.markdown(f"<div class='metric-card'><div class='metric-value'>{nonbrand_rate:.1f}%</div><div class='metric-label'>Non‑Branded Mentions</div></div>", unsafe_allow_html=True)
            with c5: st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_comp:.1f}</div><div class='metric-label'>Avg Competitors/Query</div></div>", unsafe_allow_html=True)

            st.divider()
            st.subheader("🏅 Performance by LLM Source")
            perf = df.groupby('Source').agg({
                'Brand_Mentioned': lambda x: (x.sum()/len(x)*100),
                'Position_Category': lambda x: sum([1 for p in x if p=='First Third'])/len(x)*100,
                'Context_Type': lambda x: sum([1 for c in x if c=='Positive'])/len(x)*100,
                'Context_Sentiment': 'mean',
                'Competitors_Count': 'mean'
            }).round(2)
            perf.columns = ['Mention Rate (%)','First Third (%)','Positive Context (%)','Avg Sentiment','Avg Competitors']

            fig = make_subplots(rows=2, cols=2, subplot_titles=('Mention Rate by Source','Position Performance','Context Analysis','Competitive Landscape'))
            sources = perf.index.tolist()
            fig.add_trace(go.Bar(name='Mention Rate', x=sources, y=perf['Mention Rate (%)']), row=1, col=1)
            fig.add_trace(go.Bar(name='First Third', x=sources, y=perf['First Third (%)']), row=1, col=2)
            fig.add_trace(go.Bar(name='Positive Context', x=sources, y=perf['Positive Context (%)']), row=2, col=1)
            fig.add_trace(go.Bar(name='Avg Competitors', x=sources, y=perf['Avg Competitors']), row=2, col=2)
            fig.update_layout(height=600, showlegend=False, title_text="Comprehensive Performance Analysis")
            st.plotly_chart(fig, use_container_width=True)

            if len(df['Date'].unique()) > 1:
                st.subheader("📊 Performance Trends")
                daily = df.groupby(['Date','Source']).agg({'Brand_Mentioned': lambda x: (x.sum()/len(x)*100)}).reset_index()
                daily.rename(columns={'Brand_Mentioned':'Mention_Rate'}, inplace=True)
                st.plotly_chart(px.line(daily, x='Date', y='Mention_Rate', color='Source', title='Mention Rate Trends Over Time'), use_container_width=True)

            st.subheader("🎯 Opportunity Analysis")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Top Improvement Opportunities:**")
                opp = df[(~df['Brand_Mentioned']) & (df['Competitors_Count']>0)]['Query'].dropna().unique()
                for i, q in enumerate(opp[:5]): st.markdown(f"{i+1}. {q}")
            with c2:
                st.markdown("**Performance Strengths:**")
                strong = df[(df['Brand_Mentioned']) & (df['Position_Category']=='First Third')]['Query'].dropna().unique()
                for i, q in enumerate(strong[:5]): st.markdown(f"{i+1}. {q}")

            st.subheader("💡 Actionable Recommendations")
            recs = []
            if mention_rate < THRESHOLDS['mention_rate_warn']: recs.append("🔴 Increase overall brand visibility across LLMs with targeted content & citations.")
            if first_rate < THRESHOLDS['first_third_warn']: recs.append("🟡 Improve early brand placement (title/intro sentences, value prop upfront).")
            if pos_rate < THRESHOLDS['positive_context_warn']: recs.append("🟡 Address sentiment drivers; strengthen proof points and case studies.")
            if nonbrand_rate < THRESHOLDS['nonbranded_mention_warn']: recs.append("🔴 Boost non‑branded SEO and topical authority to win generic queries.")
            if avg_comp > THRESHOLDS['competitor_density_warn']: recs.append("🟡 Differentiate with niche positioning and clearer category claims.")
            if not recs: recs.append("🟢 Solid performance across metrics; continue current strategy and monitor trends.")
            for r in recs: st.markdown(r)

            st.divider()
            summary = pd.DataFrame({
                'Date':[datetime.now().date()],
                'Total_Queries':[total],
                'Mention_Rate':[mention_rate],
                'First_Position_Rate':[first_rate],
                'Positive_Context_Rate':[pos_rate],
                'NonBranded_Mention_Rate':[nonbrand_rate],
                'Avg_Competitors':[avg_comp],
            })
            st.download_button("📊 Download Executive Summary", summary.to_csv(index=False), f"executive_summary_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

# TAB 5: Time Series Analysis (Google Sheets)
with tab5:
    st.markdown("### 📈 Time Series Analysis")
    st.caption("Track changes in key metrics over time.")
    json_key = st.file_uploader("Upload Google Sheets service account key (.json)", type="json")
    if json_key is not None:
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials
            from gspread_dataframe import get_as_dataframe
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(json_key.read()); key_path = tmp.name
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
            client = gspread.authorize(creds)
            st.divider()
            sheet_name = st.text_input("Google Sheet Name:", value=f"{BRAND}_Search_Visibility_Data")
            if sheet_name and st.button("📊 Load Time Series Data"):
                with st.spinner("Loading data from Google Sheets…"):
                    sheet = client.open(sheet_name).sheet1
                    df_ts = get_as_dataframe(sheet).dropna(how='all')
                    df_ts = df_ts.dropna(axis=1, how='all')
                    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
                    df_ts['Brand_Mentioned'] = df_ts['Response'].astype(str).str.contains(BRAND, case=False, na=False)
                    df_ts['Branded_Query'] = df_ts['Query'].astype(str).str.contains(BRAND, case=False, na=False)
                    pos_data = df_ts['Response'].apply(lambda x: analyze_position(x, BRAND))
                    ctx_data = df_ts['Response'].apply(lambda x: analyze_context(x, BRAND))
                    comp_data = df_ts['Response'].apply(extract_competitors_detailed)
                    df_ts['Position_Category'] = [p[0] for p in pos_data]
                    df_ts['Context_Type'] = [c[0] for c in ctx_data]
                    df_ts['Context_Sentiment'] = [c[1] for c in ctx_data]
                    df_ts['Competitors_Count'] = [len(c[0]) for c in comp_data]
                    df_ts['Brand_URL_Cited'] = df_ts.get('Sources_Cited', "").astype(str).str.contains(BRAND_DOMAIN, case=False, na=False)
                    st.success(f"✅ Loaded {len(df_ts)} records from {len(df_ts['Date'].unique())} dates")

                    st.subheader("📊 Performance Trends")
                    daily = df_ts.groupby(['Date','Source']).agg({
                        'Brand_Mentioned': lambda x: (x.sum()/len(x)*100),
                        'Position_Category': lambda x: sum([1 for p in x if p=='First Third'])/len(x)*100,
                        'Context_Type': lambda x: sum([1 for c in x if c=='Positive'])/len(x)*100,
                        'Context_Sentiment': 'mean',
                        'Brand_URL_Cited': lambda x: (x.sum()/len(x)*100),
                        'Competitors_Count': 'mean'
                    }).reset_index()
                    daily.columns = ['Date','Source','Mention_Rate','First_Position_Rate','Positive_Context_Rate','Avg_Sentiment','Citation_Rate','Avg_Competitors']

                    fig = make_subplots(rows=3, cols=2, subplot_titles=(
                        'Mention Rate Trends','Position Performance','Context Analysis','Citation Rates','Sentiment Trends','Competitive Density'
                    ), vertical_spacing=0.08)
                    for src in daily['Source'].unique():
                        d = daily[daily['Source']==src]
                        fig.add_trace(go.Scatter(x=d['Date'], y=d['Mention_Rate'], name=f'{src} Mention'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=d['Date'], y=d['First_Position_Rate'], name=f'{src} FirstThird'), row=1, col=2)
                        fig.add_trace(go.Scatter(x=d['Date'], y=d['Positive_Context_Rate'], name=f'{src} Positive'), row=2, col=1)
                        fig.add_trace(go.Scatter(x=d['Date'], y=d['Citation_Rate'], name=f'{src} Cite'), row=2, col=2)
                        fig.add_trace(go.Scatter(x=d['Date'], y=d['Avg_Sentiment'], name=f'{src} Sent'), row=3, col=1)
                        fig.add_trace(go.Scatter(x=d['Date'], y=d['Avg_Competitors'], name=f'{src} Comp'), row=3, col=2)
                    fig.update_layout(height=1000, title_text="Comprehensive Time Series Analysis")
                    st.plotly_chart(fig, use_container_width=True)

                    # Downloads
                    st.download_button("📥 Download Time Series Data", daily.to_csv(index=False), f"time_series_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        except Exception as e:
            st.error(f"Error loading Google Sheets data: {e}")
    else:
        st.info("⬆️ Upload a service account JSON to connect Google Sheets (optional).")

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#666; font-size:0.9rem; padding:2rem 0;'>
  <p><strong>LLM Search Visibility Tool</strong></p>
  <p>Advanced Analytics • Competitive Intelligence • Executive Insights</p>
  <p>Powered by OpenAI, Google Gemini, and Perplexity AI</p>
  <p style='font-size:0.8rem; margin-top:1rem;'>Version 2.0 (Client-Config)</p>
</div>
""", unsafe_allow_html=True)
