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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
from datetime import datetime, date
import numpy as np

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # For newer NLTK versions
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")
    # Fallback to basic sentence splitting if NLTK punkt is not available

sia = SentimentIntensityAnalyzer()

# ─── PAGE CONFIG & GLOBAL CSS ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Falcon Structures LLM Tool", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with new styling
st.markdown("""
<style>
/* Center the tabs */
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

/* Button centering */
div.stButton > button {
    margin: 0 auto;
    display: block;
}

/* Executive dashboard styling */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.8;
}

/* Query template styling */
.template-card {
    border: 1px solid #e1e5e9;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f8f9fa;
}

.template-title {
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

.position-indicator {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: bold;
}

.position-first { background: #28a745; color: white; }
.position-middle { background: #ffc107; color: black; }
.position-last { background: #dc3545; color: white; }
.position-none { background: #6c757d; color: white; }
</style>
""", unsafe_allow_html=True)

# ─── QUERY TEMPLATES ─────────────────────────────────────────────────────────
QUERY_TEMPLATES = {
    "Product Discovery": [
        "What companies provide modular container offices in the US?",
        "Best portable office solutions for construction sites",
        "Mobile office rental companies near me",
        "Temporary office buildings for sale",
        "Modular construction office trailers"
    ],
    "Brand Comparison": [
        "Compare modular office companies",
        "ROXBOX vs Falcon Structures office containers",
        "Best alternative to Mobile Modular offices",
        "Pac-Van competitors for portable buildings",
        "Wilmot vs other modular office providers"
    ],
    "Solution Seeking": [
        "How to set up temporary office space on job site",
        "Portable office rental vs purchase decision",
        "Custom modular office building design",
        "Commercial portable building solutions",
        "Quick office setup for remote locations"
    ],
    "Industry Specific": [
        "Construction site office trailer requirements",
        "Healthcare modular building solutions",
        "School portable classroom alternatives",
        "Government modular office buildings",
        "Emergency response portable facilities"
    ],
    "Technical Queries": [
        "Modular office building specifications",
        "Portable office electrical requirements",
        "Climate control in container offices",
        "ADA compliant modular buildings",
        "Insulation options for portable offices"
    ]
}

# ─── COMPETITOR ANALYSIS SETUP ─────────────────────────────────────────────
COMPETITORS = [
    "ROXBOX", "Wilmot", "Pac-Van", "BMarko", "Giant", 
    "XCaliber", "Conexwest", "Mobile Modular", "WillScot"
]

# ─── LOGO & HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:1rem 0;'>
  <img src='https://github.com/raianrith/AI-Client-Research-Tool/blob/main/Weidert_Logo_primary-logomark-antique.png?raw=true' width='60'/>
  <h1>Falcon AI‑Powered LLM Search Visibility Tool</h1>
  <h4 style='color:#ccc;'>Enhanced Analytics & Competitive Intelligence</h4>
  <p style='color:#999; font-size:0.9rem;'>Created by Weidert Group, Inc.</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR: ENHANCED CONFIGURATION ─────────────────────────────────────────
st.sidebar.title("🛠️ Model Configuration")

# Model selection
openai_model = st.sidebar.selectbox(
    "OpenAI model", 
    ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"], 
    index=1
)
gemini_model_name = st.sidebar.selectbox(
    "Gemini model", 
    ["gemini-2.5-flash", "gemini-2.5-pro"], 
    index=0
)
perplexity_model_name = st.sidebar.selectbox(
    "Perplexity model", 
    ["sonar", "sonar-pro"], 
    index=0
)

st.sidebar.divider()

# Advanced settings
st.sidebar.subheader("⚙️ Advanced Settings")
max_workers = st.sidebar.slider("Parallel Processing Workers", 3, 12, 6)
delay_between_requests = st.sidebar.slider("Delay Between Requests (seconds)", 0.0, 2.0, 0.1)

# Batch processing settings
st.sidebar.subheader("📦 Batch Processing")
batch_size = st.sidebar.number_input("Batch Size", 1, 50, 10)
enable_pause_resume = st.sidebar.checkbox("Enable Pause/Resume", value=True)

# ─── API CLIENTS ────────────────────────────────────────────────────────────────
openai_key = st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
gemini_key = st.secrets.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
perp_key = st.secrets.get("perplexity_api_key") or os.getenv("PERPLEXITY_API_KEY")

openai_client = OpenAI(api_key=openai_key)
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel(gemini_model_name)
perplexity_client = OpenAI(api_key=perp_key, base_url="https://api.perplexity.ai")

SYSTEM_PROMPT = "Provide a helpful answer to the user's query."

# ─── ENHANCED API FUNCTIONS ─────────────────────────────────────────────────
def get_openai_response(q):
    try:
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)
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
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)
        r = gemini_model.generate_content(q)
        return r.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return "ERROR"

def get_perplexity_response(q):
    try:
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)
        r = perplexity_client.chat.completions.create(
            model=perplexity_model_name,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Perplexity error: {e}")
        return "ERROR"

# ─── ENHANCED ANALYSIS FUNCTIONS ─────────────────────────────────────────────
def safe_sentence_tokenize(text):
    """Safe sentence tokenization with fallback"""
    try:
        return nltk.sent_tokenize(str(text))
    except:
        # Fallback: simple sentence splitting on periods, exclamations, questions
        import re
        sentences = re.split(r'[.!?]+', str(text))
        return [s.strip() for s in sentences if s.strip()]

def analyze_position(text, brand="Falcon"):
    """Analyze where in the response the brand appears"""
    if not text or pd.isna(text):
        return "Not Mentioned", 0, "N/A"
    
    sentences = safe_sentence_tokenize(str(text))
    total_sentences = len(sentences)
    
    if total_sentences == 0:
        return "Not Mentioned", 0, "N/A"
    
    for i, sentence in enumerate(sentences):
        if brand.lower() in sentence.lower():
            position_pct = (i + 1) / total_sentences
            if position_pct <= 0.33:
                return "First Third", i + 1, f"{position_pct:.1%}"
            elif position_pct <= 0.66:
                return "Middle Third", i + 1, f"{position_pct:.1%}"
            else:
                return "Last Third", i + 1, f"{position_pct:.1%}"
    
    return "Not Mentioned", 0, "N/A"

def analyze_context(text, brand="Falcon"):
    """Analyze the context around brand mentions"""
    if not text or pd.isna(text) or brand.lower() not in text.lower():
        return "Not Mentioned", 0, []
    
    sentences = safe_sentence_tokenize(str(text))
    contexts = []
    
    for sentence in sentences:
        if brand.lower() in sentence.lower():
            # Analyze sentiment of the sentence
            sentiment = sia.polarity_scores(sentence)
            if sentiment['compound'] >= 0.1:
                context_type = "Positive"
            elif sentiment['compound'] <= -0.1:
                context_type = "Negative"
            else:
                context_type = "Neutral"
            
            contexts.append({
                'sentence': sentence,
                'sentiment': sentiment['compound'],
                'context': context_type
            })
    
    if contexts:
        avg_sentiment = np.mean([c['sentiment'] for c in contexts])
        return contexts[0]['context'], avg_sentiment, contexts
    
    return "Neutral", 0, []

def extract_competitors_detailed(text):
    """Enhanced competitor extraction with position tracking"""
    if not text or pd.isna(text):
        return [], {}
    
    pattern = re.compile(r'\b(' + '|'.join(re.escape(c) for c in COMPETITORS) + r')\b', flags=re.IGNORECASE)
    matches = pattern.finditer(str(text))
    
    found_competitors = []
    positions = {}
    
    sentences = safe_sentence_tokenize(str(text))
    
    for match in matches:
        competitor = match.group(1)
        # Normalize competitor name
        for comp in COMPETITORS:
            if competitor.lower() == comp.lower():
                competitor = comp
                break
        
        if competitor not in found_competitors:
            found_competitors.append(competitor)
            
            # Find position
            for i, sentence in enumerate(sentences):
                if competitor.lower() in sentence.lower():
                    positions[competitor] = i + 1
                    break
    
    return found_competitors, positions

# ─── BATCH PROCESSING WITH PAUSE/RESUME ─────────────────────────────────────
class BatchProcessor:
    def __init__(self):
        if 'batch_state' not in st.session_state:
            st.session_state.batch_state = {
                'is_running': False,
                'is_paused': False,
                'current_batch': 0,
                'total_batches': 0,
                'results': [],
                'queries': []
            }
    
    def start_batch_processing(self, queries, batch_size):
        st.session_state.batch_state.update({
            'is_running': True,
            'is_paused': False,
            'current_batch': 0,
            'total_batches': (len(queries) + batch_size - 1) // batch_size,
            'results': [],
            'queries': queries
        })
    
    def pause_processing(self):
        st.session_state.batch_state['is_paused'] = True
    
    def resume_processing(self):
        st.session_state.batch_state['is_paused'] = False
    
    def stop_processing(self):
        st.session_state.batch_state.update({
            'is_running': False,
            'is_paused': False
        })

batch_processor = BatchProcessor()

def get_response_with_source(source_func_tuple):
    """Enhanced helper function with error handling and timing"""
    source, func, query = source_func_tuple
    start_time = time.time()
    try:
        response = func(query)
        end_time = time.time()
        return {
            "Query": query, 
            "Source": source, 
            "Response": response,
            "Response_Time": round(end_time - start_time, 2),
            "Timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        end_time = time.time()
        return {
            "Query": query, 
            "Source": source, 
            "Response": f"ERROR: {e}",
            "Response_Time": round(end_time - start_time, 2),
            "Timestamp": datetime.now().isoformat()
        }

def process_queries_parallel(queries):
    """Enhanced parallel processing with batch support"""
    all_tasks = []
    
    for q in queries:
        all_tasks.extend([
            ("OpenAI", get_openai_response, q),
            ("Gemini", get_gemini_response, q),
            ("Perplexity", get_perplexity_response, q)
        ])
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(get_response_with_source, all_tasks))
    
    return results

# ─── PROMPT ENGINEERING SUGGESTIONS ─────────────────────────────────────────
def generate_prompt_suggestions(base_query):
    """Generate variations of queries for A/B testing"""
    suggestions = {
        "Brand-Focused": [
            f"What are the benefits of {base_query} from Falcon Structures?",
            f"How does Falcon Structures handle {base_query}?",
            f"Falcon Structures solutions for {base_query}"
        ],
        "Comparison-Focused": [
            f"Compare top companies for {base_query}",
            f"Best alternatives for {base_query} including Falcon Structures",
            f"Falcon Structures vs competitors for {base_query}"
        ],
        "Problem-Solution": [
            f"How to solve {base_query} challenges",
            f"What's the best approach to {base_query}",
            f"Professional solutions for {base_query}"
        ],
        "Location-Specific": [
            f"{base_query} companies in the United States",
            f"Local providers of {base_query}",
            f"Regional specialists in {base_query}"
        ]
    }
    return suggestions

# ─── MAIN APPLICATION TABS ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Multi-LLM Response Generator", 
    "Search Visibility Analysis", 
    "Competitor Comparison", 
    "Executive Dashboard",
    "Time Series Analysis"
])

# ─── TAB 1: ENHANCED MULTI-LLM GENERATOR ─────────────────────────────────────
with tab1:
    st.markdown(
        '<h5 style="text-align:center; margin-bottom:1rem; color:#a9a9a9">'
        'Generate and analyze responses from OpenAI, Gemini, & Perplexity with enhanced analytics'
        '</h5>',
        unsafe_allow_html=True
    )
    
    # Query Templates Section
    with st.expander("📋 Query Templates", expanded=False):
        st.markdown("**Select from pre-built query templates or use them as inspiration:**")
        
        selected_category = st.selectbox("Choose Category:", list(QUERY_TEMPLATES.keys()))
        
        cols = st.columns(2)
        for i, template in enumerate(QUERY_TEMPLATES[selected_category]):
            col = cols[i % 2]
            with col:
                st.markdown(f"""
                <div class="template-card">
                    <div class="template-title">Query {i+1}</div>
                    <div>{template}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Use Template {i+1}", key=f"template_{selected_category}_{i}"):
                    st.session_state.template_query = template
    
    # A/B Testing Section
    with st.expander("🧪 A/B Testing & Prompt Engineering", expanded=False):
        ab_base_query = st.text_input("Base query for A/B testing:", 
                                     placeholder="e.g., modular office solutions")
        
        if ab_base_query:
            suggestions = generate_prompt_suggestions(ab_base_query)
            
            for category, prompts in suggestions.items():
                st.markdown(f"**{category} Variations:**")
                for prompt in prompts:
                    st.markdown(f"• {prompt}")
                st.markdown("")
    
    # Example queries for first-time users
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
        example_queries = [
            "What companies provide modular container offices in the US?",
            "Best portable office solutions for construction sites",
            "Compare modular office rental vs purchase options"
        ]
        st.info(f"💡 **First time here?** Try these example queries:\n\n" + 
                "\n".join([f"• {q}" for q in example_queries]))
    
    # Main query input
    initial_value = st.session_state.get('template_query', '')
    queries_input = st.text_area(
        "Queries (one per line)",
        value=initial_value,
        height=200,
        placeholder="e.g. What companies provide modular container offices in the US?\nBest portable office solutions for construction sites"
    )
    
    # Processing options
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🔍 Run Analysis", key="run_analysis", type="primary"):
            st.session_state.run_triggered = True
    
    with col2:
        if enable_pause_resume and st.session_state.batch_state['is_running']:
            if st.session_state.batch_state['is_paused']:
                if st.button("▶️ Resume", key="resume"):
                    batch_processor.resume_processing()
            else:
                if st.button("⏸️ Pause", key="pause"):
                    batch_processor.pause_processing()
    
    with col3:
        if st.session_state.batch_state['is_running']:
            if st.button("⏹️ Stop", key="stop"):
                batch_processor.stop_processing()

    # Process queries
    if st.session_state.get('run_triggered', False):
        qs = [q.strip() for q in queries_input.splitlines() if q.strip()]
        if not qs:
            st.warning("Please enter at least one query.")
        else:
            with st.spinner("Gathering responses with enhanced analytics..."):
                start_time = time.time()
                results = process_queries_parallel(qs)
                end_time = time.time()
                
                st.success(f"✅ Completed {len(results)} API calls in {end_time - start_time:.1f} seconds!")

            # Enhanced results processing
            df = pd.DataFrame(results)
            
            # Add enhanced analytics
            df['Falcon_Position'], df['Falcon_Sentence_Num'], df['Falcon_Position_Pct'] = zip(*df['Response'].apply(lambda x: analyze_position(x, "Falcon")))
            df['Context_Type'], df['Context_Sentiment'], df['Context_Details'] = zip(*df['Response'].apply(lambda x: analyze_context(x, "Falcon")))
            
            # Competitor analysis
            competitor_data = df['Response'].apply(extract_competitors_detailed)
            df['Competitors_Found'] = [comp[0] for comp in competitor_data]
            df['Competitor_Positions'] = [comp[1] for comp in competitor_data]
            
            # Display enhanced results
            st.subheader("📊 Enhanced Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                falcon_mention_rate = (df['Response'].str.contains('falcon', case=False, na=False).sum() / len(df) * 100)
                st.metric("Falcon Mention Rate", f"{falcon_mention_rate:.1f}%")
            
            with col2:
                avg_response_time = df['Response_Time'].mean()
                st.metric("Avg Response Time", f"{avg_response_time:.1f}s")
            
            with col3:
                first_position_rate = (df['Falcon_Position'] == 'First Third').sum() / len(df) * 100
                st.metric("First Third Mentions", f"{first_position_rate:.1f}%")
            
            with col4:
                positive_context_rate = (df['Context_Type'] == 'Positive').sum() / len(df) * 100
                st.metric("Positive Context", f"{positive_context_rate:.1f}%")
            
            # Detailed results table
            display_df = df[['Query', 'Source', 'Response', 'Response_Time', 'Falcon_Position', 'Context_Type', 'Competitors_Found']].copy()
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download enhanced results
            st.download_button(
                "📥 Download Enhanced Results",
                df.to_csv(index=False),
                "enhanced_responses.csv",
                "text/csv"
            )
            
        st.session_state.run_triggered = False

# ─── TAB 2: ENHANCED SEARCH VISIBILITY ANALYSIS ─────────────────────────────
with tab2:
    st.markdown("### 🔍 Enhanced Search Visibility Analysis")
    
    uploaded = st.file_uploader("Upload your results CSV", type="csv", key="visibility_upload")

    if uploaded:
        df_main = pd.read_csv(uploaded)
        
        # Enhanced data processing
        pattern = re.compile(r'\b(' + '|'.join(re.escape(c) for c in COMPETITORS) + r')\b', flags=re.IGNORECASE)
        
        # Add enhanced analytics columns
        df_main['Date'] = pd.to_datetime(df_main.get('Date', datetime.today().date()))
        df_main['Falcon_Position'], df_main['Falcon_Sentence_Num'], df_main['Falcon_Position_Pct'] = zip(*df_main['Response'].apply(lambda x: analyze_position(x, "Falcon")))
        df_main['Context_Type'], df_main['Context_Sentiment'], df_main['Context_Details'] = zip(*df_main['Response'].apply(lambda x: analyze_context(x, "Falcon")))
        
        competitor_data = df_main['Response'].apply(extract_competitors_detailed)
        df_main['Competitors_Found'] = [', '.join(comp[0]) for comp in competitor_data]
        df_main['Competitor_Positions'] = [comp[1] for comp in competitor_data]
        
        # Original columns
        df_main['Branded_Query'] = df_main['Query'].astype(str).str.contains('falcon', case=False, na=False).map({True: 'Y', False: 'N'})
        df_main['Falcon_Mentioned'] = df_main['Response'].astype(str).str.contains('falcon', case=False, na=False).map({True: 'Y', False: 'N'})
        df_main['Sources_Cited'] = df_main['Response'].astype(str).str.findall(r'(https?://\S+)').apply(lambda lst: ', '.join(lst) if lst else '')
        df_main['Response_Word_Count'] = df_main['Response'].astype(str).str.split().str.len()
        df_main['Query_Number'] = pd.factorize(df_main['Query'])[0] + 1
        
        # Position Analysis
        st.subheader("📍 Position Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            position_counts = df_main['Falcon_Position'].value_counts()
            fig = px.pie(values=position_counts.values, names=position_counts.index, 
                        title="Falcon Mention Position Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Position by source
            position_by_source = df_main.groupby(['Source', 'Falcon_Position']).size().unstack(fill_value=0)
            fig = px.bar(position_by_source, title="Position Distribution by Source", 
                        barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        
        # Context Analysis
        st.subheader("💭 Context Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            context_counts = df_main['Context_Type'].value_counts()
            fig = px.bar(x=context_counts.index, y=context_counts.values, 
                        title="Context Type Distribution",
                        color=context_counts.index,
                        color_discrete_map={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Context sentiment by source
            context_by_source = df_main.groupby(['Source', 'Context_Type']).size().unstack(fill_value=0)
            fig = px.bar(context_by_source, title="Context Distribution by Source", 
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced cleaned dataset
        st.subheader("🧹 Enhanced Dataset")
        enhanced_columns = [
            "Date", "Query_Number", "Query", "Source", "Response", "Response_Word_Count",
            "Branded_Query", "Falcon_Mentioned", "Falcon_Position", "Context_Type", 
            "Context_Sentiment", "Competitors_Found", "Sources_Cited"
        ]
        
        display_df = df_main[enhanced_columns]
        st.dataframe(display_df, use_container_width=True, height=400)
        
        st.download_button(
            "📥 Download Enhanced Analysis",
            display_df.to_csv(index=False),
            "enhanced_visibility_analysis.csv",
            "text/csv"
        )
        
        # Rest of the original analysis...
        st.divider()
        
        st.subheader("📊 Traditional Mention Rates")
        overall_rate = df_main.groupby('Source')['Falcon_Mentioned'].apply(lambda x: (x == 'Y').mean() * 100).round(1)
        
        cols = st.columns(len(overall_rate))
        for col, src in zip(cols, overall_rate.index):
            col.metric(f"{src} Mentions Falcon", f"{overall_rate[src]}%")

# ─── TAB 3: COMPETITOR COMPARISON ─────────────────────────────────────────────
with tab3:
    st.markdown("### 🏆 Competitor Comparison Mode")
    
    # Option selection: Upload file or run new analysis
    analysis_mode = st.radio(
        "Choose Analysis Mode:",
        ["Upload Existing Results", "Run New Competitor Analysis"],
        horizontal=True
    )
    
    if analysis_mode == "Upload Existing Results":
        st.markdown("**Upload a responses CSV file generated from the Multi-LLM Response Generator:**")
        
        # File upload for existing results
        competitor_file = st.file_uploader(
            "Select your responses CSV file:",
            type="csv",
            key="competitor_upload",
            help="Upload a CSV file with Query, Source, and Response columns"
        )
        
        if competitor_file:
            try:
                df_comp = pd.read_csv(competitor_file)
                
                # Validate required columns
                required_cols = ['Query', 'Source', 'Response']
                missing_cols = [col for col in required_cols if col not in df_comp.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.info("Required columns: Query, Source, Response")
                else:
                    st.success(f"✅ Loaded {len(df_comp)} responses from {len(df_comp['Query'].unique())} queries")
                    
                    # Show data preview
                    with st.expander("📋 Data Preview", expanded=False):
                        st.dataframe(df_comp.head(10), use_container_width=True)
                    
                    # Competitor selection
                    st.subheader("🎯 Select Competitors to Analyze")
                    selected_competitors = st.multiselect(
                        "Choose competitors to compare:",
                        ["Falcon Structures"] + COMPETITORS,
                        default=["Falcon Structures", "ROXBOX", "Mobile Modular", "WillScot"],
                        help="Select the competitors you want to include in the comparison analysis"
                    )
                    
                    if selected_competitors and st.button("🔍 Analyze Competitor Performance", key="analyze_uploaded"):
                        with st.spinner("Analyzing competitor performance..."):
                            # Analyze mentions for each selected competitor
                            competitor_analysis = {}
                            
                            for competitor in selected_competitors:
                                mentions = df_comp['Response'].astype(str).str.contains(competitor, case=False, na=False)
                                positions = df_comp['Response'].apply(lambda x: analyze_position(x, competitor))
                                contexts = df_comp['Response'].apply(lambda x: analyze_context(x, competitor))
                                
                                competitor_analysis[competitor] = {
                                    'mention_rate': (mentions.sum() / len(df_comp) * 100),
                                    'total_mentions': mentions.sum(),
                                    'avg_position': sum([p[1] for p in positions if p[1] > 0]) / max(sum([1 for p in positions if p[1] > 0]), 1),
                                    'positive_context': sum([1 for c in contexts if c[0] == 'Positive']) / len(df_comp) * 100,
                                    'negative_context': sum([1 for c in contexts if c[0] == 'Negative']) / len(df_comp) * 100,
                                    'first_third_rate': sum([1 for p in positions if p[0] == 'First Third']) / len(df_comp) * 100,
                                    'avg_sentiment': np.mean([c[1] for c in contexts if c[1] != 0])
                                }
                            
                            # Display results
                            display_competitor_analysis(competitor_analysis, df_comp, selected_competitors)
                            
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.info("Please ensure the file is a valid CSV with Query, Source, and Response columns.")
    
    else:  # Run New Competitor Analysis
        st.markdown("**Generate new responses specifically for competitor comparison:**")
        
        # Competitor selection
        selected_competitors = st.multiselect(
            "Select Competitors to Compare:",
            ["Falcon Structures"] + COMPETITORS,
            default=["Falcon Structures", "ROXBOX", "Mobile Modular", "WillScot"]
        )
        
        # Side-by-side comparison queries
        comparison_queries = st.text_area(
            "Comparison Queries (one per line):",
            height=150,
            placeholder="Compare modular office companies\nBest portable building solutions\nModular office rental vs purchase",
            help="Enter queries that are likely to mention multiple competitors"
        )
        
        if st.button("🔍 Run New Competitor Analysis", key="new_competitor_analysis"):
            if comparison_queries.strip():
                queries = [q.strip() for q in comparison_queries.splitlines() if q.strip()]
                
                with st.spinner("Running competitor comparison analysis..."):
                    results = process_queries_parallel(queries)
                    df_comp = pd.DataFrame(results)
                    
                    # Analyze mentions for each selected competitor
                    competitor_analysis = {}
                    
                    for competitor in selected_competitors:
                        mentions = df_comp['Response'].astype(str).str.contains(competitor, case=False, na=False)
                        positions = df_comp['Response'].apply(lambda x: analyze_position(x, competitor))
                        contexts = df_comp['Response'].apply(lambda x: analyze_context(x, competitor))
                        
                        competitor_analysis[competitor] = {
                            'mention_rate': (mentions.sum() / len(df_comp) * 100),
                            'total_mentions': mentions.sum(),
                            'avg_position': sum([p[1] for p in positions if p[1] > 0]) / max(sum([1 for p in positions if p[1] > 0]), 1),
                            'positive_context': sum([1 for c in contexts if c[0] == 'Positive']) / len(df_comp) * 100,
                            'negative_context': sum([1 for c in contexts if c[0] == 'Negative']) / len(df_comp) * 100,
                            'first_third_rate': sum([1 for p in positions if p[0] == 'First Third']) / len(df_comp) * 100,
                            'avg_sentiment': np.mean([c[1] for c in contexts if c[1] != 0])
                        }
                    
                    # Display results
                    display_competitor_analysis(competitor_analysis, df_comp, selected_competitors)
            else:
                st.warning("Please enter at least one comparison query.")

def display_competitor_analysis(competitor_analysis, df_comp, selected_competitors):
    """Display comprehensive competitor analysis results"""
    
    # Performance Matrix
    st.subheader("🏆 Competitor Performance Matrix")
    
    comparison_df = pd.DataFrame(competitor_analysis).T.round(1)
    comparison_df.columns = [
        'Mention Rate (%)', 'Total Mentions', 'Avg Position', 
        'Positive Context (%)', 'Negative Context (%)', 
        'First Third (%)', 'Avg Sentiment'
    ]
    
    # Reorder to put Falcon first if it exists
    if 'Falcon Structures' in comparison_df.index:
        falcon_row = comparison_df.loc[['Falcon Structures']]
        other_rows = comparison_df.drop('Falcon Structures').sort_values('Mention Rate (%)', ascending=False)
        comparison_df = pd.concat([falcon_row, other_rows])
    
    # Color-code the dataframe
    styled_df = comparison_df.style.background_gradient(
        subset=['Mention Rate (%)'], cmap='RdYlGn'
    ).background_gradient(
        subset=['Positive Context (%)'], cmap='RdYlGn'
    ).background_gradient(
        subset=['First Third (%)'], cmap='RdYlGn'
    ).background_gradient(
        subset=['Avg Position'], cmap='RdYlGn_r'
    ).background_gradient(
        subset=['Avg Sentiment'], cmap='RdYlGn'
    ).background_gradient(
        subset=['Negative Context (%)'], cmap='RdYlGn_r'
    ).format({
        'Mention Rate (%)': '{:.1f}%',
        'Positive Context (%)': '{:.1f}%', 
        'Negative Context (%)': '{:.1f}%',
        'First Third (%)': '{:.1f}%',
        'Avg Position': '{:.1f}',
        'Avg Sentiment': '{:.2f}',
        'Total Mentions': '{:.0f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Key Insights
    st.subheader("💡 Key Insights")
    
    # Find top performers
    top_mention = comparison_df['Mention Rate (%)'].idxmax()
    top_position = comparison_df.loc[comparison_df['First Third (%)'].idxmax()]
    top_sentiment = comparison_df.loc[comparison_df['Avg Sentiment'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Most Mentioned",
            top_mention,
            f"{comparison_df.loc[top_mention, 'Mention Rate (%)']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Best Positioning",
            top_position.name,
            f"{top_position['First Third (%)']:.1f}% first mentions"
        )
    
    with col3:
        st.metric(
            "Most Positive Sentiment",
            top_sentiment.name,
            f"{top_sentiment['Avg Sentiment']:.2f} sentiment score"
        )
    
    # Visualization Section
    st.subheader("📊 Performance Visualizations")
    
    # Create comprehensive comparison charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mention Rate Comparison', 'Position Performance', 
                       'Sentiment Analysis', 'Context Breakdown'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    competitors = comparison_df.index.tolist()
    colors = px.colors.qualitative.Set3[:len(competitors)]
    
    # Mention Rate
    fig.add_trace(
        go.Bar(
            name='Mention Rate',
            x=competitors,
            y=comparison_df['Mention Rate (%)'],
            marker_color=colors,
            text=[f"{x:.1f}%" for x in comparison_df['Mention Rate (%)']],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Position Performance
    fig.add_trace(
        go.Bar(
            name='First Third Rate',
            x=competitors,
            y=comparison_df['First Third (%)'],
            marker_color=colors,
            text=[f"{x:.1f}%" for x in comparison_df['First Third (%)']],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Sentiment Analysis
    fig.add_trace(
        go.Bar(
            name='Average Sentiment',
            x=competitors,
            y=comparison_df['Avg Sentiment'],
            marker_color=colors,
            text=[f"{x:.2f}" for x in comparison_df['Avg Sentiment']],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # Context Breakdown (Positive vs Negative)
    fig.add_trace(
        go.Bar(
            name='Positive Context',
            x=competitors,
            y=comparison_df['Positive Context (%)'],
            marker_color='lightgreen',
            text=[f"{x:.1f}%" for x in comparison_df['Positive Context (%)']],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            name='Negative Context',
            x=competitors,
            y=comparison_df['Negative Context (%)'],
            marker_color='lightcoral',
            text=[f"{x:.1f}%" for x in comparison_df['Negative Context (%)']],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Comprehensive Competitor Performance Analysis"
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Mention Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="First Third Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)
    fig.update_yaxes(title_text="Context Rate (%)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Side-by-side response comparison
    st.subheader("📄 Side-by-Side Response Analysis")
    
    unique_queries = df_comp['Query'].unique()
    
    for i, query in enumerate(unique_queries[:5]):  # Limit to first 5 queries for readability
        with st.expander(f"Query {i+1}: {query}", expanded=False):
            query_responses = df_comp[df_comp['Query'] == query]
            
            # Create columns for each source
            sources = query_responses['Source'].unique()
            cols = st.columns(len(sources))
            
            for col, source in zip(cols, sources):
                with col:
                    st.markdown(f"**{source}**")
                    
                    source_response = query_responses[query_responses['Source'] == source]['Response'].iloc[0]
                    
                    # Highlight competitor mentions
                    highlighted_text = str(source_response)
                    competitor_mentions = []
                    
                    for competitor in selected_competitors:
                        if competitor.lower() in highlighted_text.lower():
                            # Simple highlighting
                            highlighted_text = re.sub(
                                f'({re.escape(competitor)})',
                                f'**{competitor}**',
                                highlighted_text,
                                flags=re.IGNORECASE
                            )
                            competitor_mentions.append(competitor)
                    
                    # Show truncated response
                    if len(highlighted_text) > 500:
                        highlighted_text = highlighted_text[:500] + "..."
                    
                    st.markdown(highlighted_text)
                    
                    # Show competitor metrics for this response
                    if competitor_mentions:
                        st.markdown("**Competitors Found:**")
                        for competitor in competitor_mentions:
                            pos_info = analyze_position(source_response, competitor)
                            context_info = analyze_context(source_response, competitor)
                            
                            # Color coding based on position
                            if pos_info[0] == "First Third":
                                color_class = "🟢"
                            elif pos_info[0] == "Middle Third":
                                color_class = "🟡"
                            elif pos_info[0] == "Last Third":
                                color_class = "🔴"
                            else:
                                color_class = "⚪"
                            
                            # Sentiment emoji
                            sentiment_emoji = "😊" if context_info[0] == "Positive" else "😐" if context_info[0] == "Neutral" else "😞"
                            
                            st.markdown(f"{color_class} {sentiment_emoji} **{competitor}**: {pos_info[0]} | {context_info[0]}")
                    else:
                        st.markdown("*No selected competitors mentioned*")
    
    # Export options
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download detailed results
        detailed_results = []
        for _, row in df_comp.iterrows():
            for competitor in selected_competitors:
                pos_info = analyze_position(row['Response'], competitor)
                context_info = analyze_context(row['Response'], competitor)
                mentioned = competitor.lower() in str(row['Response']).lower()
                
                detailed_results.append({
                    'Query': row['Query'],
                    'Source': row['Source'],
                    'Competitor': competitor,
                    'Mentioned': 'Yes' if mentioned else 'No',
                    'Position': pos_info[0],
                    'Context': context_info[0],
                    'Sentiment_Score': context_info[1]
                })
        
        detailed_df = pd.DataFrame(detailed_results)
        
        st.download_button(
            "📥 Download Detailed Analysis",
            detailed_df.to_csv(index=False),
            f"competitor_analysis_detailed_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    
    with col2:
        st.download_button(
            "📊 Download Performance Matrix",
            comparison_df.to_csv(),
            f"competitor_performance_matrix_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

# ─── TAB 4: EXECUTIVE DASHBOARD ─────────────────────────────────────────────
with tab4:
    st.markdown("### 📈 Executive Dashboard")
    st.markdown("*Comprehensive overview of search visibility performance*")
    
    # File upload for dashboard
    dashboard_file = st.file_uploader("Upload analysis results for dashboard", type="csv", key="dashboard_upload")
    
    if dashboard_file:
        df_dashboard = pd.read_csv(dashboard_file)
        
        # Ensure we have the necessary columns
        if 'Response' in df_dashboard.columns:
            # Process data for dashboard
            df_dashboard['Date'] = pd.to_datetime(df_dashboard.get('Date', datetime.today().date()))
            df_dashboard['Falcon_Mentioned'] = df_dashboard['Response'].astype(str).str.contains('falcon', case=False, na=False)
            df_dashboard['Branded_Query'] = df_dashboard.get('Query', pd.Series()).astype(str).str.contains('falcon', case=False, na=False)
            
            # Enhanced analytics for dashboard
            position_data = df_dashboard['Response'].apply(lambda x: analyze_position(x, "Falcon"))
            context_data = df_dashboard['Response'].apply(lambda x: analyze_context(x, "Falcon"))
            competitor_data = df_dashboard['Response'].apply(extract_competitors_detailed)
            
            df_dashboard['Position_Category'] = [p[0] for p in position_data]
            df_dashboard['Context_Type'] = [c[0] for c in context_data]
            df_dashboard['Context_Sentiment'] = [c[1] for c in context_data]
            df_dashboard['Competitors_Count'] = [len(c[0]) for c in competitor_data]
            
            # Key Performance Indicators
            st.subheader("🎯 Key Performance Indicators")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # Calculate KPIs
            total_queries = len(df_dashboard)
            falcon_mentions = df_dashboard['Falcon_Mentioned'].sum()
            mention_rate = (falcon_mentions / total_queries * 100) if total_queries > 0 else 0
            
            first_position_count = sum([1 for p in position_data if p[0] == 'First Third'])
            first_position_rate = (first_position_count / total_queries * 100) if total_queries > 0 else 0
            
            positive_context_count = sum([1 for c in context_data if c[0] == 'Positive'])
            positive_context_rate = (positive_context_count / total_queries * 100) if total_queries > 0 else 0
            
            avg_competitors = df_dashboard['Competitors_Count'].mean()
            
            branded_queries = df_dashboard['Branded_Query'].sum()
            nonbranded_mention_rate = (
                df_dashboard[~df_dashboard['Branded_Query']]['Falcon_Mentioned'].sum() / 
                len(df_dashboard[~df_dashboard['Branded_Query']]) * 100
            ) if len(df_dashboard[~df_dashboard['Branded_Query']]) > 0 else 0
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{mention_rate:.1f}%</div>
                    <div class="metric-label">Overall Mention Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{first_position_rate:.1f}%</div>
                    <div class="metric-label">First Third Position</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{positive_context_rate:.1f}%</div>
                    <div class="metric-label">Positive Context</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{nonbranded_mention_rate:.1f}%</div>
                    <div class="metric-label">Non-Branded Mentions</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_competitors:.1f}</div>
                    <div class="metric-label">Avg Competitors/Query</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Performance by Source
            st.subheader("🏅 Performance by LLM Source")
            
            source_performance = df_dashboard.groupby('Source').agg({
                'Falcon_Mentioned': lambda x: (x.sum() / len(x) * 100),
                'Position_Category': lambda x: sum([1 for p in x if p == 'First Third']) / len(x) * 100,
                'Context_Type': lambda x: sum([1 for c in x if c == 'Positive']) / len(x) * 100,
                'Context_Sentiment': 'mean',
                'Competitors_Count': 'mean'
            }).round(2)
            
            source_performance.columns = [
                'Mention Rate (%)', 'First Third (%)', 'Positive Context (%)', 
                'Avg Sentiment', 'Avg Competitors'
            ]
            
            # Create a comprehensive performance chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Mention Rate by Source', 'Position Performance', 
                               'Context Analysis', 'Competitive Landscape'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            sources = source_performance.index.tolist()
            
            # Mention Rate
            fig.add_trace(
                go.Bar(name='Mention Rate', x=sources, y=source_performance['Mention Rate (%)'],
                      marker_color='lightblue'),
                row=1, col=1
            )
            
            # Position Performance
            fig.add_trace(
                go.Bar(name='First Third', x=sources, y=source_performance['First Third (%)'],
                      marker_color='lightgreen'),
                row=1, col=2
            )
            
            # Context Analysis
            fig.add_trace(
                go.Bar(name='Positive Context', x=sources, y=source_performance['Positive Context (%)'],
                      marker_color='gold'),
                row=2, col=1
            )
            
            # Competitive Landscape
            fig.add_trace(
                go.Bar(name='Avg Competitors', x=sources, y=source_performance['Avg Competitors'],
                      marker_color='coral'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Comprehensive Performance Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Trends Analysis (if we have date data)
            if len(df_dashboard['Date'].unique()) > 1:
                st.subheader("📊 Performance Trends")
                
                daily_performance = df_dashboard.groupby(['Date', 'Source']).agg({
                    'Falcon_Mentioned': lambda x: (x.sum() / len(x) * 100)
                }).reset_index()
                
                fig = px.line(daily_performance, x='Date', y='Falcon_Mentioned', 
                             color='Source', title='Mention Rate Trends Over Time')
                st.plotly_chart(fig, use_container_width=True)
            
            # Opportunity Analysis
            st.subheader("🎯 Opportunity Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Improvement Opportunities:**")
                
                # Find queries where Falcon wasn't mentioned but competitors were
                opportunity_queries = df_dashboard[
                    (~df_dashboard['Falcon_Mentioned']) & 
                    (df_dashboard['Competitors_Count'] > 0)
                ]['Query'].unique()
                
                for i, query in enumerate(opportunity_queries[:5]):
                    st.markdown(f"{i+1}. {query}")
                    
            with col2:
                st.markdown("**Performance Strengths:**")
                
                # Find queries where Falcon performed well
                strong_queries = df_dashboard[
                    (df_dashboard['Falcon_Mentioned']) & 
                    (df_dashboard['Position_Category'] == 'First Third')
                ]['Query'].unique()
                
                for i, query in enumerate(strong_queries[:5]):
                    st.markdown(f"{i+1}. {query}")
            
            # Actionable Recommendations
            st.subheader("💡 Actionable Recommendations")
            
            recommendations = []
            
            if mention_rate < 50:
                recommendations.append("🔴 **Critical**: Overall mention rate is below 50%. Focus on brand visibility strategies.")
            
            if first_position_rate < 30:
                recommendations.append("🟡 **Important**: Low first-position rate. Optimize content for earlier mentions.")
            
            if positive_context_rate < 60:
                recommendations.append("🟡 **Attention**: Context sentiment needs improvement. Review brand messaging.")
            
            if nonbranded_mention_rate < 20:
                recommendations.append("🔴 **Priority**: Very low non-branded mentions. Strengthen SEO and content strategy.")
            
            if avg_competitors > 3:
                recommendations.append("🟡 **Competitive**: High competitor density. Differentiate value propositions.")
            
            if not recommendations:
                recommendations.append("🟢 **Excellent**: Performance is strong across all metrics. Continue current strategies.")
            
            for rec in recommendations:
                st.markdown(rec)
            
            # Export dashboard summary
            st.divider()
            
            dashboard_summary = {
                'Date': [datetime.now().date()],
                'Total_Queries': [total_queries],
                'Mention_Rate': [mention_rate],
                'First_Position_Rate': [first_position_rate],
                'Positive_Context_Rate': [positive_context_rate],
                'NonBranded_Mention_Rate': [nonbranded_mention_rate],
                'Avg_Competitors': [avg_competitors],
                'Top_Opportunity': [opportunity_queries[0] if len(opportunity_queries) > 0 else 'None'],
                'Performance_Grade': ['A' if mention_rate > 70 else 'B' if mention_rate > 50 else 'C' if mention_rate > 30 else 'D']
            }
            
            summary_df = pd.DataFrame(dashboard_summary)
            
            st.download_button(
                "📊 Download Executive Summary",
                summary_df.to_csv(index=False),
                f"executive_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

# ─── TAB 5: ENHANCED TIME SERIES ANALYSIS ─────────────────────────────────────
with tab5:
    st.markdown("### 📈 Time Series Analysis")
    st.caption("Track changes in key search visibility metrics over time with enhanced analytics.")

    # Upload service account key
    json_key = st.file_uploader("Upload your Google Sheets service account key (.json)", type="json", key="time_series_key")
    
    if json_key is not None:
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials
            from gspread_dataframe import get_as_dataframe
            
            # Save uploaded key to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
                tmp_file.write(json_key.read())
                tmp_file_path = tmp_file.name
            
            # Authenticate
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name(tmp_file_path, scope)
            client = gspread.authorize(creds)

            st.divider()
            
            # Sheet selection
            sheet_name = st.text_input("Google Sheet Name:", value="Falcon_Search_Visibility_Data")
            
            if sheet_name and st.button("📊 Load Time Series Data"):
                with st.spinner("Loading data from Google Sheets..."):
                    try:
                        sheet = client.open(sheet_name).sheet1
                        df_ts = get_as_dataframe(sheet).dropna(how='all')
                        df_ts = df_ts.dropna(axis=1, how='all')
                        
                        # Ensure Date column is datetime
                        df_ts['Date'] = pd.to_datetime(df_ts['Date'])
                        
                        # Enhanced time series processing
                        df_ts['Falcon_Mentioned'] = df_ts['Response'].astype(str).str.contains('falcon', case=False, na=False)
                        df_ts['Branded_Query'] = df_ts['Query'].astype(str).str.contains('falcon', case=False, na=False)
                        
                        # Add enhanced analytics
                        position_data = df_ts['Response'].apply(lambda x: analyze_position(x, "Falcon"))
                        context_data = df_ts['Response'].apply(lambda x: analyze_context(x, "Falcon"))
                        competitor_data = df_ts['Response'].apply(extract_competitors_detailed)
                        
                        df_ts['Position_Category'] = [p[0] for p in position_data]
                        df_ts['Context_Type'] = [c[0] for c in context_data]
                        df_ts['Context_Sentiment'] = [c[1] for c in context_data]
                        df_ts['Competitors_Count'] = [len(c[0]) for c in competitor_data]
                        
                        # URL Citation
                        df_ts["Falcon_URL_Cited"] = df_ts.get("Sources_Cited", pd.Series()).astype(str).str.contains("falconstructures.com", na=False, case=False)
                        
                        st.success(f"✅ Loaded {len(df_ts)} records from {len(df_ts['Date'].unique())} dates")
                        
                        # Historical comparison controls
                        st.subheader("🔍 Historical Comparison Settings")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            date_range = st.selectbox(
                                "Time Period:",
                                ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
                                index=1
                            )
                            
                        with col2:
                            comparison_metric = st.selectbox(
                                "Primary Metric:",
                                ["Mention Rate", "First Position Rate", "Positive Context", "Citation Rate"],
                                index=0
                            )
                            
                        with col3:
                            overlay_comparison = st.checkbox("Enable Period Comparison", value=True)
                        
                        # Filter data based on date range
                        end_date = df_ts['Date'].max()
                        if date_range == "Last 7 days":
                            start_date = end_date - pd.Timedelta(days=7)
                        elif date_range == "Last 30 days":
                            start_date = end_date - pd.Timedelta(days=30)
                        elif date_range == "Last 90 days":
                            start_date = end_date - pd.Timedelta(days=90)
                        else:
                            start_date = df_ts['Date'].min()
                        
                        df_filtered = df_ts[df_ts['Date'] >= start_date].copy()
                        
                        # Enhanced Time Series Visualizations
                        st.subheader("📊 Enhanced Performance Trends")
                        
                        # Multi-metric dashboard
                        daily_metrics = df_filtered.groupby(['Date', 'Source']).agg({
                            'Falcon_Mentioned': lambda x: (x.sum() / len(x) * 100),
                            'Position_Category': lambda x: sum([1 for p in x if p == 'First Third']) / len(x) * 100,
                            'Context_Type': lambda x: sum([1 for c in x if c == 'Positive']) / len(x) * 100,
                            'Context_Sentiment': 'mean',
                            'Falcon_URL_Cited': lambda x: (x.sum() / len(x) * 100),
                            'Competitors_Count': 'mean'
                        }).reset_index()
                        
                        daily_metrics.columns = [
                            'Date', 'Source', 'Mention_Rate', 'First_Position_Rate', 
                            'Positive_Context_Rate', 'Avg_Sentiment', 'Citation_Rate', 'Avg_Competitors'
                        ]
                        
                        # Create comprehensive time series visualization
                        fig = make_subplots(
                            rows=3, cols=2,
                            subplot_titles=(
                                'Mention Rate Trends', 'Position Performance', 
                                'Context Analysis', 'Citation Rates',
                                'Sentiment Trends', 'Competitive Density'
                            ),
                            vertical_spacing=0.08
                        )
                        
                        sources = daily_metrics['Source'].unique()
                        colors = px.colors.qualitative.Set1[:len(sources)]
                        
                        for i, source in enumerate(sources):
                            source_data = daily_metrics[daily_metrics['Source'] == source]
                            
                            # Mention Rate
                            fig.add_trace(
                                go.Scatter(x=source_data['Date'], y=source_data['Mention_Rate'],
                                          name=f'{source} Mention Rate', line=dict(color=colors[i]),
                                          showlegend=True if i == 0 else False),
                                row=1, col=1
                            )
                            
                            # First Position Rate
                            fig.add_trace(
                                go.Scatter(x=source_data['Date'], y=source_data['First_Position_Rate'],
                                          name=f'{source} First Position', line=dict(color=colors[i], dash='dash'),
                                          showlegend=False),
                                row=1, col=2
                            )
                            
                            # Positive Context Rate
                            fig.add_trace(
                                go.Scatter(x=source_data['Date'], y=source_data['Positive_Context_Rate'],
                                          name=f'{source} Positive Context', line=dict(color=colors[i], dash='dot'),
                                          showlegend=False),
                                row=2, col=1
                            )
                            
                            # Citation Rate
                            fig.add_trace(
                                go.Scatter(x=source_data['Date'], y=source_data['Citation_Rate'],
                                          name=f'{source} Citations', line=dict(color=colors[i], dash='dashdot'),
                                          showlegend=False),
                                row=2, col=2
                            )
                            
                            # Sentiment
                            fig.add_trace(
                                go.Scatter(x=source_data['Date'], y=source_data['Avg_Sentiment'],
                                          name=f'{source} Sentiment', line=dict(color=colors[i]),
                                          showlegend=False),
                                row=3, col=1
                            )
                            
                            # Competitors
                            fig.add_trace(
                                go.Scatter(x=source_data['Date'], y=source_data['Avg_Competitors'],
                                          name=f'{source} Competitors', line=dict(color=colors[i]),
                                          showlegend=False),
                                row=3, col=2
                            )
                        
                        fig.update_layout(height=1000, title_text="Comprehensive Time Series Analysis")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Period-over-period comparison
                        if overlay_comparison:
                            st.subheader("📈 Period-over-Period Comparison")
                            
                            # Calculate current vs previous period
                            period_days = (end_date - start_date).days
                            previous_start = start_date - pd.Timedelta(days=period_days)
                            previous_end = start_date
                            
                            current_period = df_ts[df_ts['Date'] >= start_date]
                            previous_period = df_ts[
                                (df_ts['Date'] >= previous_start) & (df_ts['Date'] < previous_end)
                            ]
                            
                            if len(previous_period) > 0:
                                # Calculate metrics for both periods
                                def calculate_period_metrics(data):
                                    return {
                                        'mention_rate': (data['Falcon_Mentioned'].sum() / len(data) * 100) if len(data) > 0 else 0,
                                        'first_position_rate': sum([1 for _, row in data.iterrows() if analyze_position(row['Response'], "Falcon")[0] == 'First Third']) / len(data) * 100,
                                        'positive_context_rate': sum([1 for _, row in data.iterrows() if analyze_context(row['Response'], "Falcon")[0] == 'Positive']) / len(data) * 100,
                                        'avg_sentiment': np.mean([analyze_context(row['Response'], "Falcon")[1] for _, row in data.iterrows()]),
                                    }
                                
                                current_metrics = calculate_period_metrics(current_period)
                                previous_metrics = calculate_period_metrics(previous_period)
                                
                                # Display comparison
                                col1, col2, col3, col4 = st.columns(4)
                                
                                metrics_comparison = [
                                    ("Mention Rate", "mention_rate", "%"),
                                    ("First Position", "first_position_rate", "%"),
                                    ("Positive Context", "positive_context_rate", "%"),
                                    ("Avg Sentiment", "avg_sentiment", "")
                                ]
                                
                                for col, (label, key, suffix) in zip([col1, col2, col3, col4], metrics_comparison):
                                    current_val = current_metrics[key]
                                    previous_val = previous_metrics[key]
                                    change = current_val - previous_val
                                    change_pct = (change / previous_val * 100) if previous_val != 0 else 0
                                    
                                    col.metric(
                                        label,
                                        f"{current_val:.1f}{suffix}",
                                        f"{change:+.1f}{suffix} ({change_pct:+.1f}%)"
                                    )
                        
                        # Data export
                        st.divider()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                "📥 Download Time Series Data",
                                daily_metrics.to_csv(index=False),
                                f"time_series_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                                "text/csv"
                            )
                        
                        with col2:
                            # Generate insights report
                            insights = []
                            
                            latest_data = daily_metrics[daily_metrics['Date'] == daily_metrics['Date'].max()]
                            avg_mention_rate = latest_data['Mention_Rate'].mean()
                            
                            if avg_mention_rate > 70:
                                insights.append("🟢 Strong overall mention rate performance")
                            elif avg_mention_rate > 50:
                                insights.append("🟡 Moderate mention rate - room for improvement")
                            else:
                                insights.append("🔴 Low mention rate - requires immediate attention")
                            
                            # Trend analysis
                            if len(daily_metrics) > 7:
                                recent_trend = daily_metrics.tail(7)['Mention_Rate'].mean()
                                older_trend = daily_metrics.head(7)['Mention_Rate'].mean()
                                
                                if recent_trend > older_trend * 1.1:
                                    insights.append("📈 Positive trend in recent performance")
                                elif recent_trend < older_trend * 0.9:
                                    insights.append("📉 Declining trend detected")
                                else:
                                    insights.append("➡️ Stable performance trend")
                            
                            insights_text = "\n".join(insights)
                            
                            st.download_button(
                                "💡 Download Insights Report",
                                f"Time Series Analysis Insights\n{'='*40}\n{insights_text}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                f"insights_report_{datetime.now().strftime('%Y%m%d')}.txt",
                                "text/plain"
                            )
                        
                    except Exception as e:
                        st.error(f"Error loading Google Sheets data: {e}")
                        st.info("Please ensure the sheet name is correct and you have proper access permissions.")
            
        except ImportError:
            st.error("Required packages for Google Sheets integration are not installed.")
            st.info("Install them with: pip install gspread oauth2client gspread-dataframe")
    
    else:
        st.info("⬆️ Upload your Google Sheets service account JSON file to begin time series analysis.")
        
        # Show sample time series visualization with dummy data
        st.subheader("📊 Sample Time Series Visualization")
        st.caption("This is how your time series analysis will look once you upload your data:")
        
        # Create sample data for demonstration
        sample_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
        sample_data = []
        
        for date in sample_dates:
            for source in ['OpenAI', 'Gemini', 'Perplexity']:
                # Generate realistic sample data with trends
                base_rate = 45 + np.random.normal(0, 10)
                seasonal_factor = 5 * np.sin((date.dayofyear / 365) * 2 * np.pi)
                mention_rate = max(0, min(100, base_rate + seasonal_factor))
                
                sample_data.append({
                    'Date': date,
                    'Source': source,
                    'Mention_Rate': mention_rate,
                    'First_Position_Rate': mention_rate * 0.6 + np.random.normal(0, 5),
                    'Positive_Context_Rate': mention_rate * 0.8 + np.random.normal(0, 3),
                    'Citation_Rate': mention_rate * 0.4 + np.random.normal(0, 8)
                })
        
        sample_df = pd.DataFrame(sample_data)
        
        # Create sample visualization
        fig = px.line(sample_df, x='Date', y='Mention_Rate', color='Source',
                      title='Sample: Falcon Mention Rate Trends Over Time')
        fig.update_layout(yaxis_title="Mention Rate (%)", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ─── ADDITIONAL UTILITY FUNCTIONS ─────────────────────────────────────────────

def generate_executive_summary(df):
    """Generate an automated executive summary"""
    
    total_queries = len(df)
    mention_rate = df['Falcon_Mentioned'].sum() / total_queries * 100 if total_queries > 0 else 0
    
    # Position analysis
    position_data = df['Response'].apply(lambda x: analyze_position(x, "Falcon"))
    first_third_rate = sum([1 for p in position_data if p[0] == 'First Third']) / total_queries * 100
    
    # Context analysis  
    context_data = df['Response'].apply(lambda x: analyze_context(x, "Falcon"))
    positive_context_rate = sum([1 for c in context_data if c[0] == 'Positive']) / total_queries * 100
    
    # Competitive analysis
    competitor_data = df['Response'].apply(extract_competitors_detailed)
    avg_competitors = np.mean([len(c[0]) for c in competitor_data])
    
    # Generate summary text
    summary = f"""
    FALCON STRUCTURES - LLM SEARCH VISIBILITY EXECUTIVE SUMMARY
    ===========================================================
    
    OVERALL PERFORMANCE
    • Total Queries Analyzed: {total_queries:,}
    • Overall Mention Rate: {mention_rate:.1f}%
    • First Position Rate: {first_third_rate:.1f}%
    • Positive Context Rate: {positive_context_rate:.1f}%
    • Average Competitors per Query: {avg_competitors:.1f}
    
    PERFORMANCE GRADE: {'A+' if mention_rate > 80 else 'A' if mention_rate > 70 else 'B+' if mention_rate > 60 else 'B' if mention_rate > 50 else 'C+' if mention_rate > 40 else 'C' if mention_rate > 30 else 'D'}
    
    KEY INSIGHTS
    {'• Excellent brand visibility across all LLMs' if mention_rate > 70 else '• Good brand recognition with room for improvement' if mention_rate > 50 else '• Brand visibility needs significant improvement'}
    {'• Strong positioning when mentioned' if first_third_rate > 50 else '• Moderate positioning performance' if first_third_rate > 30 else '• Poor positioning - often mentioned later in responses'}
    {'• Positive brand sentiment overall' if positive_context_rate > 60 else '• Mixed brand sentiment' if positive_context_rate > 40 else '• Concerning negative sentiment patterns'}
    {'• Highly competitive query landscape' if avg_competitors > 3 else '• Moderately competitive environment' if avg_competitors > 2 else '• Low competitive density - opportunity for dominance'}
    
    TOP RECOMMENDATIONS
    {generate_recommendations(mention_rate, first_third_rate, positive_context_rate, avg_competitors)}
    
    Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
    """
    
    return summary

def generate_recommendations(mention_rate, first_third_rate, positive_context_rate, avg_competitors):
    """Generate specific recommendations based on performance metrics"""
    
    recommendations = []
    
    if mention_rate < 50:
        recommendations.append("1. CRITICAL: Implement comprehensive SEO and content marketing strategy")
        recommendations.append("2. Focus on thought leadership content in modular construction space")
    elif mention_rate < 70:
        recommendations.append("1. Strengthen brand visibility through targeted content optimization")
        recommendations.append("2. Increase industry publication presence and citations")
    
    if first_third_rate < 40:
        recommendations.append("3. Optimize content structure for earlier brand mentions")
        recommendations.append("4. Develop stronger value proposition statements")
    
    if positive_context_rate < 60:
        recommendations.append("5. Review and improve brand messaging and positioning")
        recommendations.append("6. Address any negative sentiment drivers identified")
    
    if avg_competitors > 3:
        recommendations.append("7. Develop competitive differentiation strategy")
        recommendations.append("8. Focus on unique value propositions and market positioning")
    
    if not recommendations:
        recommendations.append("1. Maintain current high-performance strategies")
        recommendations.append("2. Continue monitoring for competitive threats")
        recommendations.append("3. Explore expansion into new query categories")
    
    return "\n    ".join(recommendations)


# ─── SESSION STATE MANAGEMENT ─────────────────────────────────────────────────

# Initialize session state variables
if 'template_query' not in st.session_state:
    st.session_state.template_query = ''

if 'run_triggered' not in st.session_state:
    st.session_state.run_triggered = False

if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True

# ─── FOOTER ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.9rem; padding:2rem 0;'>
    <p><strong>Enhanced Falcon Structures LLM Search Visibility Tool</strong></p>
    <p>Advanced Analytics • Competitive Intelligence • Executive Insights</p>
    <p>Powered by OpenAI, Google Gemini, and Perplexity AI</p>
    <p style='font-size:0.8rem; margin-top:1rem;'>
        Created by <a href='https://www.weidert.com' target='_blank' style='color:#666;'>Weidert Group, Inc.</a>
        | Version 2.0 Enhanced
    </p>
</div>
""", unsafe_allow_html=True)

# ─── HELPFUL TIPS SIDEBAR ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("---")
    st.subheader("💡 Pro Tips")
    
    tips = [
        "**Query Templates**: Use the built-in templates to get started quickly",
        "**A/B Testing**: Try different phrasings of the same query to optimize results", 
        "**Position Tracking**: Monitor where Falcon appears in responses for optimization opportunities",
        "**Context Analysis**: Pay attention to positive vs negative mention contexts",
        "**Competitor Comparison**: Use side-by-side analysis to understand competitive landscape",
        "**Executive Dashboard**: Perfect for stakeholder reporting and strategic planning",
        "**Time Series**: Track performance changes over time to measure improvement"
    ]
    
    for tip in tips:
        st.markdown(f"• {tip}")
    
    st.markdown("---")
    st.subheader("🔧 Support")
    st.markdown("Having issues? Contact the Weidert Group team for technical support.")
    
    # System information
    with st.expander("System Info", expanded=False):
        st.markdown(f"""
        **Current Configuration:**
        - OpenAI Model: {openai_model}
        - Gemini Model: {gemini_model_name}
        - Perplexity Model: {perplexity_model_name}
        - Max Workers: {max_workers}
        - Request Delay: {delay_between_requests}s
        - Batch Size: {batch_size}
        """)

# ─── ERROR HANDLING & LOGGING ─────────────────────────────────────────────────

def log_error(error_type, error_message, query=None):
    """Simple error logging function"""
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] {error_type}: {error_message}"
    if query:
        log_entry += f" | Query: {query}"
    
    # In a production environment, you might want to log to a file or external service
    print(log_entry)  # For now, just print to console

# ─── PERFORMANCE MONITORING ─────────────────────────────────────────────────

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start_timing(self, operation):
        self.start_time = time.time()
        self.metrics[operation] = {'start': self.start_time}
    
    def end_timing(self, operation):
        if operation in self.metrics and 'start' in self.metrics[operation]:
            end_time = time.time()
            duration = end_time - self.metrics[operation]['start']
            self.metrics[operation]['duration'] = duration
            return duration
        return None
    
    def get_summary(self):
        summary = {}
        for operation, data in self.metrics.items():
            if 'duration' in data:
                summary[operation] = f"{data['duration']:.2f}s"
        return summary

# Initialize performance monitor
perf_monitor = PerformanceMonitor()

# ─── FINAL CLEANUP ─────────────────────────────────────────────────────────────

# Clean up temporary files if they exist
import atexit
import os

def cleanup_temp_files():
    # Clean up any temporary files created during execution
    temp_files = getattr(cleanup_temp_files, 'files', [])
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass

cleanup_temp_files.files = []
atexit.register(cleanup_temp_files)
