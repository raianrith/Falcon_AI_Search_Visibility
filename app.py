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

# ─── PREDEFINED QUERIES ─────────────────────────────────────────────────────────
PREDEFINED_QUERIES = [
    "Who are the leading U.S. manufacturers of modified shipping-container buildings for industrial sites? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "What companies supply turnkey container workforce housing for energy projects? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Best insulation solutions for container offices in sub-arctic climates -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Typical deployment timeline and installed cost per square foot for modular container offices (2025 data) -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Key permitting steps for shipping container buildings -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Energy-efficiency comparison: insulated container office vs. temporary mobile trailer -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "What OSHA considerations apply to container-based modular restrooms on construction sites? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Options for containerized network equipment shelters with integrated HVAC and power -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Benefits of combining office and storage in one 40-ft 'work-and-store' container unit -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Which manufacturers offer IBC-compliant stackable container structures in North America? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "How can container structures achieve NFPA 101 egress and fire-rating standards? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "How does combining office and storage in a work-and-store unit improve job-site logistics compared with using separate trailers? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Five-year total cost-of-ownership factors when choosing shipping container buildings vs. stick-built offices -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Does a standard shipping container office door width satisfy ADA requirements, and how can it be modified? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "How do insulated container offices from Falcon compare with Mobile Modular trailer offices on energy performance in Alaska winters? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "How do Falcon container living quarters compare with WillScot dorm modules for NFPA 101 life-safety and crew comfort in seismic Zone 4 regions? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "How do storage containers from Falcon and Conexwest differ in ASTM B117 salt-spray endurance for long-term coastal deployments? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "What roof-coating options do Falcon and Roxbox recommend for container offices in high-UV desert climates, and how does each impact 10-year ROI? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Which theft-deterrent lockbox designs from Falcon, Wilmot, and Conexwest most effectively secure job-site storage containers against break-ins? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "How do Falcon container restroom modules compare with Triumph Modular units on LEED v4 water-conservation performance and freeze-protection reliability at −40 °F"
]

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
    "XCaliber", "Conexwest", "Mobile Modular", "WillScot", "Triumph Modular"
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
tab1, tab2, tab3, tab4 = st.tabs([
    "Multi-LLM Response Generator", 
    "Search Visibility Analysis", 
    "Competitor Comparison", 
    "Executive Dashboard & Time Series"
])

# ─── TAB 1: ENHANCED MULTI-LLM GENERATOR ─────────────────────────────────────
with tab1:
    st.markdown(
        '<h5 style="text-align:center; margin-bottom:1rem; color:#a9a9a9">'
        'Generate and analyze responses from OpenAI, Gemini, & Perplexity with enhanced analytics'
        '</h5>',
        unsafe_allow_html=True
    )
    
    # Predefined Query Set Section
    with st.expander("🎯 Predefined Query Set (20 Queries)", expanded=False):
        st.markdown("**Run the complete set of 20 predefined queries with one click:**")
        st.caption("This comprehensive query set covers key industry topics and competitive comparisons")
        
        # Display queries in a scrollable container
        query_display = st.container()
        with query_display:
            for i, query in enumerate(PREDEFINED_QUERIES, 1):
                st.text(f"{i}. {query[:100]}..." if len(query) > 100 else f"{i}. {query}")
        
        if st.button("🚀 Run All 20 Predefined Queries", key="run_predefined", type="primary"):
            st.session_state.use_predefined = True
            st.session_state.run_triggered = True
    
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
    
    # Main query input
    initial_value = st.session_state.get('template_query', '')
    queries_input = st.text_area(
        "Custom Queries (one per line)",
        value=initial_value,
        height=200,
        placeholder="e.g. What companies provide modular container offices in the US?\nBest portable office solutions for construction sites"
    )
    
    # Processing options
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🔍 Run Custom Queries", key="run_analysis", type="primary"):
            st.session_state.use_predefined = False
            st.session_state.run_triggered = True
    
    with col2:
        if enable_pause_resume and st.session_state.batch_state.get('is_running', False):
            if st.session_state.batch_state['is_paused']:
                if st.button("▶️ Resume", key="resume"):
                    batch_processor.resume_processing()
            else:
                if st.button("⏸️ Pause", key="pause"):
                    batch_processor.pause_processing()
    
    with col3:
        if st.session_state.batch_state.get('is_running', False):
            if st.button("⏹️ Stop", key="stop"):
                batch_processor.stop_processing()

    # Process queries
    if st.session_state.get('run_triggered', False):
        # Determine which queries to use
        if st.session_state.get('use_predefined', False):
            qs = PREDEFINED_QUERIES
        else:
            qs = [q.strip() for q in queries_input.splitlines() if q.strip()]
        
        if not qs:
            st.warning("Please enter at least one query or use predefined queries.")
        else:
            with st.spinner(f"Gathering responses for {len(qs)} queries with enhanced analytics..."):
                start_time = time.time()
                results = process_queries_parallel(qs)
                end_time = time.time()
                
                st.success(f"✅ Completed {len(results)} API calls in {end_time - start_time:.1f} seconds!")

            # Enhanced results processing
            df = pd.DataFrame(results)
            
            # Add Date column
            df['Date'] = datetime.now().date()
            
            # Add enhanced analytics
            df['Falcon_Position'], df['Falcon_Sentence_Num'], df['Falcon_Position_Pct'] = zip(*df['Response'].apply(lambda x: analyze_position(x, "Falcon")))
            df['Context_Type'], df['Context_Sentiment'], df['Context_Details'] = zip(*df['Response'].apply(lambda x: analyze_context(x, "Falcon")))
            
            # Competitor analysis
            competitor_data = df['Response'].apply(extract_competitors_detailed)
            df['Competitors_Found'] = [comp[0] for comp in competitor_data]
            df['Competitor_Positions'] = [comp[1] for comp in competitor_data]
            
            # Add additional columns for analysis
            df['Branded_Query'] = df['Query'].str.contains('falcon', case=False, na=False)
            df['Falcon_Mentioned'] = df['Response'].str.contains('falcon', case=False, na=False)
            df['Sources_Cited'] = df['Response'].str.findall(r'(https?://\S+)').apply(lambda lst: ', '.join(lst) if lst else '')
            df['Falcon_URL_Cited'] = df['Sources_Cited'].str.contains('falconstructures.com', case=False, na=False)
            
            # Store in session state for other tabs
            st.session_state.latest_results = df
            
            # Display enhanced results
            st.subheader("📊 Enhanced Analysis Results")
            st.caption("Summary metrics showing overall performance across all queries and sources")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                falcon_mention_rate = (df['Falcon_Mentioned'].sum() / len(df) * 100)
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
            st.subheader("📋 Detailed Results")
            st.caption("Complete dataset with all responses and analytical metrics")
            display_df = df[['Query', 'Source', 'Response', 'Response_Time', 'Falcon_Position', 'Context_Type', 'Competitors_Found']].copy()
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download enhanced results
            st.download_button(
                "📥 Download Enhanced Results (CSV)",
                df.to_csv(index=False),
                "enhanced_responses.csv",
                "text/csv",
                help="Download the complete dataset with all analytics"
            )
            
        st.session_state.run_triggered = False

# ─── TAB 2: ENHANCED SEARCH VISIBILITY ANALYSIS ─────────────────────────────
with tab2:
    st.markdown("### 🔍 Enhanced Search Visibility Analysis")
    
    uploaded = st.file_uploader("Upload your results CSV", type="csv", key="visibility_upload")
    
    # Check if we have results from Tab 1
    if 'latest_results' in st.session_state:
        use_latest = st.checkbox("Use results from Multi-LLM Response Generator", value=True)
        if use_latest:
            df_main = st.session_state.latest_results.copy()
    elif uploaded:
        df_main = pd.read_csv(uploaded)
    else:
        df_main = None

    if df_main is not None:
        # Enhanced data processing
        pattern = re.compile(r'\b(' + '|'.join(re.escape(c) for c in COMPETITORS) + r')\b', flags=re.IGNORECASE)
        
        # Ensure all necessary columns exist
        if 'Date' not in df_main.columns:
            df_main['Date'] = pd.to_datetime(datetime.today().date())
        else:
            df_main['Date'] = pd.to_datetime(df_main['Date'])
        
        if 'Falcon_Position' not in df_main.columns:
            df_main['Falcon_Position'], df_main['Falcon_Sentence_Num'], df_main['Falcon_Position_Pct'] = zip(*df_main['Response'].apply(lambda x: analyze_position(x, "Falcon")))
        
        if 'Context_Type' not in df_main.columns:
            df_main['Context_Type'], df_main['Context_Sentiment'], df_main['Context_Details'] = zip(*df_main['Response'].apply(lambda x: analyze_context(x, "Falcon")))
        
        if 'Competitors_Found' not in df_main.columns:
            competitor_data = df_main['Response'].apply(extract_competitors_detailed)
            df_main['Competitors_Found'] = [', '.join(comp[0]) if comp[0] else '' for comp in competitor_data]
            df_main['Competitor_Positions'] = [comp[1] for comp in competitor_data]
        
        # Ensure other columns exist
        df_main['Branded_Query'] = df_main['Query'].str.contains('falcon', case=False, na=False).map({True: 'Y', False: 'N'})
        df_main['Falcon_Mentioned'] = df_main['Response'].str.contains('falcon', case=False, na=False).map({True: 'Y', False: 'N'})
        
        if 'Sources_Cited' not in df_main.columns:
            df_main['Sources_Cited'] = df_main['Response'].str.findall(r'(https?://\S+)').apply(lambda lst: ', '.join(lst) if lst else '')
        
        df_main['Falcon_URL_Cited'] = df_main['Sources_Cited'].str.contains('falconstructures.com', case=False, na=False)
        df_main['Response_Word_Count'] = df_main['Response'].astype(str).str.split().str.len()
        df_main['Query_Number'] = pd.factorize(df_main['Query'])[0] + 1
        
        # Traditional Mention Rates
        st.subheader("📊 Traditional Mention Rates")
        st.caption("Overall percentage of responses mentioning Falcon by each LLM source")
        overall_rate = df_main.groupby('Source')['Falcon_Mentioned'].apply(lambda x: (x == 'Y').mean() * 100).round(1)
        
        cols = st.columns(len(overall_rate))
        for col, src in zip(cols, overall_rate.index):
            col.metric(f"{src} Mentions Falcon", f"{overall_rate[src]}%")
        
        st.divider()
        
        # NEW ANALYSIS 1: Branded vs Non-Branded Breakdown
        st.subheader("🎯 Branded vs. Non-Branded Query Performance")
        st.caption("Comparison of Falcon mention rates between queries that include 'Falcon' (branded) and those that don't (non-branded)")
        
        branded_analysis = df_main.groupby(['Source', 'Branded_Query'])['Falcon_Mentioned'].apply(
            lambda x: (x == 'Y').mean() * 100
        ).unstack(fill_value=0).round(1)
        
        # Rename columns for clarity
        branded_analysis.columns = ['Non-Branded Queries', 'Branded Queries']
        
        fig = px.bar(branded_analysis.T, title="Falcon Mention Rate: Branded vs Non-Branded Queries",
                    labels={'value': 'Mention Rate (%)', 'index': 'Query Type'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.dataframe(branded_analysis.style.format("{:.1f}%"), use_container_width=True)
        
        st.divider()
        
        # NEW ANALYSIS 2: URL Citation Rate
        st.subheader("🔗 Falcon URL Citation Rate")
        st.caption("How often each LLM includes a link to falconstructures.com in their responses")
        
        url_citation_rate = df_main.groupby('Source')['Falcon_URL_Cited'].apply(
            lambda x: (x == True).mean() * 100
        ).round(1)
        
        fig = px.bar(x=url_citation_rate.index, y=url_citation_rate.values,
                    title="Falcon Website Citation Rate by Source",
                    labels={'x': 'Source', 'y': 'Citation Rate (%)'},
                    color=url_citation_rate.values,
                    color_continuous_scale='blues')
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # NEW ANALYSIS 3: Competitor Mentions Without Falcon
        st.subheader("⚠️ Competitor-Only Mentions")
        st.caption("Cases where competitors are mentioned but Falcon is not - these represent missed opportunities")
        
        # Find responses where competitors are mentioned but Falcon is not
        df_main['Has_Competitors'] = df_main['Competitors_Found'].apply(lambda x: len(x) > 0 if isinstance(x, list) else len(str(x)) > 0)
        competitor_only = df_main[(df_main['Has_Competitors']) & (df_main['Falcon_Mentioned'] == 'N')]
        
        if len(competitor_only) > 0:
            # Count by source
            competitor_only_by_source = competitor_only.groupby('Source').size()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(values=competitor_only_by_source.values, 
                           names=competitor_only_by_source.index,
                           title="Distribution of Competitor-Only Mentions by Source")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show which competitors appear most when Falcon doesn't
                all_competitors = []
                for comp_list in competitor_only['Competitors_Found']:
                    if isinstance(comp_list, list):
                        all_competitors.extend(comp_list)
                    elif isinstance(comp_list, str) and comp_list:
                        all_competitors.extend(comp_list.split(', '))
                
                if all_competitors:
                    comp_counts = pd.Series(all_competitors).value_counts().head(5)
                    fig = px.bar(x=comp_counts.values, y=comp_counts.index, orientation='h',
                               title="Top Competitors Mentioned When Falcon Isn't",
                               labels={'x': 'Number of Mentions', 'y': 'Competitor'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Show example queries
            st.markdown("**Example Queries Where Competitors Were Mentioned But Not Falcon:**")
            for i, query in enumerate(competitor_only['Query'].unique()[:5], 1):
                st.markdown(f"{i}. {query}")
        else:
            st.info("Good news! No cases found where competitors were mentioned without Falcon.")
        
        st.divider()
        
        # NEW ANALYSIS 4: Brand Share in Non-Branded Queries
        st.subheader("📈 Brand Share in Non-Branded Queries")
        st.caption("Among non-branded (generic) queries, what percentage of brand mentions go to each company")
        
        non_branded_df = df_main[df_main['Branded_Query'] == 'N'].copy()
        
        if len(non_branded_df) > 0:
            # Count mentions for each brand
            brand_mentions = {'Falcon': (non_branded_df['Falcon_Mentioned'] == 'Y').sum()}
            
            for competitor in COMPETITORS:
                brand_mentions[competitor] = non_branded_df['Response'].str.contains(
                    competitor, case=False, na=False
                ).sum()
            
            # Convert to percentages
            total_responses = len(non_branded_df)
            brand_share = {k: (v/total_responses * 100) for k, v in brand_mentions.items()}
            brand_share_df = pd.DataFrame(list(brand_share.items()), columns=['Brand', 'Share %'])
            brand_share_df = brand_share_df.sort_values('Share %', ascending=False)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(brand_share_df.head(10), x='Brand', y='Share %',
                           title="Brand Mention Share in Non-Branded Queries",
                           color='Share %', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show by source
                source_brand_share = []
                for source in non_branded_df['Source'].unique():
                    source_df = non_branded_df[non_branded_df['Source'] == source]
                    falcon_share = (source_df['Falcon_Mentioned'] == 'Y').mean() * 100
                    source_brand_share.append({
                        'Source': source,
                        'Falcon Share %': falcon_share
                    })
                
                source_share_df = pd.DataFrame(source_brand_share)
                fig = px.bar(source_share_df, x='Source', y='Falcon Share %',
                           title="Falcon's Share by LLM Source (Non-Branded Queries)",
                           color='Falcon Share %', color_continuous_scale='blues')
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.markdown("**Detailed Brand Mention Rates in Non-Branded Queries:**")
            st.dataframe(brand_share_df.style.format({'Share %': '{:.1f}%'}), use_container_width=True)
        else:
            st.info("No non-branded queries found in the dataset.")
        
        st.divider()
        
        # Position Analysis (Original)
        st.subheader("📍 Position Analysis")
        st.caption("Where in the response Falcon appears - earlier mentions indicate stronger brand association")
        
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
        
        # Context Analysis (Original)
        st.subheader("💭 Context Analysis")
        st.caption("Sentiment and context of Falcon mentions - positive context indicates favorable brand perception")
        
        col1, col2 = st.columns(2)
        
        with col1:
            context_counts = df_main['Context_Type'].value_counts()
            fig = px.bar(x=context_counts.index, y=context_counts.values, 
                        title="Context Type Distribution",
                        color=context_counts.index,
                        color_discrete_map={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red', 'Not Mentioned': 'gray'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Context sentiment by source
            context_by_source = df_main.groupby(['Source', 'Context_Type']).size().unstack(fill_value=0)
            fig = px.bar(context_by_source, title="Context Distribution by Source", 
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced cleaned dataset
        st.subheader("🧹 Enhanced Dataset")
        st.caption("Complete processed dataset with all analytical columns for further analysis")
        enhanced_columns = [
            "Date", "Query_Number", "Query", "Source", "Response", "Response_Word_Count",
            "Branded_Query", "Falcon_Mentioned", "Falcon_Position", "Context_Type", 
            "Context_Sentiment", "Competitors_Found", "Falcon_URL_Cited", "Sources_Cited"
        ]
        
        display_df = df_main[enhanced_columns]
        st.dataframe(display_df, use_container_width=True, height=400)
        
        st.download_button(
            "📥 Download Enhanced Analysis",
            display_df.to_csv(index=False),
            "enhanced_visibility_analysis.csv",
            "text/csv"
        )

# ─── TAB 3: COMPETITOR COMPARISON ─────────────────────────────────────────────
with tab3:
    st.markdown("### 🏆 Competitor Comparison Mode")
    
    # Option to use data from Tab 1 or upload new data
    data_source = st.radio(
        "Choose data source:",
        ["Run new queries", "Use results from Multi-LLM Generator", "Upload CSV file"]
    )
    
    if data_source == "Use results from Multi-LLM Generator":
        if 'latest_results' in st.session_state:
            st.success("Using results from Multi-LLM Response Generator")
            df_comp = st.session_state.latest_results.copy()
            
            # Run the competitor analysis
            selected_competitors = st.multiselect(
                "Select Competitors to Analyze:",
                ["Falcon"] + COMPETITORS,
                default=["Falcon", "ROXBOX", "Mobile Modular", "WillScot"]
            )
            
            if st.button("🔍 Analyze Competitors", key="analyze_existing"):
                # Analyze mentions for each selected competitor
                competitor_analysis = {}
                
                for competitor in selected_competitors:
                    mentions = df_comp['Response'].str.contains(competitor, case=False, na=False)
                    positions = df_comp['Response'].apply(lambda x: analyze_position(x, competitor))
                    contexts = df_comp['Response'].apply(lambda x: analyze_context(x, competitor))
                    
                    competitor_analysis[competitor] = {
                        'mention_rate': (mentions.sum() / len(df_comp) * 100),
                        'avg_position': sum([p[1] for p in positions if p[1] > 0]) / max(sum([1 for p in positions if p[1] > 0]), 1),
                        'positive_context': sum([1 for c in contexts if c[0] == 'Positive']) / len(df_comp) * 100,
                        'first_third_rate': sum([1 for p in positions if p[0] == 'First Third']) / len(df_comp) * 100
                    }
                
                # Display comparison matrix
                st.subheader("🏆 Competitor Performance Matrix")
                st.caption("Comprehensive comparison of brand performance metrics across all selected competitors")
                
                comparison_df = pd.DataFrame(competitor_analysis).T.round(1)
                comparison_df.columns = ['Mention Rate (%)', 'Avg Position', 'Positive Context (%)', 'First Third (%)']
                
                # Color-code the dataframe
                st.dataframe(
                    comparison_df.style.background_gradient(subset=['Mention Rate (%)'], cmap='RdYlGn')
                                      .background_gradient(subset=['Positive Context (%)'], cmap='RdYlGn')
                                      .background_gradient(subset=['First Third (%)'], cmap='RdYlGn')
                                      .background_gradient(subset=['Avg Position'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Mention rate comparison
                    mention_rates = [competitor_analysis[comp]['mention_rate'] for comp in selected_competitors]
                    fig = px.bar(x=selected_competitors, y=mention_rates, 
                                title="Mention Rate Comparison",
                                labels={'x': 'Competitor', 'y': 'Mention Rate (%)'},
                                color=mention_rates,
                                color_continuous_scale='RdYlGn')
                    fig.add_annotation(text="Higher is better - indicates stronger brand visibility",
                                     xref="paper", yref="paper",
                                     x=0.5, y=-0.15, showarrow=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Position comparison
                    first_third_rates = [competitor_analysis[comp]['first_third_rate'] for comp in selected_competitors]
                    fig = px.bar(x=selected_competitors, y=first_third_rates, 
                                title="First Third Position Rate",
                                labels={'x': 'Competitor', 'y': 'First Third Rate (%)'},
                                color=first_third_rates,
                                color_continuous_scale='RdYlGn')
                    fig.add_annotation(text="Higher is better - indicates prominent positioning",
                                     xref="paper", yref="paper",
                                     x=0.5, y=-0.15, showarrow=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No results available from Multi-LLM Generator. Please run some queries first.")
    
    elif data_source == "Upload CSV file":
        uploaded_comp = st.file_uploader("Upload your CSV file", type="csv", key="comp_upload")
        if uploaded_comp:
            df_comp = pd.read_csv(uploaded_comp)
            # Continue with same analysis logic as above
            st.info("File uploaded. Select competitors and click 'Analyze Competitors' to proceed.")
    
    else:  # Run new queries
        # Option to use predefined queries
        use_predefined = st.checkbox("Use predefined 20 queries", value=False)
        
        if use_predefined:
            comparison_queries = "\n".join(PREDEFINED_QUERIES)
            st.text_area(
                "Queries to run:",
                value=comparison_queries,
                height=200,
                disabled=True
            )
        else:
            # Competitor selection
            selected_competitors = st.multiselect(
                "Select Competitors to Compare:",
                ["Falcon"] + COMPETITORS,
                default=["Falcon", "ROXBOX", "Mobile Modular", "WillScot"]
            )
            
            # Side-by-side comparison queries
            comparison_queries = st.text_area(
                "Comparison Queries (one per line):",
                height=150,
                placeholder="Compare modular office companies\nBest portable building solutions\nModular office rental vs purchase"
            )
        
        if st.button("🔍 Run Competitor Analysis", key="competitor_analysis"):
            if use_predefined or comparison_queries.strip():
                queries = PREDEFINED_QUERIES if use_predefined else [q.strip() for q in comparison_queries.splitlines() if q.strip()]
                
                with st.spinner("Running competitor comparison analysis..."):
                    results = process_queries_parallel(queries)
                    df_comp = pd.DataFrame(results)
                    
                    if use_predefined:
                        selected_competitors = ["Falcon"] + COMPETITORS[:4]  # Default selection for predefined
                    
                    # Analyze mentions for each selected competitor
                    competitor_analysis = {}
                    
                    for competitor in selected_competitors:
                        mentions = df_comp['Response'].str.contains(competitor, case=False, na=False)
                        positions = df_comp['Response'].apply(lambda x: analyze_position(x, competitor))
                        contexts = df_comp['Response'].apply(lambda x: analyze_context(x, competitor))
                        
                        competitor_analysis[competitor] = {
                            'mention_rate': (mentions.sum() / len(df_comp) * 100),
                            'avg_position': sum([p[1] for p in positions if p[1] > 0]) / max(sum([1 for p in positions if p[1] > 0]), 1),
                            'positive_context': sum([1 for c in contexts if c[0] == 'Positive']) / len(df_comp) * 100,
                            'first_third_rate': sum([1 for p in positions if p[0] == 'First Third']) / len(df_comp) * 100
                        }
                    
                    # Display comparison matrix
                    st.subheader("🏆 Competitor Performance Matrix")
                    st.caption("Comprehensive comparison showing how each brand performs across key visibility metrics")
                    
                    comparison_df = pd.DataFrame(competitor_analysis).T.round(1)
                    comparison_df.columns = ['Mention Rate (%)', 'Avg Position', 'Positive Context (%)', 'First Third (%)']
                    
                    # Color-code the dataframe
                    st.dataframe(
                        comparison_df.style.background_gradient(subset=['Mention Rate (%)'], cmap='RdYlGn')
                                          .background_gradient(subset=['Positive Context (%)'], cmap='RdYlGn')
                                          .background_gradient(subset=['First Third (%)'], cmap='RdYlGn')
                                          .background_gradient(subset=['Avg Position'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Mention rate comparison
                        mention_rates = [competitor_analysis[comp]['mention_rate'] for comp in selected_competitors]
                        fig = px.bar(x=selected_competitors, y=mention_rates, 
                                    title="Mention Rate Comparison",
                                    labels={'x': 'Competitor', 'y': 'Mention Rate (%)}'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Position comparison
                        first_third_rates = [competitor_analysis[comp]['first_third_rate'] for comp in selected_competitors]
                        fig = px.bar(x=selected_competitors, y=first_third_rates, 
                                    title="First Third Position Rate",
                                    labels={'x': 'Competitor', 'y': 'First Third Rate (%)'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Side-by-side response comparison
                    st.subheader("📄 Side-by-Side Response Analysis")
                    st.caption("Detailed comparison of how each LLM responds to queries, with competitor mentions highlighted")
                    
                    for i, query in enumerate(queries[:5]):  # Show first 5 queries
                        with st.expander(f"Query {i+1}: {query[:100]}..." if len(query) > 100 else f"Query {i+1}: {query}", expanded=False):
                            query_responses = df_comp[df_comp['Query'] == query]
                            
                            cols = st.columns(min(len(query_responses), 3))
                            for col, (_, response_row) in zip(cols, query_responses.iterrows()):
                                with col:
                                    st.markdown(f"**{response_row['Source']}**")
                                    
                                    # Highlight competitor mentions
                                    response_text = response_row['Response']
                                    for competitor in selected_competitors:
                                        if competitor.lower() in response_text.lower():
                                            # Simple highlighting
                                            response_text = response_text.replace(
                                                competitor, f"**{competitor}**"
                                            )
                                    
                                    st.markdown(response_text[:500] + "..." if len(response_text) > 500 else response_text)
                                    
                                    # Show metrics for this response
                                    for competitor in selected_competitors:
                                        if competitor.lower() in response_row['Response'].lower():
                                            pos_info = analyze_position(response_row['Response'], competitor)
                                            context_info = analyze_context(response_row['Response'], competitor)
                                            
                                            color_class = "position-first" if pos_info[0] == "First Third" else \
                                                         "position-middle" if pos_info[0] == "Middle Third" else \
                                                         "position-last" if pos_info[0] == "Last Third" else "position-none"
                                            
                                            st.markdown(f"""
                                            <span class="position-indicator {color_class}">
                                                {competitor}: {pos_info[0]} | {context_info[0]}
                                            </span>
                                            """, unsafe_allow_html=True)

# ─── TAB 4: EXECUTIVE DASHBOARD WITH TIME SERIES ─────────────────────────────
with tab4:
    st.markdown("### 📈 Executive Dashboard & Time Series Analysis")
    st.markdown("*Comprehensive overview of search visibility performance with temporal trends*")
    
    # Create sub-tabs for dashboard and time series
    dashboard_tab, time_series_tab = st.tabs(["Executive Dashboard", "Time Series Analysis"])
    
    with dashboard_tab:
        # File upload for dashboard
        dashboard_file = st.file_uploader("Upload analysis results for dashboard", type="csv", key="dashboard_upload")
        
        # Check for latest results from Tab 1
        if 'latest_results' in st.session_state:
            use_latest_dash = st.checkbox("Use results from Multi-LLM Response Generator", value=True, key="dash_use_latest")
            if use_latest_dash:
                df_dashboard = st.session_state.latest_results.copy()
        elif dashboard_file:
            df_dashboard = pd.read_csv(dashboard_file)
        else:
            df_dashboard = None
        
        if df_dashboard is not None:
            # Ensure we have the necessary columns
            if 'Response' in df_dashboard.columns:
                # Process data for dashboard
                df_dashboard['Date'] = pd.to_datetime(df_dashboard.get('Date', datetime.today().date()))
                df_dashboard['Falcon_Mentioned'] = df_dashboard['Response'].str.contains('falcon', case=False, na=False)
                
                # Fix the .str.contains error by ensuring Query is string type
                df_dashboard['Query'] = df_dashboard['Query'].astype(str)
                df_dashboard['Branded_Query'] = df_dashboard['Query'].str.contains('falcon', case=False, na=False)
                
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
                st.caption("High-level metrics showing overall brand visibility and sentiment performance")
                
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
                st.caption("Detailed breakdown of how each LLM performs across key metrics")
                
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
                
                # Actionable Recommendations
                st.subheader("💡 Actionable Recommendations")
                st.caption("AI-generated insights based on current performance metrics")
                
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
    
    with time_series_tab:
        st.markdown("### 📈 Time Series Analysis")
        st.caption("Track performance trends over time to identify patterns and measure improvement")
        
        # Option to use accumulated data or upload
        ts_data_source = st.radio(
            "Choose data source for time series:",
            ["Upload CSV with historical data", "Use current session data"]
        )
        
        if ts_data_source == "Use current session data":
            if 'latest_results' in st.session_state:
                df_ts = st.session_state.latest_results.copy()
                
                # Ensure Date column exists
                if 'Date' not in df_ts.columns:
                    df_ts['Date'] = pd.to_datetime(datetime.today().date())
                else:
                    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
                
                # Process time series data
                # Fix the error by ensuring proper string type
                df_ts['Query'] = df_ts['Query'].astype(str)
                df_ts['Response'] = df_ts['Response'].astype(str)
                
                df_ts['Falcon_Mentioned'] = df_ts['Response'].str.contains('falcon', case=False, na=False)
                df_ts['Branded_Query'] = df_ts['Query'].str.contains('falcon', case=False, na=False)
                
                # Add enhanced analytics
                position_data = df_ts['Response'].apply(lambda x: analyze_position(x, "Falcon"))
                context_data = df_ts['Response'].apply(lambda x: analyze_context(x, "Falcon"))
                competitor_data = df_ts['Response'].apply(extract_competitors_detailed)
                
                df_ts['Position_Category'] = [p[0] for p in position_data]
                df_ts['Context_Type'] = [c[0] for c in context_data]
                df_ts['Context_Sentiment'] = [c[1] for c in context_data]
                df_ts['Competitors_Count'] = [len(c[0]) for c in competitor_data]
                
                # Check for Sources_Cited column
                if 'Sources_Cited' in df_ts.columns:
                    # Ensure Sources_Cited is string type before using .str accessor
                    df_ts['Sources_Cited'] = df_ts['Sources_Cited'].astype(str)
                    df_ts["Falcon_URL_Cited"] = df_ts['Sources_Cited'].str.contains("falconstructures.com", na=False, case=False)
                else:
                    df_ts["Falcon_URL_Cited"] = False
                
                # Create time series visualizations
                st.subheader("📊 Performance Metrics")
                st.caption("Key metrics aggregated by source showing current performance snapshot")
                
                # Aggregate by source (since we likely have single date)
                source_metrics = df_ts.groupby('Source').agg({
                    'Falcon_Mentioned': lambda x: (x.sum() / len(x) * 100),
                    'Position_Category': lambda x: sum([1 for p in x if p == 'First Third']) / len(x) * 100,
                    'Context_Type': lambda x: sum([1 for c in x if c == 'Positive']) / len(x) * 100,
                    'Falcon_URL_Cited': lambda x: (x.sum() / len(x) * 100),
                    'Competitors_Count': 'mean'
                }).round(1)
                
                source_metrics.columns = [
                    'Mention Rate (%)', 'First Position (%)', 'Positive Context (%)', 
                    'URL Citation (%)', 'Avg Competitors'
                ]
                
                # Display metrics table
                st.dataframe(source_metrics.style.format('{:.1f}'), use_container_width=True)
                
                # Visualization
                fig = px.bar(source_metrics.T, title="Performance Metrics by Source",
                           labels={'value': 'Percentage / Count', 'index': 'Metric'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("Note: Using current session data. For true time series analysis, upload a CSV with historical data spanning multiple dates.")
            else:
                st.warning("No data available. Please run queries in the Multi-LLM Response Generator first.")
        
        else:  # Upload CSV
            ts_file = st.file_uploader("Upload CSV with time series data", type="csv", key="ts_upload")
            
            if ts_file:
                df_ts = pd.read_csv(ts_file)
                
                # Process the uploaded file
                df_ts['Date'] = pd.to_datetime(df_ts['Date'])
                
                # Ensure string types to avoid errors
                for col in ['Query', 'Response']:
                    if col in df_ts.columns:
                        df_ts[col] = df_ts[col].astype(str)
                
                df_ts['Falcon_Mentioned'] = df_ts['Response'].str.contains('falcon', case=False, na=False)
                df_ts['Branded_Query'] = df_ts['Query'].str.contains('falcon', case=False, na=False)
                
                # Calculate daily metrics
                daily_metrics = df_ts.groupby(['Date', 'Source']).agg({
                    'Falcon_Mentioned': lambda x: (x.sum() / len(x) * 100)
                }).reset_index()
                
                daily_metrics.columns = ['Date', 'Source', 'Mention_Rate']
                
                # Create time series chart
                st.subheader("📈 Mention Rate Trends Over Time")
                st.caption("Track how mention rates change over time to identify trends and patterns")
                
                fig = px.line(daily_metrics, x='Date', y='Mention_Rate', color='Source',
                            title='Falcon Mention Rate Trends',
                            labels={'Mention_Rate': 'Mention Rate (%)'},
                            markers=True)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Period comparison
                if len(df_ts['Date'].unique()) > 1:
                    st.subheader("📊 Period Comparison")
                    st.caption("Compare performance between different time periods")
                    
                    # Split data into periods
                    mid_date = df_ts['Date'].median()
                    first_half = df_ts[df_ts['Date'] <= mid_date]
                    second_half = df_ts[df_ts['Date'] > mid_date]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**First Period**")
                        first_rate = (first_half['Falcon_Mentioned'].sum() / len(first_half) * 100)
                        st.metric("Mention Rate", f"{first_rate:.1f}%")
                    
                    with col2:
                        st.markdown("**Second Period**")
                        second_rate = (second_half['Falcon_Mentioned'].sum() / len(second_half) * 100)
                        change = second_rate - first_rate
                        st.metric("Mention Rate", f"{second_rate:.1f}%", f"{change:+.1f}%")

# ─── FOOTER ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.9rem; padding:2rem 0;'>
    <p><strong>Enhanced Falcon Structures LLM Search Visibility Tool</strong></p>
    <p>Advanced Analytics • Competitive Intelligence • Executive Insights</p>
    <p>Powered by OpenAI, Google Gemini, and Perplexity AI</p>
    <p style='font-size:0.8rem; margin-top:1rem;'>
        Created by <a href='https://www.weidert.com' target='_blank' style='color:#666;'>Weidert Group, Inc.</a>
        | Version 3.0 Enhanced
    </p>
</div>
""", unsafe_allow_html=True)

# ─── HELPFUL TIPS SIDEBAR ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("---")
    st.subheader("💡 Pro Tips")
    
    tips = [
        "**Predefined Queries**: Use the 20-query set for comprehensive analysis",
        "**Cross-Tab Data**: Results from Tab 1 can be used in other tabs",
        "**Brand vs Non-Brand**: Compare performance on branded vs generic queries",
        "**URL Citations**: Track how often your website is referenced",
        "**Competitor Gaps**: Find queries where competitors appear but you don't",
        "**Time Series**: Upload historical data to track trends over time",
        "**Executive View**: Dashboard provides high-level insights for stakeholders"
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

# ─── SESSION STATE MANAGEMENT ─────────────────────────────────────────────────

# Initialize session state variables
if 'template_query' not in st.session_state:
    st.session_state.template_query = ''

if 'run_triggered' not in st.session_state:
    st.session_state.run_triggered = False

if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True

if 'use_predefined' not in st.session_state:
    st.session_state.use_predefined = False

if 'latest_results' not in st.session_state:
    st.session_state.latest_results = None
