import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import re
import pandas as pd
import time
import os

# ========== SIDEBAR: MODEL SELECTION ==========
st.sidebar.title("🛠️ Model Configuration")

# OpenAI model options
openai_choices = [
    "gpt-4", "gpt-4o", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
]
openai_model = st.sidebar.selectbox(
    "OpenAI model",
    openai_choices,
    index=openai_choices.index("gpt-4")
)

# Gemini model options
gemini_choices = [
    "gemini-2.5-flash", "gemini-2.5-pro"
]
gemini_model_name = st.sidebar.selectbox(
    "Gemini model",
    gemini_choices,
    index=gemini_choices.index("gemini-2.5-flash")
)

# Perplexity model options
perplexity_choices = [
    "sonar", "sonar-pro"
]
perplexity_model_name = st.sidebar.selectbox(
    "Perplexity model",
    perplexity_choices,
    index=perplexity_choices.index("sonar")
)

# ========== CONFIGURATION: API KEYS ==========
openai_api_key = st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
gemini_api_key = st.secrets.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
perplexity_api_key = st.secrets.get("perplexity_api_key") or os.getenv("PERPLEXITY_API_KEY")

# ========== CLIENT INITIALIZATION ==========
openai_client = OpenAI(api_key=openai_api_key)

genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel(gemini_model_name)

perplexity_client = OpenAI(
    api_key=perplexity_api_key,
    base_url="https://api.perplexity.ai"
)

# ========== SYSTEM PROMPT ==========
SYSTEM_PROMPT = (
    "You are a marketing agent trying to analyze search visibility. "
    "I am passing a few queries. You need to give me a response that you would "
    "provide to anyone else querying the same thing."
)

# ========== RESPONSE FUNCTIONS ==========
def get_openai_response(query: str) -> str:
    try:
        resp = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI error for query '{query}': {e}")
        return "ERROR"

def get_gemini_response(query: str) -> str:
    try:
        response = gemini_model.generate_content(query)
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        st.error(f"Gemini error for query '{query}': {e}")
        return "ERROR"

def get_perplexity_response(query: str) -> str:
    try:
        resp = perplexity_client.chat.completions.create(
            model=perplexity_model_name,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": query}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Perplexity error for query '{query}': {e}")
        return "ERROR"

# ========== HELPER FUNCTION ==========
def extract_links(text: str) -> list:
    return re.findall(r'https?://\S+', text)

# ========== COMPETITOR LIST ==========
competitors = [
    "ROXBOX Containers", "Wilmot Modular", "Pac-Van", "BMarko Structures",
    "Giant Containers", "XCaliber Container", "Conexwest",
    "Mobile Modular Portable Storage", "WillScot"
]

# ========== STREAMLIT UI ==========
import streamlit as st

# ─── INJECT CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── App background & text ───────────────────────────────────────── */
    /* Target the main view container */
    [data-testid="stAppViewContainer"] {
        background-color: #000 !important;
    }
    /* Force white text everywhere in the main view */
    [data-testid="stAppViewContainer"], 
    [data-testid="stAppViewContainer"] * {
        color: #fff !important;
    }

    /* ── Centered headings & paragraphs ─────────────────────────────── */
    h1, p.centered {
        text-align: center;
    }

    /* ── Styled code snippets in your centered paragraph ────────────── */
    p.centered code {
        background-color: #222;
        color: #0f0;
        padding: 2px 4px;
        border-radius: 4px;
    }

    /* ── Textarea styling ───────────────────────────────────────────── */
    .stTextArea>div>div>textarea {
        background-color: #121212 !important;
        color: #fff !important;
        border: 2px solid #4CAF50 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── CENTERED TITLE & TEXT ─────────────────────────────────────────────────
st.markdown("<h1>🔍 Falcon Structures AI Powered LLM Search Visibility Tool</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='centered'>Paste multiple search queries (one per line) and compare answers from OpenAI, Gemini, and Perplexity.</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p class='centered'><code>-- Provide sources where you are extracting information from in this format - 'https?://\\S+' --</code></p>",
    unsafe_allow_html=True
)
# ─── INJECT CSS Center button ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* center & restyle the “Run Analysis” button */
    .stButton>button {
        margin: 24px auto !important;
        display: block !important;
        background-color: #4CAF50 !important;
        color: #fff !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ─── NICED‑UP TEXTAREA ────────────────────────────────────────────────────────
queries_input = st.text_area(
    "Enter your queries here:",  
    height=150,
    placeholder=(
        "e.g. What companies provide modular container offices in the US? "
        "-- Provide sources where you are extracting information from in this format - 'https?://\\S+'"
    )
)


if st.button("Run Analysis"):
    queries = [q.strip() for q in queries_input.splitlines() if q.strip()]
    if not queries:
        st.warning("Please enter at least one query.")
    else:
        results = []
        with st.spinner("Gathering responses..."):
            for query in queries:
                for source, func in [
                    ("OpenAI", get_openai_response),
                    ("Gemini", get_gemini_response),
                    ("Perplexity", get_perplexity_response)
                ]:
                    text = func(query)
                    wc = len(text.split())
                    falcon_flag = "Y" if re.search(r'\bfalcon\b|\bfalconstructures\b', text, re.IGNORECASE) else "N"
                    cite_flag = "Y" if re.search(r'https?://|\[\d+\]', text) else "N"
                    found = [c for c in competitors if re.search(re.escape(c), text, re.IGNORECASE)]
                    links = extract_links(text)
                    # Determine position type
                    snippet = text[:max(1, int(0.2 * len(text)))].lower()
                    if any(k.lower() in snippet for k in ["falcon"] + competitors):
                        pos = "lead answer"
                    elif cite_flag == "Y" and falcon_flag == "N":
                        pos = "citation"
                    elif falcon_flag == "Y" or found:
                        pos = "embedded mention"
                    else:
                        pos = "absent"
                    results.append({
                        "Query": query,
                        "Source": source,
                        "Response": text,
                        "Word Count": wc,
                        "Falcon Mentioned": falcon_flag,
                        "Citation Present": cite_flag,
                        "Competitors Mentioned": ", ".join(found),
                        "Position Type": pos,
                        "Links": ", ".join(links)
                    })
                    time.sleep(1)  # rate-limit safety

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download Results as CSV",
            df.to_csv(index=False),
            file_name="results.csv",
            mime="text/csv"
        )
