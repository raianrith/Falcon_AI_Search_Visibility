import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import re
import pandas as pd
import time
import os

# ─── PAGE CONFIG & GLOBAL CSS ─────────────────────────────────────────────────
st.set_page_config(page_title="Falcon Structures LLM Tool", layout="wide")

st.markdown(
    """
    <div style='text-align: center; padding-top: 10px; padding-bottom: 10px;'>
        <img src='https://github.com/raianrith/AI-Client-Research-Tool/blob/main/Weidert_Logo_primary-logomark-antique.png?raw=true' 
             width='100' style='margin-bottom: 10px;' />
        <h1 style='font-size: 2.4em; margin-bottom: 0;'>Falcon AI‑Powered LLM Search Visibility Tool</h1>
        <!-- subheader -->
        <h4 style='margin-top: 4px; font-size: 1em; color: #ccc;'>
            Created by Weidert Group, Inc.
        </h4>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* ── Main background & text ───────────────────────────────────── */
    [data-testid="stAppViewContainer"] {
        background-color: #000 !important;
    }
    [data-testid="stAppViewContainer"] *,
    [data-testid="stAppViewContainer"] {
        color: #fff !important;
        background-color: transparent !important;
    }

    /* ── Sidebar styling ─────────────────────────────────────────── */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #011a01 !important;  /* tech green */
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox__label {
        color: #fff !important;
    }
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div,
    .stNumberInput>div>div>input {
        background-color: #001f00 !important;  /* darker green */
        color: #fff !important;
        border: 1px solid #024504 !important;
        border-radius: 6px !important;
    }
    .stSelectbox>div>div>div>div {
        background-color: #001f00 !important;
        color: #fff !important;
    }

    /* ── Header & instructions ───────────────────────────────────── */
    .title-container {
        text-align: center;
        margin-bottom: 1rem;
    }
    .title-container h1 {
        margin: 0;
        font-size: 2.2rem;
    }
    .title-container img {
        vertical-align: middle;
        margin-right: 0.5rem;
        width: 48px;
    }
    .instructions {
        text-align: center;
        margin-bottom: 1.5rem;
        line-height: 1.4;
        font-size: 1rem;
    }
    .instructions code {
        background-color: #024504;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        color: #fff;
    }

    /* ── Textarea ─────────────────────────────────────────────────── */
    .stTextArea>div>div>textarea {
        background-color: #001f00 !important;
        color: #fff !important;
        border: 1px solid #024504 !important;
        border-radius: 6px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #038c3d !important;
        outline: none !important;
    }

    /* ── Button ───────────────────────────────────────────────────── */
    .stButton>button {
        background-color: #024504 !important;
        color: #fff !important;
        border: 1px solid #fff !important;
        border-radius: 6px !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1rem !important;
        display: block !important;
        margin: 1.5rem auto !important;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #038c3d !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── SIDEBAR: MODEL CONFIGURATION ─────────────────────────────────────────────
st.sidebar.title("🛠️ Model Configuration")
openai_model = st.sidebar.selectbox(
    "OpenAI model",
    ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
    index=0
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

# ─── API KEYS & CLIENT INITIALIZATION ─────────────────────────────────────────
openai_api_key     = st.secrets.get("openai_api_key")    or os.getenv("OPENAI_API_KEY")
gemini_api_key     = st.secrets.get("gemini_api_key")    or os.getenv("GEMINI_API_KEY")
perplexity_api_key = st.secrets.get("perplexity_api_key") or os.getenv("PERPLEXITY_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel(gemini_model_name)
perplexity_client = OpenAI(
    api_key=perplexity_api_key,
    base_url="https://api.perplexity.ai"
)

# ─── SYSTEM PROMPT & HELPERS ─────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "I am passing a few queries. You need to give me a response that you would "
    "provide to anyone else querying the same thing."
)

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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Perplexity error for query '{query}': {e}")
        return "ERROR"

def extract_links(text: str) -> list:
    return re.findall(r'https?://\S+', text)

competitors = [
    "ROXBOX Containers", "Wilmot Modular", "Pac-Van", "BMarko Structures",
    "Giant Containers", "XCaliber Container", "Conexwest",
    "Mobile Modular Portable Storage", "WillScot"
]

# ─── HEADER & INSTRUCTIONS ───────────────────────────────────────────────────
st.markdown(
    """
    <div class="instructions">
      Paste multiple search queries (one per line) and compare answers from OpenAI, Gemini, and Perplexity.<br>
    </div>
    """,
    unsafe_allow_html=True
)

# ─── QUERY BOX & RUN LOGIC ────────────────────────────────────────────────────
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
                    time.sleep(1)

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download Results as CSV",
            df.to_csv(index=False),
            file_name="results.csv",
            mime="text/csv"
        )
