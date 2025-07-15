# app.py
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import re
import pandas as pd
import time
import os

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Falcon Structures LLM Search",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Settings")
    sleep_sec = st.slider("⏱️ Delay between calls (s)", 0.5, 3.0, 1.0, step=0.5)
    show_links = st.checkbox("🔗 Show extracted URLs", value=True)
    st.markdown("---")
    st.write("API Models")
    openai_model = st.selectbox("OpenAI model", ["gpt-4", "gpt-3.5-turbo"], index=0)
    gemini_model_name = st.text_input("Gemini model", "gemini-2.5-flash")
    perplexity_model_name = st.text_input("Perplexity model", "sonar-pro")

# ========== CONFIGURATION ==========
openai_api_key     = st.secrets.get("openai_api_key")     or os.getenv("OPENAI_API_KEY")
gemini_api_key     = st.secrets.get("gemini_api_key")     or os.getenv("GEMINI_API_KEY")
perplexity_api_key = st.secrets.get("perplexity_api_key") or os.getenv("PERPLEXITY_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel(gemini_model_name)
perplexity_client = OpenAI(
    api_key=openai_api_key,
    base_url="https://api.perplexity.ai"
)

SYSTEM_PROMPT = (
    "You are a marketing agent analyzing search visibility for Falcon Structures. "
    "Answer succinctly and include requested sources."
)

competitors = [
    "ROXBOX Containers", "Wilmot Modular", "Pac-Van", "BMarko Structures",
    "Giant Containers", "XCaliber Container", "Conexwest",
    "Mobile Modular Portable Storage", "WillScot"
]

# ========== UTILS ==========
def extract_links(text: str) -> list[str]:
    return re.findall(r'https?://\S+', text)

def get_openai_response(q): 
    resp = openai_client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": q}
        ]
    )
    return resp.choices[0].message.content.strip()

def get_gemini_response(q):
    r = gemini_model.generate_content(q)
    return r.candidates[0].content.parts[0].text.strip()

def get_perplexity_response(q):
    resp = perplexity_client.chat.completions.create(
        model=perplexity_model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": q}
        ]
    )
    return resp.choices[0].message.content.strip()

# ========== MAIN UI ==========
st.title("🔍 Falcon Structures AI‑Powered LLM Search Visibility")
st.markdown(
    "Paste **one query per line** below, then hit **Run Analysis**.  "
    "Make sure to append your source‑format hint (`https?://\\S+`) if you need URLs extracted."
)

with st.form("query_form", clear_on_submit=False):
    col1, col2 = st.columns([4, 1])
    with col1:
        queries_input = st.text_area(
            "Enter your queries here:",
            height=180,
            placeholder="What companies provide modular container offices in the US? Provide sources where you are extracting information from in this format - 'https?://\\S+'"
        )
    with col2:
        run = st.form_submit_button("🔍 Run Analysis")

if run:
    queries = [q.strip() for q in queries_input.splitlines() if q.strip()]
    if not queries:
        st.warning("Please enter at least one query.")
    else:
        results = []
        with st.spinner("Contacting LLMs…"):
            for q in queries:
                for source, fn in [
                    ("ChatGPT",    get_openai_response),
                    ("Gemini",     get_gemini_response),
                    ("Perplexity", get_perplexity_response)
                ]:
                    text = fn(q)
                    wc = len(text.split())
                    falcon_flag = "Y" if re.search(r'\bfalcon\b', text, re.IGNORECASE) else "N"
                    cite_flag   = "Y" if bool(extract_links(text)) else "N"
                    found = [c for c in competitors if re.search(re.escape(c), text, re.IGNORECASE)]
                    pos = (
                        "lead answer" if any(k.lower() in text[:int(0.2*len(text))].lower() 
                                            for k in ["falcon"]+competitors)
                        else "citation" if cite_flag=="Y" and falcon_flag=="N"
                        else "embedded mention" if falcon_flag=="Y" or found
                        else "absent"
                    )
                    links = extract_links(text) if show_links else []
                    results.append({
                        "Query": q,
                        "Source": source,
                        "Response": text,
                        "Word Count": wc,
                        "Falcon Mentioned": falcon_flag,
                        "Citation Present": cite_flag,
                        "Competitors Mentioned": ", ".join(found),
                        "Position Type": pos,
                        "Links": ", ".join(links),
                    })
                    time.sleep(sleep_sec)

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "💾 Download CSV", df.to_csv(index=False), "falcon_search_results.csv"
        )
