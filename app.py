import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import re
import pandas as pd
import time
import os

# ========== CONFIGURATION ==========
# API keys: store in Streamlit secrets or environment variables
openai_api_key = st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
gemini_api_key = st.secrets.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
perplexity_api_key = st.secrets.get("perplexity_api_key") or os.getenv("PERPLEXITY_API_KEY")

# Model names
openai_model = st.secrets.get("openai_model", "gpt-4")
gemini_model_name = st.secrets.get("gemini_model_name", "gemini-2.5-flash")
perplexity_model_name = st.secrets.get("perplexity_model_name", "sonar-pro")

# Competitor list
competitors = ["ROXBOX Containers", "Wilmot Modular", "Pac-Van", "BMarko Structures", "Giant Containers", "XCaliber Container", "Conexwest", "Mobile Modular Portable Storage", "WillScot"]

# ========== CLIENT INITIALIZATION ==========
openai_client = OpenAI(api_key=openai_api_key)

genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel(gemini_model_name)

perplexity_client = OpenAI(
    api_key=perplexity_api_key,
    base_url="https://api.perplexity.ai"
)

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

# ========== HELPER ==========
def extract_links(text: str) -> list:
    return re.findall(r'https?://\S+', text)

# ========== STREAMLIT UI ==========
st.title("🔍 Falcon Structures AI Powered LLM Search Visibility Tool")
st.markdown(
    "Paste multiple search queries (one per line) and compare answers from ChatGPT, Gemini, and Perplexity.  Add -- "Provide sources where you are extracting information from in this format - 'https?://\\S+'" -- to the end of each querry."
)

queries_input = st.text_area(
    "Enter your queries here:",
    height=150,
    placeholder="What companies provide modular container offices in the US? Provide sources where you are extracting information from in this format - 'https?://\\S+'""
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
                    # metrics
                    wc = len(text.split())
                    falcon_flag = "Y" if re.search(r'\bfalcon\b|\bfalconstructures\b', text, re.IGNORECASE) else "N"
                    cite_flag = "Y" if re.search(r'https?://|\[\d+\]', text) else "N"
                    found = [c for c in competitors if re.search(re.escape(c), text, re.IGNORECASE)]
                    links = extract_links(text)
                    # position type logic
                    p20 = text[:max(1, int(0.2*len(text)))].lower()
                    if any(k.lower() in p20 for k in ["falcon"]+competitors):
                        pos = "lead answer"
                    elif cite_flag == "Y" and falcon_flag == "N":
                        pos = "citation"
                    elif falcon_flag == "Y" or found:
                        pos = "embedded mention"
                    else:
                        pos = "absent"
                    # collect row
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
            "Download Results as CSV", df.to_csv(index=False), "results.csv"
        )
