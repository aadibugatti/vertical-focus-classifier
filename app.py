import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import anthropic
from datetime import datetime
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import io



# CONFIG
CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_WORKERS = 5
PASSWORD = st.secrets["APP_PASSWORD"]

client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# THREAD-SAFE GLOBALS
if "input_tokens" not in st.session_state:
    st.session_state.input_tokens = 0
if "output_tokens" not in st.session_state:
    st.session_state.output_tokens = 0

token_lock = threading.Lock()

# PROMPTS
classification_prompt = """Given the following company website content, identify the company's vertical focus in 1–5 words using common industry terms or acronyms (e.g., "HOA", "VAR").

Avoid vague or overly broad categories. Be specific about the industry the company serves.
Examples:
- "IT Consulting" — too generic
- "Real Estate IT Consulting" — specific and useful

Return ONLY the vertical focus term (e.g., "Self Storage", "Outdoor Hospitality", "HOA", etc). Ensure the output is less than 50 characters. If the industry is unclear, return exactly: ERROR

Website content:
{content}
"""

url_fallback_prompt = """The content of the following company website could not be accessed. Based only on the company URL and your best inference, guess the likely vertical focus in 1–5 words.

Return ONLY the vertical focus term (e.g., "Self Storage", "Outdoor Hospitality", "HOA", etc). This is a low-confidence guess.

URL: {url}
"""

def estimate_tokens(text):
    return len(text) // 4

def add_token_usage(response):
    if hasattr(response, "usage"):
        with token_lock:
            st.session_state.input_tokens += response.usage.input_tokens
            st.session_state.output_tokens += response.usage.output_tokens


def call_claude(prompt):
    global total_input_tokens, total_output_tokens

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    add_token_usage(response)
    output = response.content[0].text.strip()

    if len(output) > 50:
        time.sleep(1)
        return call_claude(prompt)

    return output

def get_visible_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=1)
        soup = BeautifulSoup(res.text, 'html.parser')
        for s in soup(['script', 'style', 'noscript']):
            s.decompose()
        return ' '.join(soup.stripped_strings)
    except:
        return None

def get_full_site_content(base_url):
    try:
        combined_text = get_visible_text(base_url)
        if not combined_text:
            return None
        res = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=1)
        soup = BeautifulSoup(res.text, 'html.parser')
        found_links = set()
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            if any(kw in href for kw in ['about', 'services', 'what-we-do', 'who-we-are']):
                full_url = urljoin(base_url, href)
                domain = urlparse(base_url).netloc
                if domain in urlparse(full_url).netloc:
                    found_links.add(full_url)
        for link in found_links:
            page_text = get_visible_text(link)
            if page_text:
                combined_text += '\n' + page_text
        return combined_text
    except:
        return None

def classify_website(i, url, df):
    if not url.lower().startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        content = get_full_site_content(url)

        if content and len(content.split()) >= 30:
            trimmed = ' '.join(content.split()[:800])
            prompt = classification_prompt.format(content=trimmed)
            vertical = call_claude(prompt)
            if vertical == "ERROR":
                vertical = call_claude(url_fallback_prompt.format(url=url)) + " *"
        else:
            vertical = call_claude(url_fallback_prompt.format(url=url)) + " *"

        result = vertical if vertical and vertical != "ERROR" else "GENERATION ERROR"
    except Exception as e:
        result = f"[ERROR]: {e}"

    df.at[i, "Vertical Focus Claude"] = result

# STREAMLIT UI
st.title("Vertical Focus Classifier")
st.markdown("Upload a CSV with a column called `Account: Website` to classify each website by industry vertical.")

# Password protection
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Enter password", type="password")
    if password == PASSWORD:
        st.session_state.authenticated = True
    else:
        st.stop()

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    if "processed_df" not in st.session_state:
        df = pd.read_csv(uploaded_file)
        if "Vertical Focus Claude" not in df.columns:
            df["Vertical Focus Claude"] = ''

        with st.spinner("Processing websites..."):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, row in df.iterrows():
            url = str(row["Account: Website"]).strip()
            if url and url != 'nan':
                futures.append(executor.submit(classify_website, i, url, df))

        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        total = len(futures)
        completed = 0
        start_time = time.time()
        
        for future in as_completed(futures):
            future.result()
            completed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            est_remaining = int(avg_time * (total - completed))
        
            progress_bar.progress(completed / total)
            progress_text.markdown(
                f"**{completed}/{total} completed** ⏳ _Estimated time left: {est_remaining} sec_"
            )
        
        progress_text.markdown("✅ **All tasks completed!**")

        
        st.session_state["processed_df"] = df
    else:
        df = st.session_state["processed_df"]

    st.success("Processing complete.")
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        label="📥 Download Results CSV",
        data=buffer,
        file_name="output.csv",
        mime="text/csv"
    )


    input_cost = (st.session_state.input_tokens / 1_000_000) * 3
    output_cost = (st.session_state.output_tokens / 1_000_000) * 15
    total_cost = input_cost + output_cost
    
    st.markdown(f"**Input Tokens:** {st.session_state.input_tokens:,}")
    st.markdown(f"**Output Tokens:** {st.session_state.output_tokens:,}")
    st.markdown(f"**Estimated Claude API Cost:** `${total_cost:.4f}`")
    st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
