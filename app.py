import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading
import time
import random
import io
from PIL import Image

logo = Image.open("images/Housatonic_Partners_Logo.jpg")
st.image(logo, width=500)

# CONFIG
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective model, change to "gpt-4o" if needed
MAX_WORKERS = 5
PASSWORD = st.secrets["APP_PASSWORD"]

client = OpenAI(api_key=OPENAI_API_KEY)

# THREAD-SAFE GLOBALS
total_input_tokens = 0
total_output_tokens = 0
token_lock = threading.Lock()
df_lock = threading.Lock()
api_semaphore = threading.Semaphore(2)

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
    global total_input_tokens, total_output_tokens
    if hasattr(response, "usage"):
        with token_lock:
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

def call_openai(prompt):
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        add_token_usage(response)
        output = response.choices[0].message.content.strip()

        if len(output) > 50:
            time.sleep(1)
            return call_openai(prompt)

        return output
    except Exception as e:
        print(f"[API ERROR] {e}")
        return "GENERATION ERROR"

def is_visible(tag):
    return tag.name == 'p' and tag.get_text(strip=True) and len(tag.get_text(strip=True)) > 40

def clean_text(text_list):
    return '\n\n'.join([t.strip() for t in text_list if len(t.strip()) > 40])

def extract_main_text(base_url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        fallback_paths = ['/about', '/services', '/what-we-do', '/who-we-are', '']
        for path in fallback_paths:
            try:
                target_url = urljoin(base_url, path)
                response = requests.get(target_url, timeout=5, headers=headers)
                if response.status_code != 200:
                    continue
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.find('main') or max(soup.find_all('div'), key=lambda d: len(d.get_text(strip=True)), default=soup)
                paragraphs = content.find_all(is_visible)
                text = clean_text([p.get_text() for p in paragraphs])
                if len(text) > 300:
                    return text
            except:
                continue
        return None
    except:
        return None

def classify_website(i, url, df):
    if not url.lower().startswith(('http://', 'https://')):
        url = 'https://' + url

    with api_semaphore:
        try:
            content = extract_main_text(url)
            if content and len(content.split()) >= 30:
                trimmed = ' '.join(content.split()[:800])
                prompt = classification_prompt.format(content=trimmed)
                vertical = call_openai(prompt)
                if vertical == "ERROR":
                    vertical = call_openai(url_fallback_prompt.format(url=url)) + " *"
            else:
                vertical = call_openai(url_fallback_prompt.format(url=url)) + " *"

            result = vertical if vertical and vertical != "ERROR" else "GENERATION ERROR"
        except Exception as e:
            result = f"[ERROR] {url}: {e}"
            content = ''

        with df_lock:
            df.at[i, 'Vertical Focus OpenAI'] = result
            df.at[i, 'Website Content'] = content if content else ''

        time.sleep(random.uniform(0.5, 1.5))

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
        if 'Vertical Focus OpenAI' not in df.columns:
            df['Vertical Focus OpenAI'] = ''
        if 'Website Content' not in df.columns:
            df['Website Content'] = ''

        df['Vertical Focus OpenAI'] = df['Vertical Focus OpenAI'].astype(str)
        df['Website Content'] = df['Website Content'].astype(str)

        total_urls = len(df)
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        with st.spinner("Processing websites..."):
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                # Detect which column to use for URLs
                url_column = None
                for candidate in ["Account: Website", "URL"]:
                    if candidate in df.columns:
                        url_column = candidate
                        break
                
                if not url_column:
                    st.error("CSV must contain a column named either 'Account: Website' or 'URL'.")
                    st.stop()
                
                for i, row in df.iterrows():
                    url = str(row[url_column]).strip()
                    if url and url != 'nan':
                        futures.append(executor.submit(classify_website, i, url, df))

                for count, future in enumerate(as_completed(futures)):
                    future.result()
                    elapsed = int(time.time() - start_time)
                    remaining = int((elapsed / (count + 1)) * (total_urls - count - 1)) if count > 0 else 0
                    status_text.text(f"Processed {count + 1}/{total_urls} | ~{remaining}s remaining")
                    progress_bar.progress((count + 1) / total_urls)

        st.session_state.processed_df = df
        st.session_state.token_stats = (total_input_tokens, total_output_tokens)

    else:
        df = st.session_state.processed_df
        total_input_tokens, total_output_tokens = st.session_state.token_stats

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

    # Updated cost calculation for OpenAI pricing
    # GPT-4o-mini pricing: $0.15 per 1M input tokens, $0.60 per 1M output tokens
    input_cost = (total_input_tokens / 1_000_000) * 0.15
    output_cost = (total_output_tokens / 1_000_000) * 0.60
    total_cost = input_cost + output_cost

    st.markdown(f"**Input Tokens:** {total_input_tokens:,}")
    st.markdown(f"**Output Tokens:** {total_output_tokens:,}")
    st.markdown(f"**Estimated OpenAI API Cost:** `${total_cost:.4f}`")
    st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
