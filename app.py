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
import json

# Load logo
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

# CLASSIFICATION PROMPTS
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

# FILTERING PROMPTS
def create_batch_filter_prompt(entries, target_vertical):
    entries_text = "\n".join([f"{i+1}. {entry}" for i, entry in enumerate(entries)])
    
    return f"""
You are a business vertical classification expert. I need you to determine which of the following business verticals/industries match or are closely related to "{target_vertical}".

Consider these as matches:
- Direct matches (exact same industry)
- Closely related sub-industries or specializations
- Services that primarily serve the target vertical
- Different naming conventions for the same industry

Do NOT consider these as matches:
- Completely unrelated industries
- Generic business services unless specifically focused on the target vertical

Here are the entries to classify:
{entries_text}

Please respond with ONLY a JSON array of true/false values, where true means the entry matches "{target_vertical}" and false means it doesn't. The array should have exactly {len(entries)} elements corresponding to the numbered entries above.

Example response format: [true, false, true, false, ...]
"""

def estimate_tokens(text):
    return len(text) // 4

def add_token_usage(response):
    global total_input_tokens, total_output_tokens
    if hasattr(response, "usage"):
        with token_lock:
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

def call_openai(prompt, max_tokens=100):
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens
        )
        add_token_usage(response)
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return "ERROR"

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

def classify_batch_for_filtering(entries, target_vertical):
    """Classify a batch of vertical entries for filtering"""
    prompt = create_batch_filter_prompt(entries, target_vertical)
    
    try:
        response = call_openai(prompt, max_tokens=500)
        results = json.loads(response)
        
        if not isinstance(results, list) or len(results) != len(entries):
            return [False] * len(entries)
        
        return [bool(r) for r in results]
    except json.JSONDecodeError:
        return [False] * len(entries)

def clean_vertical_entry(entry):
    """Clean vertical entry by removing asterisks and extra whitespace"""
    if pd.isna(entry):
        return ""
    return str(entry).replace('*', '').strip()

# STREAMLIT UI
st.title("Housatonic Partners Tools")

# Password protection
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Enter password", type="password")
    if password == PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    else:
        st.stop()

# Tool selection
st.markdown("---")
st.subheader("Select a Tool")

tool_option = st.selectbox(
    "Choose which tool you'd like to use:",
    ["Vertical Focus Classifier", "Vertical Focus Filter"]
)

st.markdown("---")

# TOOL 1: VERTICAL FOCUS CLASSIFIER
if tool_option == "Vertical Focus Classifier":
    st.header("🔍 Vertical Focus Classifier")
    st.markdown("Upload a CSV with a column called `Account: Website` or `URL` to classify each website by industry vertical.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="classifier_upload")
    
    if uploaded_file:
        if "processed_df" not in st.session_state:
            df = pd.read_csv(uploaded_file)
            
            # Detect which column to use for URLs
            url_column = None
            for candidate in ["Account: Website", "URL"]:
                if candidate in df.columns:
                    url_column = candidate
                    break
            
            if not url_column:
                st.error("CSV must contain a column named either 'Account: Website' or 'URL'.")
                st.stop()
            
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

        st.success("✅ Processing complete!")
        
        # Display results preview
        st.subheader("Results Preview")
        st.dataframe(df[['Vertical Focus OpenAI']].head(10))
        
        # Download button
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Results CSV",
            data=buffer,
            file_name=f"classified_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # Cost estimation
        input_cost = (total_input_tokens / 1_000_000) * 0.15
        output_cost = (total_output_tokens / 1_000_000) * 0.60
        total_cost = input_cost + output_cost

        st.markdown("### 📊 Usage Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input Tokens", f"{total_input_tokens:,}")
        with col2:
            st.metric("Output Tokens", f"{total_output_tokens:,}")
        with col3:
            st.metric("Estimated Cost", f"${total_cost:.4f}")

# TOOL 2: VERTICAL FOCUS FILTER
elif tool_option == "Vertical Focus Filter":
    st.header("🔧 Vertical Focus Filter")
    st.markdown("Upload a CSV with vertical focus data and filter rows that match a specific vertical using ChatGPT classification.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="filter_upload")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Column selection
        column_options = list(df.columns)
        default_col = "Vertical Focus" if "Vertical Focus" in column_options else column_options[0]
        
        selected_column = st.selectbox(
            "Select the column containing vertical focus data:",
            column_options,
            index=column_options.index(default_col)
        )
        
        # Vertical input
        search_vertical = st.text_input(
            "Enter the vertical to search for:",
            placeholder="e.g., Legal Services, Education, Property Management"
        )
        
        if search_vertical and st.button("🔍 Filter Data", type="primary"):
            # Clean the data
            df[selected_column] = df[selected_column].apply(clean_vertical_entry)
            non_empty_df = df[df[selected_column] != ''].copy()
            
            if len(non_empty_df) == 0:
                st.error("No valid entries found in the selected column.")
                st.stop()
            
            # Reset token counters for this operation
            total_input_tokens = 0
            total_output_tokens = 0
            
            # Process in batches
            batch_size = 20
            entries_list = non_empty_df[selected_column].tolist()
            total_batches = (len(entries_list) + batch_size - 1) // batch_size
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_matches = []
            
            with st.spinner("Filtering data using ChatGPT..."):
                for i in range(0, len(entries_list), batch_size):
                    batch = entries_list[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    
                    status_text.text(f"Processing batch {batch_num}/{total_batches}...")
                    
                    batch_results = classify_batch_for_filtering(batch, search_vertical)
                    all_matches.extend(batch_results)
                    
                    progress_bar.progress(batch_num / total_batches)
                    time.sleep(0.5)  # Rate limiting
            
            # Filter results
            non_empty_df['chatgpt_match'] = all_matches
            matching_rows = non_empty_df[non_empty_df['chatgpt_match'] == True].copy()
            matching_rows = matching_rows.drop('chatgpt_match', axis=1)
            
            st.success(f"✅ Found {len(matching_rows)} matching rows for '{search_vertical}'")
            
            if len(matching_rows) > 0:
                # Display results preview
                st.subheader("Matching Entries Preview")
                st.dataframe(matching_rows[[selected_column]].head(10))
                
                # Download button
                buffer = io.BytesIO()
                matching_rows.to_csv(buffer, index=False)
                buffer.seek(0)
                
                safe_vertical = search_vertical.replace(' ', '_').replace('/', '_')
                filename = f"filtered_{safe_vertical}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                st.download_button(
                    label="📥 Download Filtered Results",
                    data=buffer,
                    file_name=filename,
                    mime="text/csv"
                )
                
                # Cost estimation
                input_cost = (total_input_tokens / 1_000_000) * 0.15
                output_cost = (total_output_tokens / 1_000_000) * 0.60
                total_cost = input_cost + output_cost
                
                st.markdown("### 📊 Usage Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Input Tokens", f"{total_input_tokens:,}")
                with col2:
                    st.metric("Output Tokens", f"{total_output_tokens:,}")
                with col3:
                    st.metric("Estimated Cost", f"${total_cost:.4f}")
                
                # Show sample matches
                st.subheader("Sample Matches")
                for idx, row in matching_rows.head(5).iterrows():
                    st.write(f"• {row[selected_column]}")
            else:
                st.info("No matching entries found. Try a different vertical or check your data.")

st.markdown("---")
st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
