import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from openai import OpenAI
import yaml
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

# SCRAPING FUNCTIONS
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

def scrape_website(i, url, df, url_column):
    if not url.lower().startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        content = extract_main_text(url)
        if content and len(content.split()) >= 30:
            result = content
            status = "SUCCESS"
        else:
            result = ""
            status = "INSUFFICIENT_CONTENT"
    except Exception as e:
        result = ""
        status = f"ERROR: {str(e)}"

    with df_lock:
        df.at[i, 'Website Content'] = result
        df.at[i, 'Scrape Status'] = status

    time.sleep(random.uniform(0.5, 1.5))
def fit_ai_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    with torch.no_grad():
            outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return predicted_class_id
# CLASSIFICATION FUNCTIONS
with open("prompts.yaml", "r") as f:
    strings = yaml.safe_load(f)
classification_prompt = strings.get("classification_prompt", "")
service_classification_prompt = strings.get("service_classification_prompt","")
url_fallback_prompt = strings.get("url_fallback_prompt","")
service_url_fallback_prompt = strings.get("service_url_fallback_prompt","")
fit_prompt = strings.get("fit_prompt","")
fit_fallback_prompt = strings.get("fit_fallback_prompt","")
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

def classify_content(i, content, url, df, content_column, classification_type="vertical"):
    with api_semaphore:
        try:
            if content and len(content.split()) >= 30:
                trimmed = ' '.join(content.split()[:800])
                if classification_type == "vertical":
                    prompt = classification_prompt.format(content=trimmed)
                    result_column = 'Vertical Focus OpenAI'
                    fallback_prompt = url_fallback_prompt.format(url=url)
                elif classification_type == "fit":
                    prompt = fit_prompt.format(content = trimmed)
                    result_column = "Housatonic Fit OpenAI"
                    fallback_prompt = fit_fallback_prompt.format(url = url)
                else:  # service
                    prompt = service_classification_prompt.format(content=trimmed)
                    result_column = 'Service OpenAI'
                    fallback_prompt = service_url_fallback_prompt.format(url=url)
                
                result = call_openai(prompt)
                if result == "ERROR":
                    result = call_openai(fallback_prompt) + " *"
            else:
                if classification_type == "vertical":
                    result_column = 'Vertical Focus OpenAI'
                    fallback_prompt = url_fallback_prompt.format(url=url)
                elif classification_type == "fit":
                    result_column = 'Housatonic Fit OpenAI'
                    fallback_prompt = fit_fallback_prompt.format(url = url)
                else:  # service
                    result_column = 'Service OpenAI'
                    fallback_prompt = service_url_fallback_prompt.format(url=url)
                
                result = call_openai(fallback_prompt) + " *"

            result = result if result and result != "ERROR" else "GENERATION ERROR"
        except Exception as e:
            result = f"[ERROR] {str(e)}"

            result_column = 'Service OpenAI' 
            if classification_type == "vertical":
                result_column = 'Vertical Focus OpenAI'
            if classification_type == "fit":
                result_column = 'Housatonic Fit OpenAI'

        with df_lock:
            df.at[i, result_column] = result

        time.sleep(random.uniform(0.5, 1.5))

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

# COMPANY QUERY PROMPTS
def create_company_query_prompt(website_contents, user_question):
    content_entries = "\n".join([f"{i+1}. {content[:500]}..." if len(content) > 500 else f"{i+1}. {content}" 
                                for i, content in enumerate(website_contents)])
    
    return f"""
You are a business analysis expert. Based on the website content provided below, answer this question: "{user_question}"

For each numbered website content entry, determine if the company matches the criteria in the question. Be inclusive rather than overly strict - if a company reasonably fits the description, include it.

Here are the website contents to analyze:
{content_entries}

Please respond with ONLY a JSON array of true/false values, where true means the company matches the criteria and false means it doesn't. The array should have exactly {len(website_contents)} elements corresponding to the numbered entries above.

Example response format: [true, false, true, false, ...]
"""

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

def classify_companies_by_query(website_contents, user_question):
    """Classify companies based on website content and user question"""
    prompt = create_company_query_prompt(website_contents, user_question)
    
    try:
        response = call_openai(prompt, max_tokens=1000)
        
        # Try to parse JSON
        try:
            results = json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON array in response
            import re
            json_match = re.search(r'\[[\s\S]*?\]', response)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    results = json.loads(json_str)
                except:
                    return [False] * len(website_contents)
            else:
                return [False] * len(website_contents)
        
        # Validate results
        if not isinstance(results, list) or len(results) != len(website_contents):
            return [False] * len(website_contents)
        
        # Convert to boolean
        return [bool(r) for r in results]
        
    except Exception as e:
        return [False] * len(website_contents)

def clean_vertical_entry(entry):
    """Clean vertical entry by removing asterisks and extra whitespace"""
    if pd.isna(entry):
        return ""
    return str(entry).replace('*', '').strip()

def clean_content_entry(entry):
    """Clean website content entry"""
    if pd.isna(entry):
        return ""
    return str(entry).strip()

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
    ["Website Scraper", "Vertical Focus Classifier", "Service Classifier", "Vertical Focus Filter", "Company Query Tool", "Housatonic Fit Tool"]
)

st.markdown("---")

# TOOL 1: WEBSITE SCRAPER
if tool_option == "Website Scraper":
    st.header("🌐 Website Scraper")
    st.markdown("Upload a CSV with website URLs to scrape their content. This content can then be used in other tools.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="scraper_upload")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Column selection for URL
        column_options = list(df.columns)
        
        # Try to find common URL column names and suggest them first
        url_column_suggestions = []
        for candidate in ["Account: Website", "URL", "Website", "Domain", "Site", "Link"]:
            if candidate in column_options:
                url_column_suggestions.append(candidate)
        
        # Put suggestions at the beginning, then add remaining columns
        remaining_columns = [col for col in column_options if col not in url_column_suggestions]
        ordered_columns = url_column_suggestions + remaining_columns
        
        selected_url_column = st.selectbox(
            "Select the column containing website URLs:",
            ordered_columns,
            index=0,
            help="Choose the column that contains the website URLs you want to scrape"
        )
        
        # Show preview of selected column
        if selected_url_column:
            st.subheader("Preview of Selected Column")
            preview_data = df[selected_url_column].head(5).tolist()
            for i, url in enumerate(preview_data, 1):
                st.write(f"{i}. {url}")
            
            # Show count of non-empty URLs
            non_empty_urls = df[selected_url_column].dropna().astype(str)
            non_empty_urls = non_empty_urls[non_empty_urls.str.strip() != '']
            st.info(f"Found {len(non_empty_urls)} non-empty URLs to scrape")
        
        # Process button
        if st.button("🚀 Start Scraping", type="primary"):
            if "scraped_df" not in st.session_state:
                # Add new columns if they don't exist
                if 'Website Content' not in df.columns:
                    df['Website Content'] = ''
                if 'Scrape Status' not in df.columns:
                    df['Scrape Status'] = ''

                df['Website Content'] = df['Website Content'].astype(str)
                df['Scrape Status'] = df['Scrape Status'].astype(str)

                # Filter out empty URLs
                urls_to_process = df[df[selected_url_column].notna() & (df[selected_url_column].astype(str).str.strip() != '')].index.tolist()
                total_urls = len(urls_to_process)
                
                if total_urls == 0:
                    st.error("No valid URLs found in the selected column.")
                    st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                with st.spinner("Scraping websites..."):
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = []
                        
                        for i in urls_to_process:
                            url = str(df.at[i, selected_url_column]).strip()
                            if url and url != 'nan':
                                futures.append(executor.submit(scrape_website, i, url, df, selected_url_column))

                        for count, future in enumerate(as_completed(futures)):
                            future.result()
                            elapsed = int(time.time() - start_time)
                            remaining = int((elapsed / (count + 1)) * (total_urls - count - 1)) if count > 0 else 0
                            status_text.text(f"Scraped {count + 1}/{total_urls} | ~{remaining}s remaining")
                            progress_bar.progress((count + 1) / total_urls)

                st.session_state.scraped_df = df

            else:
                df = st.session_state.scraped_df

            st.success("✅ Scraping complete!")
            
            # Show scraping statistics
            success_count = len(df[df['Scrape Status'] == 'SUCCESS'])
            error_count = len(df[df['Scrape Status'].str.contains('ERROR', na=False)])
            insufficient_content_count = len(df[df['Scrape Status'] == 'INSUFFICIENT_CONTENT'])
            
            st.subheader("Scraping Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Successful", success_count)
            with col2:
                st.metric("⚠️ Insufficient Content", insufficient_content_count)
            with col3:
                st.metric("❌ Errors", error_count)
            
            # Display results preview
            st.subheader("Results Preview")
            results_preview = df[[selected_url_column, 'Scrape Status', 'Website Content']].head(10)
            # Truncate content for display
            results_preview['Website Content (Preview)'] = results_preview['Website Content'].apply(
                lambda x: x[:200] + "..." if len(str(x)) > 200 else x
            )
            st.dataframe(results_preview[[selected_url_column, 'Scrape Status', 'Website Content (Preview)']])
            
            # Download button
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="📥 Download Scraped Results CSV",
                data=buffer,
                file_name=f"scraped_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# TOOL 2: VERTICAL FOCUS CLASSIFIER
elif tool_option == "Vertical Focus Classifier":
    st.header("🔍 Vertical Focus Classifier")
    st.markdown("Upload a CSV with website content to classify each company by industry vertical using ChatGPT.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="classifier_upload")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Column selection for content
        column_options = list(df.columns)
        
        # Try to find common content column names and suggest them first
        content_column_suggestions = []
        for candidate in ["Website Content", "Content", "Scraped Content", "Description", "About"]:
            if candidate in column_options:
                content_column_suggestions.append(candidate)
        
        # Put suggestions at the beginning, then add remaining columns
        remaining_columns = [col for col in column_options if col not in content_column_suggestions]
        ordered_columns = content_column_suggestions + remaining_columns
        
        selected_content_column = st.selectbox(
            "Select the column containing website content:",
            ordered_columns,
            index=0,
            help="Choose the column that contains the website content you want to classify"
        )
        
        # Optional: URL column for fallback
        url_column_options = ["None"] + column_options
        url_column_suggestions = []
        for candidate in ["Account: Website", "URL", "Website", "Domain", "Site", "Link"]:
            if candidate in column_options:
                url_column_suggestions.append(candidate)
        
        url_ordered_columns = ["None"] + url_column_suggestions + [col for col in column_options if col not in url_column_suggestions]
        
        selected_url_column = st.selectbox(
            "Select URL column (optional - used for fallback when content is insufficient):",
            url_ordered_columns,
            index=0,
            help="Optional: Choose the column with URLs for fallback classification when content is insufficient"
        )
        
        # Show preview of selected columns
        if selected_content_column:
            st.subheader("Preview of Selected Content Column")
            preview_data = df[selected_content_column].head(3).tolist()
            for i, content in enumerate(preview_data, 1):
                content_preview = str(content)[:200] + "..." if len(str(content)) > 200 else str(content)
                st.write(f"{i}. {content_preview}")
            
            # Show count of non-empty content
            non_empty_content = df[selected_content_column].dropna().astype(str)
            non_empty_content = non_empty_content[non_empty_content.str.strip() != '']
            st.info(f"Found {len(non_empty_content)} non-empty content entries to classify")
        
        # Process button
        if st.button("🚀 Start Classification", type="primary"):
            if "classified_df" not in st.session_state:
                # Add new column if it doesn't exist
                if 'Vertical Focus OpenAI' not in df.columns:
                    df['Vertical Focus OpenAI'] = ''

                df['Vertical Focus OpenAI'] = df['Vertical Focus OpenAI'].astype(str)

                # Filter out empty content
                content_to_process = df[df[selected_content_column].notna() & (df[selected_content_column].astype(str).str.strip() != '')].index.tolist()
                total_content = len(content_to_process)
                
                if total_content == 0:
                    st.error("No valid content found in the selected column.")
                    st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                with st.spinner("Classifying content..."):
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = []
                        
                        for i in content_to_process:
                            content = str(df.at[i, selected_content_column]).strip()
                            url = ""
                            if selected_url_column != "None" and selected_url_column in df.columns:
                                url = str(df.at[i, selected_url_column]).strip()
                            
                            if content and content != 'nan':
                                futures.append(executor.submit(classify_content, i, content, url, df, selected_content_column))

                        for count, future in enumerate(as_completed(futures)):
                            future.result()
                            elapsed = int(time.time() - start_time)
                            remaining = int((elapsed / (count + 1)) * (total_content - count - 1)) if count > 0 else 0
                            status_text.text(f"Classified {count + 1}/{total_content} | ~{remaining}s remaining")
                            progress_bar.progress((count + 1) / total_content)

                st.session_state.classified_df = df
                st.session_state.token_stats = (total_input_tokens, total_output_tokens)

            else:
                df = st.session_state.classified_df
                total_input_tokens, total_output_tokens = st.session_state.token_stats

            st.success("✅ Classification complete!")
            
            # Display results preview
            st.subheader("Results Preview")
            results_preview = df[[selected_content_column, 'Vertical Focus OpenAI']].head(10)
            # Truncate content for display
            results_preview['Content (Preview)'] = results_preview[selected_content_column].apply(
                lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x)
            )
            st.dataframe(results_preview[['Content (Preview)', 'Vertical Focus OpenAI']])
            
            # Download button
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="📥 Download Classification Results CSV",
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
# TOOL 3: SERVICE CLASSIFIER
elif tool_option == "Service Classifier":
    st.header("🛠️ Service Classifier")
    st.markdown("Upload a CSV with website content to classify each company by **primary service** (WHAT they do) using ChatGPT.")
    st.info("**Service** = What the company does or provides (e.g., Software Development, Consulting, Marketing)")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="service_classifier_upload")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Column selection for content
        column_options = list(df.columns)
        
        # Try to find common content column names and suggest them first
        content_column_suggestions = []
        for candidate in ["Website Content", "Content", "Scraped Content", "Description", "About"]:
            if candidate in column_options:
                content_column_suggestions.append(candidate)
        
        # Put suggestions at the beginning, then add remaining columns
        remaining_columns = [col for col in column_options if col not in content_column_suggestions]
        ordered_columns = content_column_suggestions + remaining_columns
        
        selected_content_column = st.selectbox(
            "Select the column containing website content:",
            ordered_columns,
            index=0,
            help="Choose the column that contains the website content you want to classify"
        )
        
        # Optional: URL column for fallback
        url_column_options = ["None"] + column_options
        url_column_suggestions = []
        for candidate in ["Account: Website", "URL", "Website", "Domain", "Site", "Link"]:
            if candidate in column_options:
                url_column_suggestions.append(candidate)
        
        url_ordered_columns = ["None"] + url_column_suggestions + [col for col in column_options if col not in url_column_suggestions]
        
        selected_url_column = st.selectbox(
            "Select URL column (optional - used for fallback when content is insufficient):",
            url_ordered_columns,
            index=0,
            help="Optional: Choose the column with URLs for fallback classification when content is insufficient"
        )
        
        # Show preview of selected columns
        if selected_content_column:
            st.subheader("Preview of Selected Content Column")
            preview_data = df[selected_content_column].head(3).tolist()
            for i, content in enumerate(preview_data, 1):
                content_preview = str(content)[:200] + "..." if len(str(content)) > 200 else str(content)
                st.write(f"{i}. {content_preview}")
            
            # Show count of non-empty content
            non_empty_content = df[selected_content_column].dropna().astype(str)
            non_empty_content = non_empty_content[non_empty_content.str.strip() != '']
            st.info(f"Found {len(non_empty_content)} non-empty content entries to classify")
        
        # Process button
        if st.button("🚀 Start Service Classification", type="primary"):
            if "service_classified_df" not in st.session_state:
                # Add new column if it doesn't exist
                if 'Service OpenAI' not in df.columns:
                    df['Service OpenAI'] = ''

                df['Service OpenAI'] = df['Service OpenAI'].astype(str)

                # Filter out empty content
                content_to_process = df[df[selected_content_column].notna() & (df[selected_content_column].astype(str).str.strip() != '')].index.tolist()
                total_content = len(content_to_process)
                
                if total_content == 0:
                    st.error("No valid content found in the selected column.")
                    st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                with st.spinner("Classifying services..."):
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = []
                        
                        for i in content_to_process:
                            content = str(df.at[i, selected_content_column]).strip()
                            url = ""
                            if selected_url_column != "None" and selected_url_column in df.columns:
                                url = str(df.at[i, selected_url_column]).strip()
                            
                            if content and content != 'nan':
                                futures.append(executor.submit(classify_content, i, content, url, df, selected_content_column, "service"))

                        for count, future in enumerate(as_completed(futures)):
                            future.result()
                            elapsed = int(time.time() - start_time)
                            remaining = int((elapsed / (count + 1)) * (total_content - count - 1)) if count > 0 else 0
                            status_text.text(f"Classified {count + 1}/{total_content} | ~{remaining}s remaining")
                            progress_bar.progress((count + 1) / total_content)

                st.session_state.service_classified_df = df
                st.session_state.service_token_stats = (total_input_tokens, total_output_tokens)

            else:
                df = st.session_state.service_classified_df
                total_input_tokens, total_output_tokens = st.session_state.service_token_stats

            st.success("✅ Service Classification complete!")
            
            # Display results preview
            st.subheader("Results Preview")
            results_preview = df[[selected_content_column, 'Service OpenAI']].head(10)
            # Truncate content for display
            results_preview['Content (Preview)'] = results_preview[selected_content_column].apply(
                lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x)
            )
            st.dataframe(results_preview[['Content (Preview)', 'Service OpenAI']])
            
            # Download button
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="📥 Download Service Classification Results CSV",
                data=buffer,
                file_name=f"service_classified_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
# TOOL 4: VERTICAL FOCUS FILTER
elif tool_option == "Vertical Focus Filter":
    st.header("🔧 Vertical Focus Filter")
    st.markdown("Upload a CSV with vertical focus data and filter rows that match a specific vertical using ChatGPT classification.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="filter_upload")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Column selection
        column_options = list(df.columns)
        
        # Try to find common vertical column names
        vertical_column_suggestions = []
        for candidate in ["Vertical Focus", "Vertical Focus OpenAI", "Industry", "Vertical", "Category"]:
            if candidate in column_options:
                vertical_column_suggestions.append(candidate)
        
        remaining_columns = [col for col in column_options if col not in vertical_column_suggestions]
        ordered_columns = vertical_column_suggestions + remaining_columns
        
        selected_column = st.selectbox(
            "Select the column containing vertical focus data:",
            ordered_columns,
            index=0
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

# TOOL 5: COMPANY QUERY TOOL
elif tool_option == "Company Query Tool":
    st.header("💬 Company Query Tool")
    st.markdown("Upload a CSV with website content and ask questions about the companies (e.g., 'Find all SaaS companies', 'Which companies offer consulting services?').")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="query_upload")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Column selection for website content
        column_options = list(df.columns)
        default_col = "Website Content" if "Website Content" in column_options else column_options[0]
        
        selected_column = st.selectbox(
            "Select the column containing website content:",
            column_options,
            index=column_options.index(default_col)
        )
        
        # Question input
        user_question = st.text_input(
            "Ask a question about the companies:",
            placeholder="e.g., Find all SaaS companies, Which companies offer consulting services?, Show me e-commerce businesses"
        )
        
        # Example questions
        st.markdown("**Example questions:**")
        st.markdown("- Find all SaaS companies")
        st.markdown("- Which companies offer consulting services?")
        st.markdown("- Show me e-commerce businesses")
        st.markdown("- Find companies that provide software solutions")
        st.markdown("- Which companies are in the healthcare industry?")
        
        if user_question and st.button("🔍 Search Companies", type="primary"):
            # Clean the data
            df[selected_column] = df[selected_column].apply(clean_content_entry)
            non_empty_df = df[df[selected_column] != ''].copy()
            
            if len(non_empty_df) == 0:
                st.error("No valid website content found in the selected column.")
                st.stop()
            
            # Reset token counters for this operation
            total_input_tokens = 0
            total_output_tokens = 0
            
            # Process in batches
            batch_size = 10  # Smaller batch size due to longer content
            content_list = non_empty_df[selected_column].tolist()
            total_batches = (len(content_list) + batch_size - 1) // batch_size
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_matches = []
            
            with st.spinner("Analyzing companies using ChatGPT..."):
                for i in range(0, len(content_list), batch_size):
                    batch = content_list[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    
                    status_text.text(f"Processing batch {batch_num}/{total_batches}...")
                    
                    batch_results = classify_companies_by_query(batch, user_question)
                    all_matches.extend(batch_results)
                    
                    progress_bar.progress(batch_num / total_batches)
                    time.sleep(0.8)  # Slightly longer rate limiting for content analysis
            
            # Filter results
            non_empty_df['chatgpt_match'] = all_matches
            matching_rows = non_empty_df[non_empty_df['chatgpt_match'] == True].copy()
            matching_rows = matching_rows.drop('chatgpt_match', axis=1)
            
            st.success(f"✅ Found {len(matching_rows)} companies matching: '{user_question}'")
            
            if len(matching_rows) > 0:
                # Display results preview
                st.subheader("Matching Companies Preview")
                preview_columns = [col for col in matching_rows.columns if col != selected_column][:5]  # Show first 5 columns excluding content
                if preview_columns:
                    st.dataframe(matching_rows[preview_columns].head(10))
                else:
                    st.dataframe(matching_rows.head(10))
                
                # Download button
                buffer = io.BytesIO()
                matching_rows.to_csv(buffer, index=False)
                buffer.seek(0)
                
                safe_question = user_question.replace(' ', '_').replace('?', '').replace('/', '_')[:30]
                filename = f"query_results_{safe_question}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                st.download_button(
                    label="📥 Download Query Results",
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
                
                # Show sample matches with company names if available
                st.subheader("Sample Matches")
                company_name_cols = [col for col in matching_rows.columns if 'name' in col.lower() or 'company' in col.lower()]
                if company_name_cols:
                    name_col = company_name_cols[0]
                    for idx, row in matching_rows.head(5).iterrows():
                        st.write(f"• {row[name_col]}")
                else:
                    st.write(f"Found {len(matching_rows)} matching companies. Download the results to see all matches.")
            else:
                st.info("No matching companies found. Try rephrasing your question or check your data.")
# TOOL 6: HOUSATONIC FIT CLASSIFIER
elif tool_option == "Housatonic Fit Tool":
    st.header("🔍 Housatonic Fit Tool")
    st.markdown("Upload a CSV with website content to classify each company by if it meets Housatonic's list builder criteria.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="classifier_upload")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Column selection for content
        column_options = list(df.columns)
        
        # Try to find common content column names and suggest them first
        content_column_suggestions = []
        for candidate in ["Website Content", "Content", "Scraped Content", "Description", "About"]:
            if candidate in column_options:
                content_column_suggestions.append(candidate)
        
        # Put suggestions at the beginning, then add remaining columns
        remaining_columns = [col for col in column_options if col not in content_column_suggestions]
        ordered_columns = content_column_suggestions + remaining_columns
        
        selected_content_column = st.selectbox(
            "Select the column containing website content:",
            ordered_columns,
            index=0,
            help="Choose the column that contains the website content you want to classify"
        )
        
        # Optional: URL column for fallback
        url_column_options = ["None"] + column_options
        url_column_suggestions = []
        for candidate in ["Account: Website", "URL", "Website", "Domain", "Site", "Link"]:
            if candidate in column_options:
                url_column_suggestions.append(candidate)
        
        url_ordered_columns = ["None"] + url_column_suggestions + [col for col in column_options if col not in url_column_suggestions]
        
        selected_url_column = st.selectbox(
            "Select URL column (optional - used for fallback when content is insufficient):",
            url_ordered_columns,
            index=0,
            help="Optional: Choose the column with URLs for fallback classification when content is insufficient"
        )
        
        # Show preview of selected columns
        if selected_content_column:
            st.subheader("Preview of Selected Content Column")
            preview_data = df[selected_content_column].head(3).tolist()
            for i, content in enumerate(preview_data, 1):
                content_preview = str(content)[:200] + "..." if len(str(content)) > 200 else str(content)
                st.write(f"{i}. {content_preview}")
            
            # Show count of non-empty content
            non_empty_content = df[selected_content_column].dropna().astype(str)
            non_empty_content = non_empty_content[non_empty_content.str.strip() != '']
            st.info(f"Found {len(non_empty_content)} non-empty content entries to classify")
        
        # Process button
        if st.button("🚀 Start Classification", type="primary"):
            if "classified_df" not in st.session_state:
                # Add new column if it doesn't exist
                if 'Housatonic Fit OpenAI' not in df.columns:
                    df['Housatonic Fit OpenAI'] = ''

                df['Housatonic Fit OpenAI'] = df['Housatonic Fit OpenAI'].astype(str)

                # Filter out empty content
                content_to_process = df[df[selected_content_column].notna() & (df[selected_content_column].astype(str).str.strip() != '')].index.tolist()
                total_content = len(content_to_process)
                
                if total_content == 0:
                    st.error("No valid content found in the selected column.")
                    st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                with st.spinner("Classifying content..."):
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = []
                        
                        for i in content_to_process:
                            content = str(df.at[i, selected_content_column]).strip()
                            url = ""
                            if selected_url_column != "None" and selected_url_column in df.columns:
                                url = str(df.at[i, selected_url_column]).strip()
                            
                            if content and content != 'nan':
                                futures.append(executor.submit(classify_content, i, content, url, df, selected_content_column, classification_type = "fit"))

                        for count, future in enumerate(as_completed(futures)):
                            future.result()
                            elapsed = int(time.time() - start_time)
                            remaining = int((elapsed / (count + 1)) * (total_content - count - 1)) if count > 0 else 0
                            status_text.text(f"Classified {count + 1}/{total_content} | ~{remaining}s remaining")
                            progress_bar.progress((count + 1) / total_content)

                st.session_state.classified_df = df
                st.session_state.token_stats = (total_input_tokens, total_output_tokens)

            else:
                df = st.session_state.classified_df
                total_input_tokens, total_output_tokens = st.session_state.token_stats

            st.success("✅ Classification complete!")
            
            # Display results preview
            st.subheader("Results Preview")
            results_preview = df[[selected_content_column, 'Housatonic Fit OpenAI']].head(10)
            # Truncate content for display
            results_preview['Content (Preview)'] = results_preview[selected_content_column].apply(
                lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x)
            )
            st.dataframe(results_preview[['Content (Preview)', 'Vertical Focus OpenAI']])
            
            # Download button
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="📥 Download Classification Results CSV",
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

st.markdown("---")
st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
