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

st.markdown("---")
st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
