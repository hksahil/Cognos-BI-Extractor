import streamlit as st
import pandas as pd
import re

def extract_levels(search_path):
    # Define the regex patterns to capture everything inside double quotes and single quotes
    pattern_double_quotes = re.compile(r'"([^"]*)"')
    pattern_single_quotes = re.compile(r"'([^']*)'")

    # Find all matches
    matches_double = pattern_double_quotes.findall(search_path)
    matches_single = pattern_single_quotes.findall(search_path)

    # Combine matches while maintaining order
    matches = []
    last_pos = 0
    for match in re.finditer(r'"([^"]*)"|\'([^\']*)\'', search_path):
        if match.group(1):
            matches.append(match.group(1))
        elif match.group(2):
            matches.append(match.group(2))

    # Create a dictionary with dynamic column names
    data = {f'Level {i+1}': match for i, match in enumerate(matches[:-1])}
    if matches:
        data['Report Name'] = matches[-1]
    data['Original Path'] = search_path
    return data

def process_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    extracted_data = [extract_levels(path) for path in df['Search Path']]
    extracted_df = pd.DataFrame(extracted_data)
    # Ensure Report Name is the last column before Original Path
    cols = [col for col in extracted_df.columns if col not in ['Report Name', 'Original Path']] + ['Report Name', 'Original Path']
    extracted_df = extracted_df[cols]
    return extracted_df

def main():
    st.title("Path Levels Extractor")
    st.write("Upload a CSV file with search paths to extract levels dynamically.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        st.write("Uploaded file:")
        st.write(uploaded_file.name)
        
        extracted_df = process_file(uploaded_file)
        
        st.write("Extracted Data:")
        st.dataframe(extracted_df)
        
        st.download_button(
            label="Download Extracted Levels as CSV",
            data=extracted_df.to_csv(index=False).encode('utf-8'),
            file_name='extracted_levels.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
