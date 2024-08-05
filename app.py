import streamlit as st
import pandas as pd
import re
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

def extract_levels(search_path):
    pattern_double_quotes = re.compile(r'"([^"]*)"')
    pattern_single_quotes = re.compile(r"'([^']*)'")

    matches_double = pattern_double_quotes.findall(search_path)
    matches_single = pattern_single_quotes.findall(search_path)

    matches = []
    last_pos = 0
    for match in re.finditer(r'"([^"]*)"|\'([^\']*)\'', search_path):
        if match.group(1):
            matches.append(match.group(1))
        elif match.group(2):
            matches.append(match.group(2))

    data = {f'Level {i+1}': match for i, match in enumerate(matches[:-1])}
    if matches:
        data['Report Name'] = matches[-1]
    data['Original Path'] = search_path
    return data

def process_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    extracted_data = [extract_levels(path) for path in df['Search Path']]
    extracted_df = pd.DataFrame(extracted_data)
    cols = [col for col in extracted_df.columns if col not in ['Report Name', 'Original Path']] + ['Report Name', 'Original Path']
    extracted_df = extracted_df[cols]
    return extracted_df

def cluster_report_names(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Report Name'])
    distance_matrix = pairwise_distances(X, metric='cosine')
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, affinity='precomputed', linkage='average')
    clustering.fit(distance_matrix)
    df['Report Group ID'] = clustering.labels_
    return df

def main():
    st.title("Cognos BI Environment Extractor & Report Rationalization ")
    st.write("Upload a CSV file with search paths to extract levels & rationalize them dynamically.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        st.write("Uploaded file:")
        st.write(uploaded_file.name)
        
        extracted_df = process_file(uploaded_file)
        
        extracted_df = cluster_report_names(extracted_df)

        cols = [col for col in extracted_df.columns if col not in ['Report Group ID', 'Original Path']] + ['Report Group ID', 'Original Path']
        extracted_df = extracted_df[cols]

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
