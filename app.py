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

    data = {f'level{i+1}': match for i, match in enumerate(matches[:-1])}
    if matches:
        data['reportName'] = matches[-1]
    data['originalPath'] = search_path
    return data

def replace_folder_keywords(path):
    folder_keywords = ['folder', 'folder@name', 'latest']
    for keyword in folder_keywords:
        path = path.replace(keyword, '')
    return path

def process_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    extracted_data = [extract_levels(path) for path in df['Search Path']]
    extracted_df = pd.DataFrame(extracted_data)
    cols = [col for col in extracted_df.columns if col not in ['reportName', 'originalPath']] + ['reportName', 'originalPath']
    extracted_df = extracted_df[cols]
    return extracted_df

def cluster_report_names(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['reportName'])
    distance_matrix = pairwise_distances(X, metric='cosine')
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, affinity='precomputed', linkage='average')
    clustering.fit(distance_matrix)
    df['reportGroupId'] = clustering.labels_
    return df

def concat_first_words(row):
    first_words = [row[col].split()[0] for col in row.index if col.startswith('level') and not pd.isna(row[col])]
    return '-'.join(first_words)

def assign_region(concatenated_first_words):
    keywords_to_level2 = {
        'NAT': 'NA',
        'NA': 'NA',
        'EMEA':'EMEA',
        'EU': 'EMEA',
        'Global': 'Global',
        'LA': 'LA',
        'AP': 'AP',
        'APAC': 'AP'
    }
    parts = concatenated_first_words.split('-')
    for part in parts:
        for keyword, region in keywords_to_level2.items():
            if part.lower().endswith(keyword.lower()):
                return region
    return 'Others'

def check_flags(path):
    flag_keywords = [
        'CAM', 'upgrade', 'template', 'temp', 'temporary', 'old data', 'test',
        'remove', 'audit', 'sample', 'Ibm', 'development', 'backup', 'ad hoc', 'adhoc',
        'tableau', 'archive', 'my folder', 'not used', 'old', 'delete', 'archiv', 'obsolete',
        'Jira', 'teradata', 'cleanup', 'bkp', 'copy', 'testing'
    ]
    path = replace_folder_keywords(path)
    path_lower = path.lower()
    for keyword in flag_keywords:
        if keyword.lower() in path_lower:
            return 'yes', keyword
    return 'no', ''

def main():
    st.title("Cognos BI Environment Extractor & Report Rationalization")
    st.write("Upload a CSV file with search paths to extract levels & rationalize them dynamically.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        extracted_df = process_file(uploaded_file)
        extracted_df = cluster_report_names(extracted_df)

        cols = [col for col in extracted_df.columns if col not in ['reportGroupId', 'originalPath']] + ['reportGroupId', 'originalPath']
        extracted_df = extracted_df[cols]

        extracted_df['Region Assigner'] = extracted_df.apply(concat_first_words, axis=1)
        extracted_df['Region'] = extracted_df['Region Assigner'].apply(assign_region)
        extracted_df['Flag for Decommission'], extracted_df['reasonForFlagOfDecommission'] = zip(*extracted_df['originalPath'].apply(check_flags))

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
