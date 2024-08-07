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
    # Replace empty, null, or blank 'Search Path' with 'no name'
    df['Search Path'] = df['Search Path'].fillna('').apply(lambda x: 'no name' if x.strip() == '' else x)
    
    extracted_data = [extract_levels(path) for path in df['Search Path']]
    extracted_df = pd.DataFrame(extracted_data)
    # Keep the input columns
    extracted_df = pd.concat([df.reset_index(drop=True), extracted_df], axis=1)
    cols = [col for col in extracted_df.columns if col not in ['reportName', 'originalPath']] + ['reportName', 'originalPath']
    extracted_df = extracted_df[cols]
    return extracted_df

def cluster_report_names(df):
    # Ensure all entries in 'reportName' are strings and fill missing values
    df['reportName'] = df['reportName'].astype(str).fillna('')

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
        'Jira', 'teradata', 'cleanup', 'bkp', 'copy', 'testing', '(1)','Workbook Report'
    ]
    path = replace_folder_keywords(path)
    path_lower = path.lower()
    for keyword in flag_keywords:
        if keyword.lower() in path_lower:
            return 'yes', keyword
    return 'no', ''

def assign_business_unit(search_path):
    if isinstance(search_path, str):
        business_units = {
            'Inventory': ['inventory', 'stock', 'warehouse', 'storage', 'supply chain', 'material', 'SKU', 'capacity', 'demand planning'],
            'Customer': ['customer', 'client', 'user', 'consumer', 'service'],
            'Sales': ['sales', 'revenue', 'orders', 'transactions', 'deals', 'sell out', 'targets', 'dollars'],
            'Marketing': ['marketing', 'advertisement', 'campaign', 'promotion', 'branding'],
            'Manufacturing': ['manufacturing', 'production', 'assembly', 'factory', 'plant'],
            'Human Resources (HR)': ['HR', 'human resources', 'employee', 'staff', 'recruitment', 'payroll'],
            'Finance': ['finance', 'accounting', 'budget', 'expenditure', 'cost', 'profit', 'loss', 'billing', 'cash', 'invoice'],
            'Research and Development (R&D)': ['R&D', 'research', 'development', 'innovation', 'laboratory', 'testing'],
            'Quality Assurance (QA)': ['QA', 'quality assurance', 'inspection', 'compliance', 'standards'],
            'IT and Support': ['IT', 'information technology', 'support', 'helpdesk', 'infrastructure'],
            'Logistics': ['logistics', 'transportation', 'shipping', 'delivery', 'fleet', 'shipment', 'transport', 'freight', 'cargo', 'fulfillment', 'fulfilment'],
            'Procurement': ['procurement', 'purchasing', 'supplier', 'vendor', 'sourcing'],
            'Legal': ['legal', 'compliance', 'regulation', 'contracts', 'law', 'claims'],
            'Miscellaneous': ['Amazon', 'travel', 'locations']
        }
        
        for bu, keywords in business_units.items():
            if any(keyword in search_path.lower() for keyword in keywords):
                return bu
    return 'Other'

def main():
    st.title("Cognos BI Environment Extractor & Report Rationalization")
    st.write("Upload a CSV file with search paths to extract levels & rationalize them dynamically.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            extracted_df = process_file(uploaded_file)
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return

        try:
            extracted_df = cluster_report_names(extracted_df)
        except Exception as e:
            st.error(f"Error clustering report names: {e}")
            return

        cols = [col for col in extracted_df.columns if col not in ['reportGroupId', 'originalPath']] + ['reportGroupId', 'originalPath']
        extracted_df = extracted_df[cols]

        extracted_df['Region Assigner'] = extracted_df.apply(concat_first_words, axis=1)
        extracted_df['Region'] = extracted_df['Region Assigner'].apply(assign_region)
        extracted_df['Flag for Decommission'], extracted_df['reasonForFlagOfDecommission'] = zip(*extracted_df['originalPath'].apply(check_flags))
        extracted_df['Business Unit'] = extracted_df['Search Path'].apply(assign_business_unit)

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
