import streamlit as st
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

# Function to process and cluster report names
def cluster_report_names(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Report Name'])
    distance_matrix = pairwise_distances(X, metric='cosine')
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, affinity='precomputed', linkage='average')
    clustering.fit(distance_matrix)
    df['Group ID'] = clustering.labels_
    return df

# Streamlit app
st.title("Report Name Clustering")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write("Uploaded DataFrame:")
    st.write(df)

    # Process the dataframe to cluster report names
    clustered_df = cluster_report_names(df)

    # Display the dataframe with Group IDs
    st.write("DataFrame with Group IDs:")
    st.write(clustered_df)

    # Provide a download link for the new dataframe
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(clustered_df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='grouped_report_names.csv',
        mime='text/csv',
    )
