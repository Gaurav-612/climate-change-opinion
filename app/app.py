import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"  # Update this if your FastAPI is hosted elsewhere

st.title("Climate Change Opinions Analyzer")

query = st.text_input("Enter your query:")
top_k = st.slider("Number of results", min_value=1, max_value=100, value=10)

if st.button("Search"):
    response = requests.post(f"{API_URL}/query", json={"query": query, "top_k": top_k})
    if response.status_code == 200:
        results = response.json()
        for match in results['matches']:
            st.write(f"Score: {match['score']}")
            st.write(f"Post Title: {match['metadata']['post_title']}")
            st.write(f"Subreddit: {match['metadata']['subreddit']}")
            st.write(f"Author: {match['metadata']['author_name']}")
            st.write("---")
    else:
        st.error("An error occurred while querying the data.")

if st.button("Show Index Stats"):
    response = requests.get(f"{API_URL}/stats")
    if response.status_code == 200:
        stats = response.json()
        st.write(f"Total vectors: {stats['total_vector_count']}")
        st.write(f"Dimensions: {stats['dimension']}")
    else:
        st.error("An error occurred while fetching index stats.")