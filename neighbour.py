import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard, euclidean

# Load preprocessed data
@st.cache_resource
def load_processed_data():
    data = pd.read_pickle("processed_data.pkl")
    combined_embeddings = pd.read_pickle("combined_embeddings.pkl").values
    known_embeddings = pd.read_pickle("known_embeddings.pkl").values
    pivot = pd.read_pickle("pivot.pkl")
    horse_names = pd.read_csv("horse_names_autocomplete.csv", header=None)[0].tolist()
    return data, combined_embeddings, known_embeddings, pivot, horse_names

def calculate_similarity(option, active_vector, embeddings):
    if option == "Euclidean Distance":
        # Compute Euclidean distances
        distances = np.linalg.norm(embeddings - active_vector, axis=1)
        return distances
    elif option == "Cosine Similarity":
        # Compute Cosine similarities (higher is better, so we invert it for sorting)
        similarities = cosine_similarity(active_vector, embeddings).flatten()
        distances = 1 - similarities
        return distances
    elif option == "Jaccard Similarity":
        # Compute Jaccard similarity for binary data
        # (Assume embeddings are binary for this similarity measure)
        distances = [jaccard(active_vector.flatten(), embedding) for embedding in embeddings]
        return np.array(distances)

def main():
    st.title("Horse Matching App")
    data, combined_embeddings, known_embeddings, pivot, horse_names = load_processed_data()

    # Horse name input with autocomplete
    horse_name = st.text_input("Enter Horse Name:", "").lower().strip().replace("'", "")

    # Autocomplete suggestion
    suggestions = [name for name in horse_names if horse_name in name]
    if len(suggestions) > 0:
        horse_name = st.selectbox("Suggestions:", suggestions)

    # Similarity measure toggle
    similarity_option = st.radio(
        "Choose Similarity Measure:",
        options=["Euclidean Distance", "Cosine Similarity", "Jaccard Similarity"]
    )

    if horse_name and horse_name in pivot.index:
        st.write(f"Matches for **{horse_name.title()}**")

        # Locate the active horse's vector
        active_index = pivot.index.get_loc(horse_name)
        active_vector = combined_embeddings[active_index].reshape(1, -1)

        # Calculate similarity
        distances = calculate_similarity(similarity_option, active_vector, known_embeddings)

        # Get top 5 matches
        top_indices = distances.argsort()[:5]
        top_matches = pivot.index[known_embeddings[top_indices]]

        for i, match_name in enumerate(top_matches):
            st.write(f"**Rank {i+1}: {match_name.title()}** (Distance: {distances[top_indices[i]]:.4f})")
    else:
        st.write("Horse not found. Please try another name.")

if __name__ == "__main__":
    main()
