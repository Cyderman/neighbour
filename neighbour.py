import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import zipfile
import os

# Load preprocessed data with logic to handle zipped files
@st.cache_resource
def load_processed_data():
    # Check if necessary files exist, if not, unzip them
    required_files = ["processed_data.pkl", "combined_embeddings.pkl", "known_embeddings.pkl", "pivot.pkl", "horse_names_autocomplete.csv"]
    if not all(os.path.exists(file) for file in required_files):
        if os.path.exists("processed_data.zip"):
            with zipfile.ZipFile("processed_data.zip", "r") as zip_ref:
                zip_ref.extractall()  # Extract all files to the current directory
        else:
            st.error("Missing 'processed_data.zip'. Please add it to the repository.")
            return None, None, None, None, None

    # Load data
    data = pd.read_pickle("processed_data.pkl")
    combined_embeddings = pd.read_pickle("combined_embeddings.pkl").values
    known_embeddings = pd.read_pickle("known_embeddings.pkl").values
    pivot = pd.read_pickle("pivot.pkl")
    horse_names = pd.read_csv("horse_names_autocomplete.csv", header=None)[0].dropna().tolist()  # Remove any NaN values

    return data, combined_embeddings, known_embeddings, pivot, horse_names


def calculate_similarity(option, active_vector, embeddings, additional_features=None):
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
        if additional_features is not None:
            distances = [jaccard(active_vector.flatten(), embedding) for embedding in additional_features]
            return np.array(distances)
        else:
            st.warning("Jaccard Similarity requires binary features. Falling back to Euclidean Distance.")
            return np.linalg.norm(embeddings - active_vector, axis=1)


def main():
    st.title("Horse Matching App")

    # Load data
    data, combined_embeddings, known_embeddings, pivot, horse_names = load_processed_data()

    # Stop the app if data couldn't be loaded
    if data is None:
        st.stop()

    # Normalize case for pivot.index and data.index
    pivot.index = pivot.index.str.lower()
    data.index = data.index.str.lower()
    horse_names = [name.lower() for name in horse_names if isinstance(name, str)]

    # Horse name input with autocomplete
    horse_name = st.text_input("Enter Horse Name:", "").lower().strip()

    # Validate that the input is not empty
    if not horse_name:
        st.write("Please enter a valid horse name to search.")
        return  # Stop further processing if the input is blank

    # Autocomplete suggestion
    suggestions = [name for name in horse_names if horse_name in name]
    if suggestions:
        horse_name = st.selectbox("Suggestions:", suggestions)
    else:
        st.write("No suggestions available for the entered name.")
        return  # Stop if no suggestions match

    # Similarity measure toggle
    similarity_option = st.radio(
        "Choose Similarity Measure:",
        options=["Euclidean Distance", "Cosine Similarity", "Jaccard Similarity"]
    )

    # Validate that horse_name is in the pivot index
    if horse_name not in pivot.index:
        st.write("Horse not found in the database. Please try another name.")
        return

    # Proceed with similarity calculations and display results
    st.write(f"Matches for **{horse_name.title()}**")

    # Locate the active horse's vector
    active_index = pivot.index.get_loc(horse_name)
    active_vector = combined_embeddings[active_index].reshape(1, -1)

    # Check if 'features' exists for Jaccard Similarity
    additional_features = data['features'] if 'features' in data.columns else None

    # Calculate similarity
    distances = calculate_similarity(
        similarity_option,
        active_vector,
        known_embeddings,
        additional_features=additional_features
    )

    # Get top 5 matches
    top_indices = distances.argsort()[:5]
    if len(top_indices) == 0:
        st.write("No valid matches found. Please try another name.")
        return

    # Display results with metadata
    for i, idx in enumerate(top_indices):
        match_name = pivot.index[idx]
        if match_name not in data.index:
            st.error(f"Match name '{match_name}' not found in data.index!")
            continue
        match_data = data.loc[match_name]  # Access metadata
        st.write(f"**Rank {i + 1}: {match_name.title()}**")
        st.write(f"- Grade: {match_data.get('grade', 'N/A')}")
        st.write(f"- Start: {match_data.get('start', 'N/A')}")
        st.write(f"- Speed: {match_data.get('speed', 'N/A')}")
        st.write(f"- Stamina: {match_data.get('stamina', 'N/A')}")
        st.write(f"- Finish: {match_data.get('finish', 'N/A')}")
        st.write(f"- Heart: {match_data.get('heart', 'N/A')}")
        st.write(f"- Temper: {match_data.get('temper', 'N/A')}")
        st.write(f"- [View Horse Profile](URL/{match_name})")  # Replace `URL` with the actual URL prefix


if __name__ == "__main__":
    main()
