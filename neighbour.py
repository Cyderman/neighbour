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
    horse_names = pd.read_csv("horse_names_autocomplete.csv", header=None)[0].dropna().astype(str).tolist()

    return data, combined_embeddings, known_embeddings, pivot, horse_names


def calculate_similarity(option, active_vector, embeddings, additional_features=None):
    if option == "Euclidean Distance":
        # Normalize embeddings for fairness
        embeddings_norm = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
        active_vector_norm = (active_vector - embeddings.mean(axis=0)) / embeddings.std(axis=0)
        distances = np.linalg.norm(embeddings_norm - active_vector_norm, axis=1)
        return distances

    elif option == "Cosine Similarity":
        # Normalize embeddings for cosine similarity
        active_vector_norm = active_vector / np.linalg.norm(active_vector)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = cosine_similarity(active_vector_norm, embeddings_norm).flatten()
        distances = 1 - similarities  # Convert similarity to distance
        return distances

    elif option == "Jaccard Similarity":
        # Calculate Jaccard similarity based on feature overlap
        if additional_features is None:
            raise ValueError("Additional features are required for Jaccard similarity.")
        distances = [
            jaccard(active_vector.flatten(), embedding) for embedding in additional_features
        ]
        return np.array(distances)

    else:
        raise ValueError("Invalid similarity option.")


def main():
    st.title("Horse Matching App")

    # Load data
    data, combined_embeddings, known_embeddings, pivot, horse_names = load_processed_data()

    # Stop the app if data couldn't be loaded
    if data is None:
        st.stop()

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
        distances = calculate_similarity(
            similarity_option,
            active_vector,
            known_embeddings,
            additional_features=data['features'] if similarity_option == "Jaccard Similarity" else None
        )

        # Get top 5 matches
        top_indices = distances.argsort()[:5]

        # Display results with metadata
        for i, idx in enumerate(top_indices):
            match_name = pivot.index[idx]
            match_data = data.loc[match_name]  # Access metadata
            st.write(f"**Rank {i + 1}: {match_name.title()}**")
            st.write(f"- Grade: {match_data.get('grade', 'N/A')}")
            st.write(f"- Start: {match_data.get('start', 'N/A')}")
            st.write(f"- Speed: {match_data.get('speed', 'N/A')}")
            st.write(f"- Stamina: {match_data.get('stamina', 'N/A')}")
            st.write(f"- Finish: {match_data.get('finish', 'N/A')}")
            st.write(f"- Heart: {match_data.get('heart', 'N/A')}")
            st.write(f"- Temper: {match_data.get('temper', 'N/A')}")
            st.write(f"- [Profile Link](URL/{match_name})")
            st.write(f"- Distance: {distances[idx]:.4f}")

    else:
        st.write("Horse not found. Please try another name.")


if __name__ == "__main__":
    main()
