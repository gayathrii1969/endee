from sentence_transformers import SentenceTransformer
import streamlit as st
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
with open("data.txt", "r") as f:
    data = f.readlines()

# Convert to vectors
vectors = model.encode(data)

# Search function
def search(query):
    query_vec = model.encode([query])[0]

    best_match = ""
    best_score = -1

    for i in range(len(vectors)):
        score = np.dot(query_vec, vectors[i])

        if score > best_score:
            best_score = score
            best_match = data[i]

    return best_match


# Streamlit UI
st.title("📚 AI Student Notes Search")

query = st.text_input("Ask your question:")

if query:
    result = search(query)
    st.success(result)
