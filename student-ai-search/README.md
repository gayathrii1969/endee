# 🚀 AI Student Notes Search using Endee

## 📌 Project Overview
This project is an AI-powered semantic search system built using the Endee vector database.  
It allows users to ask questions and retrieves the most relevant answers from stored notes using vector similarity.

---

## 🎯 Features
- Semantic Search using embeddings
- Fast retrieval using vector database (Endee)
- Simple user query system
- Beginner-friendly AI project

---

## 🧠 How It Works
1. Input text data (notes)
2. Convert text into vectors using embedding model
3. Store vectors in Endee
4. User enters a query
5. Query is converted to vector
6. Endee searches for similar vectors
7. Best matching answer is returned

---

## ⚙️ Tech Stack
- Python
- Endee (Vector Database)
- Sentence Transformers
- Streamlit (UI)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
