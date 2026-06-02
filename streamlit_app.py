import streamlit as st
from rag import build_vector_db, get_context
from llm import get_answer

st.title("InsightForge - AI BI Assistant")

# Build FAISS once
db = build_vector_db()

# User query
question = st.text_input("Ask a business question:")

if question:
    context = get_context(question, db)
    answer = get_answer(context, question)

    st.subheader("🧠 AI Insight")
    st.write(answer)


# Optional dataset view
if st.checkbox("Show Sales Data"):
    import pandas as pd
    df = pd.read_csv("data/sales_data.csv")
    st.write(df.head())

# run streamlit run streamlit_app.py