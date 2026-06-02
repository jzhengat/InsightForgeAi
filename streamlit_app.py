import streamlit as st
from rag import load_csv, build_retriever, search_context
from llm import get_answer

st.title("InsightForge - AI BI Assistant")

# Load data and build retriever
csv_data = load_csv()
retriever = build_retriever()

# User query
question = st.text_input("Ask a question:")

if question:
    # Get relevant CSV + PDF context
    context = search_context(question, retriever, csv_data)
    
    # )LLM response
    answer = get_answer(context, question)

    st.subheader("AI Insight")
    st.write(answer)

# Optional dataset view
if st.checkbox("Show Sales Data"):
    st.write(csv_data.head())