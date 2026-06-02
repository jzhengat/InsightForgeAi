from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pandas as pd
import os
import streamlit as st


# LOAD CSV
def load_csv():
    return pd.read_csv("data/sales_data.csv")


# LOAD PDFs
def load_pdfs(folder="data/"):
    documents = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            documents.extend(loader.load())

    return documents


# SPLIT
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


# EMBEDDINGS + VECTOR DB
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


# BUILD RETRIEVER (CACHE IMPORTANT)
@st.cache_resource
def build_retriever():
    pdf_docs = load_pdfs()
    chunks = split_docs(pdf_docs)
    vector_db = create_vector_db(chunks)
    return vector_db.as_retriever(search_kwargs={"k": 3})


# RAG FUNCTION
def search_context(query, retriever, csv_data):

    csv_context = csv_data.describe(include="all").to_string()

    docs = retriever.invoke(query)

    pdf_context = "\n\n".join(
        [doc.page_content for doc in docs if doc.page_content]
    )

    return f"""
CSV CONTEXT:
{csv_context}

PDF CONTEXT:
{pdf_context}
"""