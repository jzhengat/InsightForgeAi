from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

import pandas as pd
import os

# LOAD CSV

def load_csv():
    return pd.read_csv("data/sales_data.csv")

# LOAD PDF
def load_pdfs(folder="data/"):
    documents = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            documents.extend(loader.load())

    return documents

# SPLIT DOCUMENTS
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

# CREATE VECTOR DATABASE (FAISS)
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# BUILD RETRIEVER
def build_retriever():
    pdf_docs = load_pdfs()
    chunks = split_docs(pdf_docs)

    vector_db = create_vector_db(chunks)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    return retriever

# RAG SEARCH FUNCTION
def search_context(query, retriever, csv_data):
    # -----------------------
    # CSV CONTEXT
    # -----------------------
    csv_context = csv_data.describe(include="all").to_string()

    # -----------------------
    # PDF CONTEXT (SEMANTIC SEARCH)
    # -----------------------
    docs = retriever.get_relevant_documents(query)

    pdf_context = "\n\n".join([doc.page_content for doc in docs])

    # -----------------------
    # FINAL COMBINED CONTEXT
    # -----------------------
    return f"""
CSV CONTEXT:
{csv_context}

PDF CONTEXT:
{pdf_context}
"""