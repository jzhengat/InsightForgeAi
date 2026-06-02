import pandas as pd
import os
from pypdf import PdfReader

# Load CSV
def load_csv():
    return pd.read_csv("data/sales_data.csv")


# Load ALL PDFs in data folder
def load_all_pdfs():
    folder = "data/"
    all_text = ""

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            reader = PdfReader(path)

            for page in reader.pages:
                all_text += page.extract_text() or ""

    return all_text