# Searchable-PDF-QA

A professional PDF Question-Answering project that allows you to **search and query PDF documents** using AI embeddings. 
Upload PDFs, generate embeddings, and ask questions to get accurate answers.

## Goal of the Project

To demonstrate the ability to:
-Convert PDFs into structured searchable text
-Build embeddings using Sentence Transformers
-Use FAISS for vector similarity search
-Implement retrieval-based Q&A with LangChain
-Build a user-friendly Streamlit interface
-Create a complete, end-to-end ML/NLP pipeline


## Features
- Upload PDF files and extract text.
- Generate embeddings for semantic search.
- Ask questions and get accurate answers from PDFs.
- Supports multiple PDFs at once.
- Works with multiple PDF files in one session.


## Technologies Used
- Python 
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- PyPDF
  
I used Python 3.10.11 because LangChain and several NLP libraries provide the best stability and compatibility with this version. Higher Python versions may cause installation or runtime errors, so 3.10.11 is the recommended environment for LangChain-based projects.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/asthaashrma-del/PDFIntelliSearch.git

2-Navigate to the folder:
cd Searchable-PDF-QA

3-Create a virtual environment and activate it:
python -m venv venv
venv\Scripts\activate   # Windows

4-Install dependencies:
pip install -r requirements.txt

5-Run the app:
streamlit run app.py

## Folder Structure

Searchable-PDF-QA/

├── data
├── app.py
├── main.py
├── python
├── README.md
├── requirements.txt

##  How It Works
-PDF Upload → Extracts text
-Chunking → Splits text into meaningful segments
-Embedding Generation → Converts chunks into vector representations
-FAISS Indexing → Enables fast semantic search
-Question Answering → Retrieves best text & forms the answer


## Use Cases
-Research paper search
-Legal document question answering
-Study material Q&A
-Corporate document indexing
-Resume search
-Policy or guideline lookup


