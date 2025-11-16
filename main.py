from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

PDF_PATH = "data/2506.02153v2.pdf"

# 1. Load PDF text
reader = PdfReader(PDF_PATH)
full_text = ""
for page in reader.pages:
    txt = page.extract_text()
    if txt:
        full_text += txt

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_text(full_text)

print("Total chunks:", len(chunks))

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Convert chunks → embeddings
embeddings = model.encode(chunks, convert_to_numpy=True)

# 5. Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index built.")
print("Total vectors inside FAISS:", index.ntotal)

# 6. Search function
def search_pdf(query):
    # Convert query into embedding
    q_embed = model.encode([query], convert_to_numpy=True)

    # Search top 3 chunks
    k = 3
    distances, indices = index.search(q_embed, k)

    print("\nTop Results:")
    for i, idx in enumerate(indices[0]):
        print(f"\nResult {i+1}:")
        print(chunks[idx][:500], "...")
        

# 7. Ask user for questions
while True:
    q = input("\nAsk something about your PDF (type 'exit' to quit): ")
    if q.lower() == "exit":
        break

    search_pdf(q)

