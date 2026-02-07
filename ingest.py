from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer


def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()+"\n"
    return text

#raw_text = load_pdf("data/module.pdf")
#print(raw_text[:100])

def chunk_text(text , chunk_size = 500, overlap= 100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings, model
def store_faiss(embeddings, chunks):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, "index.faiss")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    text = load_pdf("data/module.pdf")
    chunks = chunk_text(text)
    embeddings, _ = embed_chunks(chunks)
    store_faiss(embeddings, chunks)

    print("Ingestion complete")
