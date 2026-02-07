import faiss
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

index = faiss.read_index("index.faiss")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query, k=3):
    q_emb = embedding_model.encode([query])
    distances, indices = index.search(np.array(q_emb), k)
    return [chunks[i] for i in indices[0]]

def build_prompt(context, question):
    return f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{question}
"""

def ask_groq(prompt):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    query = "Explain LRU page replacement"

    retrieved = retrieve(query, k=3)
    context = "\n\n".join(retrieved)

    print("\n Retrieved Context:\n")
    for r in retrieved:
        print("-----")
        print(r[:300])

    prompt = build_prompt(context, query)

    print("\n LLM Answer:\n")
    answer = ask_groq(prompt)
    print(answer)
