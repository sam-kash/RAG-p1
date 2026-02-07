import faiss
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------- SETUP ----------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

index = faiss.read_index("index.faiss")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------- RETRIEVAL ----------
def retrieve(query, k=3):
    q_emb = embedding_model.encode([query])
    distances, indices = index.search(np.array(q_emb), k)
    return [chunks[i] for i in indices[0]]


# ---------- PROMPT ----------
def build_prompt(context, question):
    return f"""
You are a tutor answering strictly from the given context.
Use simple language.
If the answer is not present, say "Not found in the document".

Context:
{context}

Question:
{question}
"""


# ---------- LLM ----------
def ask_groq(prompt):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return completion.choices[0].message.content


# ---------- CHAT LOOP ----------
if __name__ == "__main__":
    print("\n📘 PDF Chat (type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Bot: Bye 👋")
            break

        retrieved = retrieve(query, k=3)
        context = "\n\n".join(retrieved)

        prompt = build_prompt(context, query)
        answer = ask_groq(prompt)

        print("\nBot:", answer, "\n")
