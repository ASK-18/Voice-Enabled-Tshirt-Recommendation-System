import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading product data...")
df = pd.read_csv("data/products.csv")

texts = df.apply(
    lambda x: f"""
    {x['title']}.
    Color: {x['color']}.
    Fit: {x['fit']}.
    Style: {x['style']}.
    Price: {x['price']}.
    """,
    axis=1
).tolist()

metadatas = df.to_dict(orient="records")

print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building FAISS index...")
db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
db.save_local("faiss_index")

print("✅ FAISS index created successfully")
