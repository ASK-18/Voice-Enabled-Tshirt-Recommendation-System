from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint,ChatHuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

def get_rag_chain():
    HF_TOKEN = st.secrets.get("HF_TOKEN")

    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not found in Streamlit secrets")
    # 1️⃣ Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2️⃣ Load FAISS (SAFE – created by you)
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 3️⃣ Hugging Face Hosted Qwen
    llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-LV-7B-Instruct"
    ))

    # 4️⃣ Prompt
    prompt = PromptTemplate.from_template(
        """
        You are an AI t-shirt recommendation assistant.

        Use ONLY the following product information:
        {context}

        User question:
        {question}

        Give a helpful recommendation.
        """
    )

    # 5️⃣ RAG Chain
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
