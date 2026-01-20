import streamlit as st
import os
import tempfile
from PIL import Image

from audiorecorder import audiorecorder
from speech import speech_to_text
from rag_chain import get_rag_chain

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- Page Config ---------------- #
st.set_page_config(page_title="AI T-Shirt Recommendation System", layout="wide")
st.title("👕 AI T-Shirt Recommendation System")

st.markdown(
    """
    🎤 **Click the microphone to speak**  
    ⌨️ Or type your query  

    **Examples**
    - black oversized graphic t-shirt  
    - affordable white t-shirts  
    """
)

# ---------------- Load RAG ---------------- #
@st.cache_resource
def load_rag():
    return get_rag_chain()

rag_chain = load_rag()

# ---------------- Load Retriever ---------------- #
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db.as_retriever(search_kwargs={"k": 5})

retriever = load_retriever()

# ---------------- INPUT SECTION ---------------- #
st.subheader("🎤 Speak your requirement")

audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording")

query = None

# If audio is recorded
if len(audio) > 0:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.export(tmp.name, format="wav")
        audio_path = tmp.name

    with st.spinner("Transcribing your voice..."):
        query = speech_to_text(audio_path)

    st.success("You said:")
    st.write(query)

# Text fallback
st.subheader("⌨️ Or type your requirement")
text_query = st.text_input("Type here")

if text_query.strip():
    query = text_query

# ---------------- SEARCH ---------------- #
if st.button("Get Recommendations"):
    if not query:
        st.warning("Please speak using microphone or type your query.")
    else:
        with st.spinner("Finding best matches..."):
            docs = retriever.invoke(query)
            answer = rag_chain.invoke(query)

        # AI Response
        st.subheader("🧠 AI Recommendation")
        st.write(answer)

        # Images
        st.subheader("🖼️ Best Matching T-Shirts")

        cols = st.columns(3)
        for i, doc in enumerate(docs):
            img_path = os.path.join("data/images", doc.metadata["image"])

            with cols[i % 3]:
                if os.path.exists(img_path):
                    st.image(Image.open(img_path), use_column_width=True)
                else:
                    st.warning("Image not found")

                st.markdown(f"**{doc.metadata['title']}**")
                st.markdown(f"Color: {doc.metadata['color']}")
                st.markdown(f"Style: {doc.metadata['style']}")
                st.markdown(f"Price: ₹{doc.metadata['price']}")
