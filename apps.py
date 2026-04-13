import streamlit as st
import google.genai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Safety checks
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found in .env")
    st.stop()
if not PINECONE_API_KEY:
    st.error("⚠️ PINECONE_API_KEY not found in .env")
    st.stop()

# ---------------- CONFIGURE CLIENTS ----------------
# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Medical Assistant (Gemini)",
    page_icon="🏥",
    layout="centered"
)

# ---------------- BACKGROUND ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #87CEEB;
}
[data-testid="stSidebar"] {
    background-color: #f0f8ff;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🏥 Medical Assistant (Gemini)")
st.caption("Ask health-related questions based on medical knowledge.")
st.warning("⚠️ This is for educational purposes only. Always consult a doctor.")
st.divider()

# ---------------- CACHE CHAIN ----------------
@st.cache_resource(show_spinner="Loading medical knowledge base...")
def load_chain():
    embeddings = download_hugging_face()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medicalbot",
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",   # or gemini-1.5-pro
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful medical assistant.
    Use ONLY the context below.
    If answer is not found, say:
    "I don't have enough information. Please consult a doctor."

    Context:
    {context}

    Question:
    {question}

    Answer clearly in:
    - Definition
    - Causes
    - Symptoms
    - Management
    - Disclaimer
    """)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask a medical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = load_chain()
            try:
                response = chain.invoke(user_input)
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ---------------- CLEAR CHAT ----------------
if st.session_state.messages:
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()