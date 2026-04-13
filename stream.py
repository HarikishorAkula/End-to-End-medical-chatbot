import streamlit as st
from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

from src.helper import download_hugging_face
from langchain_pinecone import PineconeVectorStore

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Medical Assistant",
    page_icon="🏥",
    layout="centered"
)

# ---------------- HEADER ----------------
st.title("🏥 Medical Assistant")
st.caption("Ask health-related questions based on medical knowledge.")
st.warning("⚠️ Disclaimer: This is for general information only. Always consult a doctor.")
st.divider()

# ---------------- CACHED RESOURCES ----------------
@st.cache_resource(show_spinner="Loading medical knowledge base...")
def load_chain():
    embeddings = download_hugging_face()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medicalbot",
        embedding=embeddings
    )

    # Retrieve more chunks for better coverage
    retriever = docsearch.as_retriever(search_kwargs={"k": 5})

    llm = Ollama(model="llama3.1:8b")

    # Safer prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful and empathetic medical assistant.
    Use ONLY the retrieved context below to answer.
    Do NOT invent or add information outside the context.
    If the context does not contain the answer, say:
    "I don't have enough information on that. Please consult a doctor."

    Context:
    {context}

    Question: {question}

    Answer clearly and in simple language. Use bullet points where helpful.
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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

# ---------------- DISPLAY CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                rag_chain = load_chain()
                response = rag_chain.invoke(user_input)
            except Exception as e:
                response = "⚠️ Sorry, something went wrong. Please try again."
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ---------------- CLEAR BUTTON ----------------
if st.session_state.messages:
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()