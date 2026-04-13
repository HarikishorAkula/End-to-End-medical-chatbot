from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

from src.helper import download_hugging_face
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ---------------- APP ----------------
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ---------------- EMBEDDINGS ----------------
embeddings = download_hugging_face()

# ---------------- PINECONE ----------------
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medicalbot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# ---------------- LLM ----------------
llm = Ollama(model="llama3.1:8b")

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful medical assistant.

Use only the context below:
{context}

Question: {question}

Answer in simple language.
""")

# ---------------- FORMAT DOCS ----------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------- RAG CHAIN ----------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/get', methods=['POST'])
def chat():
    try:
        msg = request.form.get('msg')

        print("User:", msg)

        if not msg:
            return "Please enter a message"

        response = rag_chain.invoke(msg)

        print("Bot:", response)

        return response

    except Exception as e:
        print("Error:", e)
        return "Sorry, something went wrong."

# ---------------- RUN ----------------

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)