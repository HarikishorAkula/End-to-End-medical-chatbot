import streamlit as st
import google.genai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face
from pinecone import Pinecone
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found in .env")
    st.stop()
if not PINECONE_API_KEY:
    st.error("⚠️ PINECONE_API_KEY not found in .env")
    st.stop()

# ---------------- CONFIGURE CLIENTS ----------------
client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MedBot AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

/* ---- Reset & Base ---- */
* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Nunito', sans-serif !important;
    background: #0a1628 !important;
}

/* ---- WhatsApp-style wallpaper background ---- */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(circle at 15% 85%, rgba(0,168,132,0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 15%, rgba(37,211,102,0.08) 0%, transparent 40%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%2325d366' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    background-color: #0d1b2a;
    z-index: 0;
    pointer-events: none;
}

/* ---- Hide Streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding: 0 !important;
    max-width: 780px !important;
    margin: 0 auto !important;
    position: relative;
    z-index: 1;
}

/* ---- Top header bar (WhatsApp green) ---- */
.chat-header {
    background: linear-gradient(135deg, #075e54 0%, #128c7e 100%);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    border-radius: 0 0 6px 6px;
}
.avatar-ring {
    width: 46px; height: 46px;
    border-radius: 50%;
    background: linear-gradient(135deg, #25d366, #128c7e);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 0 3px rgba(37,211,102,0.3);
    flex-shrink: 0;
}
.header-info .name {
    font-weight: 800; font-size: 16px; color: #fff; letter-spacing: 0.3px;
}
.header-info .status {
    font-size: 12px; color: #a8f0c6; font-weight: 600; display: flex; align-items: center; gap: 5px;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #25d366;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}
.header-badge {
    margin-left: auto;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    color: #d4fce8;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ---- Chat area ---- */
.chat-area {
    min-height: 60vh;
    padding: 16px 14px 8px 14px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}

/* ---- Date chip ---- */
.date-chip {
    text-align: center; margin: 10px 0;
}
.date-chip span {
    background: rgba(37,211,102,0.15);
    border: 1px solid rgba(37,211,102,0.25);
    color: #a8f0c6;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 14px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}

/* ---- Message bubbles ---- */
.msg-row {
    display: flex;
    margin-bottom: 6px;
    animation: msgFadeIn 0.3s ease-out;
}
@keyframes msgFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user  { justify-content: flex-end; }
.msg-row.bot   { justify-content: flex-start; }

/* User bubble — teal-green like WhatsApp sent */
.bubble.user {
    background: linear-gradient(135deg, #005c4b 0%, #075e54 100%);
    color: #e8fff6;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    padding: 10px 14px 7px 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    position: relative;
}
/* Bot bubble — dark card like WhatsApp received */
.bubble.bot {
    background: linear-gradient(135deg, #1a2a3a 0%, #1e3040 100%);
    color: #e2f0fb;
    border-radius: 18px 18px 18px 4px;
    max-width: 78%;
    padding: 10px 14px 7px 14px;
    border: 1px solid rgba(37,211,102,0.12);
    box-shadow: 0 2px 10px rgba(0,0,0,0.4);
}

.bubble .msg-text {
    font-size: 14.5px;
    line-height: 1.6;
    font-weight: 500;
    white-space: pre-wrap;
    word-break: break-word;
}
.bubble .msg-text strong { color: #25d366; font-weight: 800; }
.bubble .msg-text em { color: #a8f0c6; }

.bubble .meta {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 4px;
    margin-top: 4px;
}
.bubble .time {
    font-size: 10.5px;
    color: rgba(255,255,255,0.45);
    font-weight: 600;
}
.bubble.user .ticks { color: #53bdeb; font-size: 13px; }

/* Bot sender label */
.bot-label {
    font-size: 11.5px;
    font-weight: 800;
    color: #25d366;
    margin-bottom: 3px;
    letter-spacing: 0.3px;
}

/* ---- Typing indicator ---- */
.typing-indicator {
    display: flex; align-items: center; gap: 5px;
    padding: 10px 16px;
    background: linear-gradient(135deg, #1a2a3a, #1e3040);
    border: 1px solid rgba(37,211,102,0.12);
    border-radius: 18px 18px 18px 4px;
    width: fit-content;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    animation: msgFadeIn 0.3s ease-out;
}
.typing-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #25d366;
    animation: typingBounce 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingBounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.5; }
    40%           { transform: translateY(-7px); opacity: 1; }
}

/* ---- Disclaimer box ---- */
.disclaimer {
    margin: 10px 14px 6px 14px;
    background: rgba(255,193,7,0.08);
    border: 1px solid rgba(255,193,7,0.25);
    border-radius: 10px;
    padding: 8px 14px;
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; color: #ffd060; font-weight: 600;
}

/* ---- Input area ---- */
.input-bar {
    position: sticky; bottom: 0; z-index: 100;
    background: #0d1b2a;
    padding: 10px 12px 14px 12px;
    border-top: 1px solid rgba(37,211,102,0.1);
}

/* Override Streamlit chat input */
[data-testid="stChatInput"] {
    background: transparent !important;
}
[data-testid="stChatInputTextArea"] {
    background: #1a2e40 !important;
    border: 1.5px solid rgba(37,211,102,0.25) !important;
    border-radius: 26px !important;
    color: #e2f0fb !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 14.5px !important;
    padding: 12px 18px !important;
    transition: border-color 0.2s;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color: rgba(37,211,102,0.6) !important;
    box-shadow: 0 0 0 3px rgba(37,211,102,0.08) !important;
    outline: none !important;
}
[data-testid="stChatInputTextArea"]::placeholder { color: rgba(255,255,255,0.3) !important; }

/* Send button */
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #25d366, #128c7e) !important;
    border-radius: 50% !important;
    width: 44px !important; height: 44px !important;
    border: none !important;
    box-shadow: 0 3px 10px rgba(37,211,102,0.35) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 5px 16px rgba(37,211,102,0.5) !important;
}

/* ---- Clear button ---- */
.stButton button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: rgba(255,255,255,0.5) !important;
    border-radius: 20px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    padding: 4px 16px !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    border-color: rgba(255,87,87,0.4) !important;
    color: #ff7070 !important;
    background: rgba(255,87,87,0.06) !important;
}

/* ---- Welcome card ---- */
.welcome-card {
    margin: 30px 14px 10px;
    background: linear-gradient(135deg, rgba(7,94,84,0.35), rgba(18,140,126,0.15));
    border: 1px solid rgba(37,211,102,0.2);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    animation: msgFadeIn 0.5s ease-out;
}
.welcome-card .wc-icon { font-size: 48px; margin-bottom: 10px; }
.welcome-card h3 { color: #25d366; font-size: 20px; font-weight: 800; margin-bottom: 6px; }
.welcome-card p  { color: rgba(255,255,255,0.55); font-size: 13.5px; line-height: 1.6; }
.suggestion-chips {
    display: flex; flex-wrap: wrap; gap: 8px;
    justify-content: center; margin-top: 16px;
}
.chip {
    background: rgba(37,211,102,0.12);
    border: 1px solid rgba(37,211,102,0.25);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px; color: #a8f0c6; font-weight: 700;
    cursor: pointer; transition: all 0.2s;
}
.chip:hover { background: rgba(37,211,102,0.22); }

/* Hide default streamlit elements */
[data-testid="stSpinner"] > div { color: #25d366 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="chat-header">
    <div class="avatar-ring">🩺</div>
    <div class="header-info">
        <div class="name">MedBot AI</div>
        <div class="status">
            <span class="status-dot"></span>
            Online · Powered by Gemini
        </div>
    </div>
    <div class="header-badge">🔒 Private</div>
</div>
""", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class="disclaimer">
    ⚕️ For educational use only. Always consult a qualified doctor for medical advice.
</div>
""", unsafe_allow_html=True)

# ---------------- CACHE CHAIN ----------------
@st.cache_resource(show_spinner="🔄 Connecting to medical knowledge base...")
def load_chain():
    embeddings = download_hugging_face()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medicalbot",
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template("""
You are MedBot, a friendly and knowledgeable medical assistant.
Use ONLY the context below to answer. Be clear, structured and compassionate.
If answer is not found, say: "I don't have enough information on this. Please consult a doctor."

Context:
{context}

Question:
{question}

Format your response with these sections (use bold headers with **):
**📋 Definition**
**🔍 Causes**
**🤒 Symptoms**
**💊 Management**
**⚠️ Disclaimer**
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

# ---------------- HELPER ----------------
def fmt_time():
    return datetime.now().strftime("%I:%M %p")

def render_bubble(role, content, ts):
    if role == "user":
        st.markdown(f"""
<div class="msg-row user">
  <div class="bubble user">
    <div class="msg-text">{content}</div>
    <div class="meta">
      <span class="time">{ts}</span>
      <span class="ticks">✓✓</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        # Convert markdown bold **text** to <strong> and newlines
        import re
        html_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
        html_content = html_content.replace("\n", "<br>")
        st.markdown(f"""
<div class="msg-row bot">
  <div class="bubble bot">
    <div class="bot-label">🩺 MedBot AI</div>
    <div class="msg-text">{html_content}</div>
    <div class="meta">
      <span class="time">{ts}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ---------------- CHAT AREA ----------------
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
<div class="welcome-card">
    <div class="wc-icon">🏥</div>
    <h3>Hello! I'm MedBot AI</h3>
    <p>Ask me anything about symptoms, conditions, medications or general health queries. I'll do my best to help using trusted medical knowledge.</p>
    <div class="suggestion-chips">
        <span class="chip">💊 What is diabetes?</span>
        <span class="chip">🤒 Fever symptoms</span>
        <span class="chip">❤️ Heart disease causes</span>
        <span class="chip">🧠 Migraine treatment</span>
    </div>
</div>
""", unsafe_allow_html=True)
else:
    # Date chip
    st.markdown(f"""
<div class="date-chip">
  <span>Today · {datetime.now().strftime("%B %d, %Y")}</span>
</div>
""", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_bubble(msg["role"], msg["content"], msg.get("time", fmt_time()))

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.chat_input("Type a medical question…")

if user_input:
    ts = fmt_time()
    st.session_state.messages.append({"role": "user", "content": user_input, "time": ts})

    # Re-render all messages including new user one
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    st.markdown(f"""
<div class="date-chip">
  <span>Today · {datetime.now().strftime("%B %d, %Y")}</span>
</div>
""", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_bubble(msg["role"], msg["content"], msg.get("time", ts))

    # Typing indicator
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
<div class="msg-row bot">
  <div class="typing-indicator">
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Get response
    chain = load_chain()
    try:
        response = chain.invoke(user_input)
    except Exception as e:
        response = f"⚠️ Error: {str(e)}"

    # Remove typing indicator
    typing_placeholder.empty()

    bot_ts = fmt_time()
    st.session_state.messages.append({"role": "assistant", "content": response, "time": bot_ts})
    st.rerun()

# ---------------- CLEAR ----------------
if st.session_state.messages:
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.messages = []
            st.rerun()