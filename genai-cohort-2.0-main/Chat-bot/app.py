import os
from dotenv import load_dotenv
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# ─────────────────────────────
# 1. Load OpenAI Key from .env
# ─────────────────────────────
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
url = "https://docs.chaicode.com/youtube/getting-started/"  # Replace with any target website

if not openai_key:
    st.error("❌ OPENAI_API_KEY not found in .env file!")
    st.stop()

# ─────────────────────────────
# 2. Scrape Website Content
# ─────────────────────────────
@st.cache_data
def scrape_website(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content
    except Exception as e:
        st.error(f"❌ Scraping failed: {e}")
        return ""

# ─────────────────────────────
# 3. Embed Text to FAISS Vector DB
# ─────────────────────────────
@st.cache_resource
def embed_text(text):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([text])
        embeddings = OpenAIEmbeddings(api_key=openai_key)
        db = FAISS.from_documents(docs, embeddings)
        return db
    except Exception as e:
        st.error(f"❌ Embedding failed: {e}")
        return None

# ─────────────────────────────
# 4. Streamlit Chatbot UI
# ─────────────────────────────
st.set_page_config(page_title="🧠 Website Chatbot", layout="centered")
st.title("💬 Chat with the Website 📄")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hi! 👋 Ask me anything about {url}"}
    ]

# Show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input field
query = st.chat_input("Ask your question...")

if query:
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🧠 Thinking..."):
        site_text = scrape_website(url)
        vector_db = embed_text(site_text)
        if vector_db:
            llm = ChatOpenAI(temperature=0.2, api_key=openai_key)
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())
            try:
                answer = qa.run(query)
            except Exception as e:
                answer = f"❌ GPT Error: {str(e)}"
        else:
            answer = "Vector DB creation failed."

    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
