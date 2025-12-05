📄 PDF RAG Chat — DeepSeek Cloud + Local Retrieval
Smart Document Question Answering with Automatic Indexing, Per-PDF Chat Sessions & Evidence Tracing

🔗 Live App: https://rag-tool-nbtp9no3nmsye2vkadafak.streamlit.app/

🚀 Overview

PDF RAG Chat is an AI-powered document assistant that allows users to upload PDFs and chat with them intelligently.
The system automatically:

Extracts text from PDFs

Chunks & indexes content using ChromaDB

Retrieves relevant context using vector search

Generates accurate answers using DeepSeek Cloud via Ollama API

Maintains chat history per PDF session just like ChatGPT or Claude

Displays evidence snippets for transparency

This application is built to behave like a real production-grade RAG system.

✨ Key Features
🔍 Automatic Indexing

No need to manually build vectors. When you upload a PDF, the system automatically:

Extracts text (PyMuPDF)

Chunks into sliding windows

Embeds using BGE small

Stores in ChromaDB (persistent)

💬 Multiple Chat Sessions

Each uploaded PDF becomes its own dedicated chat session:

Sessions stored in SQLite

Switching sessions instantly loads the correct PDF index + history

No confusion between documents

🧠 DeepSeek Cloud LLM Integration

The actual answer generation is done through:

DeepSeek V3.1 671B Cloud model

Accessed via Ollama Cloud API

Prompts use retrieved context only → preventing hallucinations

📚 Evidence-Based Answers

Every response includes:

Top-k retrieved chunks

Expandable evidence sections

Full transparency into how the LLM answered

🧼 Clean Streamlit UI

Left sidebar = Chat sessions

Main panel = Conversation

Works on desktop & mobile

Smooth UX similar to modern LLM apps

🏗️ Tech Stack
Backend / Core

Python 3.10+

Streamlit

SQLite (session storage)

ChromaDB (vector store)

PyMuPDF (PDF extraction)

HuggingFace embeddings (BGE-small)

Ollama Cloud Python Client (DeepSeek model)

LLM

DeepSeek V3.1 671B Cloud via OLLAMA_API_KEY

Decoupled retrieval + synthesis pipeline



🔑 Environment Variables

Inside Streamlit Secrets:

OLLAMA_API_KEY = "your_ollama_cloud_key"
OLLAMA_MODEL = "deepseek-v3.1:671b-cloud"

📦 Installation (Local)

Clone the repository:

git clone https://github.com/yourname/pdf-rag-chat
cd pdf-rag-chat


Create a virtual environment:

python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py
