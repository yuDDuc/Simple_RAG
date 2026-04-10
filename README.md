# 📚 RAG v1 — Document Q&A with LangChain + Gemini

A simple **Retrieval-Augmented Generation (RAG)** system that lets you ask questions about your PDF papers using Google Gemini for both embeddings and LLM inference.

---

## 🗂️ Project Structure

```
RAG/v1/
├── papers/          # Drop your PDF files here
├── faiss_index/     # Auto-generated FAISS vector store (created on first run)
├── .venv/           # Python virtual environment
├── .env             # Your API key
├── chatbot.py       # Main RAG script
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## ⚙️ Tech Stack

| Component        | Library / Model                    |
|------------------|------------------------------------|
| LLM              | `gemini-2.5-flash` (Google Gemini) |
| Embeddings       | `gemini-embedding-001`             |
| Vector Store     | FAISS (local, CPU)                 |
| PDF Loader       | `PyPDFLoader` (via LangChain)      |
| Text Splitter    | `RecursiveCharacterTextSplitter`   |
| Chain Framework  | LangChain + `langchain_classic`    |

---

## 🚀 Setup

### 1. Create & activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in this directory:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

> Get your key at: https://aistudio.google.com/apikey

### 4. Add PDF papers

Place your `.pdf` files inside the `papers/` folder.

---

## ▶️ Run

```powershell
python chatbot.py
```

---

## 🔄 How It Works (Workflow)

```
First run:
  papers/ (PDFs)
      ↓ PyPDFLoader
  Raw text pages
      ↓ RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
  Text chunks
      ↓ GoogleGenerativeAIEmbeddings (gemini-embedding-001)
  Vectors
      ↓ FAISS.from_documents()
  faiss_index/  ← saved to disk

Subsequent runs:
  faiss_index/
      ↓ FAISS.load_local()
  Vector store (skips PDF parsing & re-embedding)

At query time:
  User question
      ↓ FAISS retriever (top-5 chunks)
  Relevant context
      ↓ ChatPromptTemplate + gemini-2.5-flash
  Answer
```

---

## 💡 Key Design Decisions

### FAISS Index Caching
The vector store is **built once** and saved to `faiss_index/`. On subsequent runs, it loads from disk — skipping the slow PDF parsing and embedding API calls.

> **To rebuild the index** (e.g. after adding new papers), delete the `faiss_index/` folder and rerun:
> ```powershell
> Remove-Item -Recurse -Force faiss_index
> python chatbot.py
> ```

### LangChain 1.2+ Compatibility
LangChain 1.2 moved `create_retrieval_chain` and `create_stuff_documents_chain` out of `langchain.chains` into a separate `langchain_classic` package. This project uses the correct import paths:

```python
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
```

### VS Code Python Interpreter
If Pylance shows import errors, select the `.venv` interpreter:
- `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`

---

## 🛠️ Tuning Parameters

| Parameter      | Location in `chatbot.py`      | Default | Effect                                      |
|----------------|-------------------------------|---------|---------------------------------------------|
| `chunk_size`   | `RecursiveCharacterTextSplitter` | 1000 | Larger = more context per chunk             |
| `chunk_overlap`| `RecursiveCharacterTextSplitter` | 200  | Higher = less info loss at chunk boundaries |
| `k`            | `as_retriever(search_kwargs)` | 5       | Number of chunks retrieved per query        |
| `temperature`  | `ChatGoogleGenerativeAI`      | 0.3     | Lower = more factual, higher = more creative|

---

## ⚠️ Known Issues

- **Python 3.14 warning**: `pydantic.v1` is not fully compatible with Python 3.14. This is a LangChain upstream issue — the warning is safe to ignore, functionality is unaffected.
- **Model not found (404)**: Ensure you use `gemini-embedding-001` for embeddings (not `text-embedding-004` or `models/embedding-001`), as model availability depends on API version.
