# 🤖 PDF RAG Chatbot

An AI-powered PDF Question Answering application built with **Retrieval-Augmented Generation (RAG)**. Users can chat with a pre-built knowledge base or upload a new PDF and ask questions based on its content.

The application uses **Mistral AI** for embeddings and language generation, and **ChromaDB** for vector storage and retrieval.

## 🚀 Live Demo

Add your Streamlit deployment link here:

`https://rag-application22.streamlit.app/`

## ✨ Features

* 📚 Chat with a pre-built document knowledge base
* 📤 Upload a new PDF and chat with it
* 🔍 Semantic document search using vector embeddings
* 🧠 MMR-based retrieval for relevant and diverse document chunks
* 🤖 Mistral AI-powered answer generation
* 💬 Interactive Streamlit chat interface
* 🎨 Custom dark red and black gradient UI
* 🔐 API keys managed securely using environment variables and Streamlit Secrets
* 📄 Answers generated only from the retrieved document context

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │    User Question    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   MMR Retriever     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ChromaDB Vector   │
                    │      Database       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Relevant Document   │
                    │      Chunks         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Mistral LLM      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      AI Answer      │
                    └─────────────────────┘
```

## 📚 Two Chat Modes

### 1. Default Knowledge Base

The application loads a pre-built ChromaDB vector database:

```text
Chroma Vector Database
        ↓
Retriever
        ↓
Relevant Chunks
        ↓
Mistral AI
        ↓
Answer
```

The original PDF is not required at runtime because the document has already been converted into embeddings and stored in the vector database.

### 2. Upload New PDF

Users can upload a new PDF directly from the application:

```text
Upload PDF
    ↓
Load PDF
    ↓
Split into Chunks
    ↓
Generate Embeddings
    ↓
Create Temporary Vector Store
    ↓
Retrieve Relevant Chunks
    ↓
Generate Answer
```

## 🛠️ Technologies Used

* Python
* Streamlit
* LangChain
* Mistral AI
* Mistral Embeddings
* ChromaDB
* PyPDF
* Python-dotenv

## 📁 Project Structure

```text
RAG-Chatbot/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
└── Chroma_vector_store/
    ├── chroma.sqlite3
    └── other ChromaDB files
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG-Chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔑 API Key Configuration

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

Never upload your `.env` file to GitHub.

For Streamlit Cloud, add the API key in:

```text
App Settings → Secrets
```

Use:

```toml
MISTRAL_API_KEY = "your_mistral_api_key"
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔄 RAG Pipeline

The application follows these steps:

1. Load the document.
2. Split the document into smaller chunks.
3. Generate vector embeddings using `mistral-embed`.
4. Store embeddings in ChromaDB.
5. Retrieve relevant chunks using MMR search.
6. Pass the retrieved context to the Mistral LLM.
7. Generate an answer based on the provided context.

## 🧠 Prompt Behavior

The chatbot is instructed to answer only using the retrieved document context.

If the answer cannot be found in the provided context, the chatbot responds:

```text
Sorry, I don't know based on the provided document.
```

This helps reduce answers based on unrelated external knowledge.

## 🚀 Deployment

This application can be deployed using Streamlit Community Cloud.

### Deployment Steps

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Add `MISTRAL_API_KEY` in Streamlit Secrets.
6. Deploy the application.

## 🔒 Security

The following files should not be uploaded to GitHub:

```text
.env
.streamlit/secrets.toml
```

The API key should always be stored using environment variables or Streamlit Secrets.

## 🔮 Future Improvements

* Support multiple PDF uploads
* Add conversation memory
* Add source citations and page numbers
* Add document management
* Add support for DOCX and TXT files
* Add streaming responses
* Add authentication
* Add persistent vector database storage

## 👨‍💻 Author

**Nikhil Prajapati**

Aspiring Data Scientist | Machine Learning | Generative AI | RAG

---

⭐ If you found this project useful, consider giving the repository a star!
