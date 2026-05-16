# 🌿 Sea Buckthorn Healthcare Chatbot

> AI-powered RAG Healthcare Assistant using LangChain, FAISS, OCR, Groq LLM, and Evaluation Metrics

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG_Pipeline-1C3C3C?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F54036?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📖 About the Project

The **Sea Buckthorn Healthcare Chatbot** is an AI-powered Retrieval-Augmented Generation (RAG) application designed to provide accurate, context-aware, and document-grounded responses related to Sea Buckthorn healthcare information.

The system combines **LangChain**, **FAISS**, **Groq LLM**, **Tesseract OCR**, and **Evaluation Metrics** to build an intelligent assistant capable of answering questions from uploaded documents and predefined knowledge sources.

---

## 🚀 Features

- 💬 Ask healthcare-related questions about Sea Buckthorn
- 📂 Upload and process multiple document formats
- 🖼️ Extract text from images using OCR
- 🔍 Retrieve relevant content using vector similarity search
- 🤖 Generate AI responses with Groq's LLaMA 3.1 model
- 🧵 Maintain conversation context for follow-up questions
- 📊 View automatic evaluation metrics for every response

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Frontend UI |
| [LangChain](https://www.langchain.com/) | RAG pipeline orchestration |
| [FAISS](https://faiss.ai/) | Vector similarity search |
| [HuggingFace Embeddings](https://huggingface.co/) | Semantic text embeddings (`all-MiniLM-L6-v2`) |
| [Groq LLM](https://groq.com/) | Fast AI response generation (LLaMA 3.1) |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Image text extraction |
| Python | Backend development |

---

## 📸 Architecture

<p align="center">
  <img src="images/architecture.jpeg" width="860" alt="RAG Application Architecture"/>
</p>

---

## 🔄 Workflow

| Step | Stage | Description |
|---|---|---|
| 1️⃣ | **Document Upload** | Users upload `PDF`, `DOCX`, `TXT`, `CSV`, `PPT`, `JSON`, or image files |
| 2️⃣ | **OCR Processing** | Tesseract OCR extracts text from image-based uploads |
| 3️⃣ | **Text Chunking** | Documents are split using `RecursiveCharacterTextSplitter` for efficient retrieval |
| 4️⃣ | **Embedding Generation** | Chunks are encoded as dense vectors using `sentence-transformers/all-MiniLM-L6-v2` |
| 5️⃣ | **Vector Storage** | Embeddings are indexed inside a **FAISS** vector database |
| 6️⃣ | **Query Processing** | The user submits a natural language question |
| 7️⃣ | **Similarity Retrieval** | FAISS retrieves the most semantically relevant chunks |
| 8️⃣ | **Prompt Construction** | LangChain combines retrieved context + conversation history + user query into a structured prompt |
| 9️⃣ | **Response Generation** | Groq LLaMA 3.1 produces the final grounded answer |
| 🔟 | **Evaluation** | Responses are scored on **Relevance**, **Length Score**, and **Context Usage** |

---

## ⚙️ Installation & Setup

### 1 — Clone the repository

```bash
git clone https://github.com/Sanvitha-Chopperla/HealthCare_ChatBot.git
cd HealthCare_ChatBot
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Install Tesseract OCR

Download and install from the [Tesseract OCR GitHub](https://github.com/tesseract-ocr/tesseract), then set the path inside `app.py`:

```python
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

> **Linux / macOS**: Tesseract can be installed via your package manager (`apt`, `brew`), and the path can usually be omitted or detected automatically.

### 4 — Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

### 5 — Run the application

```bash
streamlit run app.py
```

---

## 🧠 How to Use

1. **Upload** one or more documents using the file uploader in the sidebar.
2. Click **Load Documents** to index and embed the content.
3. **Ask a question** in the chat input — a grounded answer and evaluation metrics will be generated automatically.

### Example questions

```
What are the benefits of Sea Buckthorn?
Who should avoid using Sea Buckthorn?
Explain Sea Buckthorn nutrition briefly.
```

---

## 📊 Evaluation Metrics

Each generated response is automatically evaluated across three dimensions:

| Metric | Description |
|---|---|
| **Relevance** | How closely the answer relates to the user's query |
| **Length Score** | Whether the response length is appropriate |
| **Context Usage** | How well the retrieved chunks were used in the answer |

---

## 📁 Supported File Formats

`PDF` · `DOCX` · `TXT` · `CSV` · `PPT` · `JSON` · `PNG / JPG / JPEG (OCR)`


---

## 👤 Author

**Sanvitha Chopperla**  
GitHub: [@Sanvitha-Chopperla](https://github.com/Sanvitha-Chopperla)