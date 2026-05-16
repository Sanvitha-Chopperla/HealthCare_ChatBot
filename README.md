🌿 Sea Buckthorn Healthcare Chatbot
📌 Project Title

Sea Buckthorn Healthcare Chatbot – AI-powered RAG Healthcare Assistant using LangChain, FAISS, OCR, Groq LLM, and Evaluation Metrics

📖 Project Description

The Sea Buckthorn Healthcare Chatbot is an AI-powered Retrieval-Augmented Generation (RAG) application designed to provide accurate, context-aware, and document-grounded responses related to Sea Buckthorn healthcare information.

The system combines:

    LangChain
    FAISS Vector Database
    Groq LLM
    OCR (Tesseract)
    Document Retrieval
    Evaluation Metrics

to create an intelligent healthcare assistant capable of answering questions from uploaded documents and predefined knowledge sources.

🚀 What the Application Does

The chatbot allows users to:

    Ask healthcare-related questions about Sea Buckthorn
    Upload multiple document formats
    Extract text from images using OCR
    Retrieve relevant chunks using vector similarity search
    Generate AI responses using Groq’s LLaMA 3.1 model
    Maintain conversation context for follow-up questions
    Display evaluation metrics for generated answers

🧠 Technologies Used
Technology	                        Purpose
Streamlit	                        Frontend UI
LangChain	                        RAG pipeline orchestration
FAISS	                            Vector similarity search
HuggingFace Embeddings	            Semantic text embeddings
Groq LLM	                        Fast AI response generation
Tesseract OCR	                    Image text extraction
Python	                            Backend development

📸 RAG Application Workflow / Architecture
System Architecture
<p align="center"> <img src="images/architecture.jpeg" width="900"/> </p>

🔄 Workflow Explanation

1️⃣ Document Upload
Users upload:
    PDF
    DOCX
    TXT
    CSV
    PPT
    JSON
    Images

2️⃣ OCR Processing
If the uploaded file is an image:
    Tesseract OCR extracts readable text

3️⃣ Text Chunking
Documents are split into smaller chunks using:
    RecursiveCharacterTextSplitter
    This improves retrieval efficiency.

4️⃣ Embedding Generation
    Chunks are converted into vector embeddings using:
    sentence-transformers/all-MiniLM-L6-v2

5️⃣ Vector Storage
    Embeddings are stored inside:
    FAISS Vector Database

6️⃣ User Query Processing
    The user asks a question.

7️⃣ Similarity Retrieval
    FAISS retrieves the most relevant chunks based on semantic similarity.

8️⃣ Prompt Construction
LangChain combines:
    Retrieved context
    Conversation history
    User query
into a structured prompt.

9️⃣ Response Generation
Groq LLaMA 3.1 generates the final response.

🔟 Evaluation
The system evaluates generated responses using:
    Relevance
    Length Score
    Context Usage

⚙️ How to Install and Run the Project

1️⃣ Clone the Repository
//Terminal
git clone https://github.com/Sanvitha-Chopperla/HealthCare_ChatBot.git
cd HealthCare_ChatBot

2️⃣ Install Required Libraries
pip install -r requirements.txt

3️⃣ Install Tesseract OCR
Download and install:
    [Tesseract OCR GitHub](https://github.com/tesseract-ocr/tesseract)
After installation, set the path inside app.py:
    pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

4️⃣ Configure Environment Variables
Create a .env file:
    GROQ_API_KEY=your_api_key_here

5️⃣ Run the Application
streamlit run app.py

🧠 How to Use the Project

💬 Ask Questions
Examples:
    What are the benefits of Sea Buckthorn?
    Who should avoid using Sea Buckthorn?
    Explain Sea Buckthorn nutrition briefly.

📂 Upload Documents
1.Upload documents
2.Click Load Documents
3.Ask questions
Relavant Answer will be Generated and Evauluation also takes place