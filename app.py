
# ── Imports ───────────────────────────────────────────────────────────────────
import os
import json
import tempfile

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pytesseract

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
st.set_page_config(page_title="Sea Buckthorn Healthcare Chatbot", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_EXTS    = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
CHUNK_SIZE    = 3000
CHUNK_OVERLAP = 300
RETRIEVAL_K   = 6
MEMORY_TURNS  = 10
DATA_FOLDER   = "data"

CASUAL_KEYWORDS = [
    "hi", "hello", "hey", "hiya", "howdy",
    "how are you", "how r u", "how are u",
    "how was your day", "how is your day",
    "good morning", "good evening", "good afternoon", "good night",
    "what's up", "whats up", "sup", "who are you", "what are you",
    "thank you", "thanks", "thank u", "thx",
    "bye", "goodbye", "see you", "see ya",
    "ok", "okay", "cool", "great", "nice", "awesome",
    "well done", "good job", "nice work",
]

OFFTOPIC_KEYWORDS = [
    "cricket", "football", "soccer", "ipl", "nba", "sports",
    "movie", "film", "series", "netflix", "song", "music", "album",
    "weather", "rain", "temperature", "stock", "share market", "bitcoin", "crypto",
    "politics", "election", "government", "war",
    "recipe", "cooking", "restaurant", "travel", "tour", "vacation", "trip",
    "joke", "funny", "meme", "girlfriend", "boyfriend", "love", "marriage",
    "programming", "code", "software", "developer",
]

UPLOADED_DOC_KEYWORDS = [
    "my uploaded document", "my document", "my pdf", "my file", "my book",
    "my image", "my photo", "my picture", "my screenshot",
    "uploaded document", "uploaded file", "uploaded pdf", "uploaded book",
    "uploaded image", "uploaded photo", "uploaded picture",
    "this document", "this file", "this pdf", "this book",
    "this image", "this photo", "this picture",
    "what i uploaded", "the document i uploaded",
    "summarize my", "summary of my",
    "tell me about my document", "tell me about my file", "tell me about my pdf",
    "tell me about my image", "tell me about my photo", "tell me about my picture",
    "what is in my", "overview of my", "shortly about my", "brief about my",
    "from my document", "from my file", "from my pdf",
    "in my document", "in my file", "in my pdf", "about my uploaded",
    "what is there in", "describe the image", "describe the picture",
    "describe my image", "describe my picture", "describe my photo",
    "read the image", "read the picture", "what does the image say",
    "text in the image", "text in the picture",
    "what is shown", "tell me about the image", "tell me about the picture",
]

FOLLOWUP_PHRASES = [
    "more specific", "be specific", "be more specific",
    "elaborate", "tell me more", "more detail", "more details",
    "expand", "explain more", "go deeper", "give example", "give examples",
    "more about it", "more about that", "continue", "what else",
    "specifically", "in detail", "say more", "and also",
    "in simple", "make it simple", "explain simply",
    "tell me briefly", "tell me in brief", "tell me in short",
    "tell me shortly", "give me brief", "give me short",
]

BRAND_KEYWORDS = [
    "brand", "brands", "name the brand", "brand name",
    "which company", "which brand", "who provides", "who sells",
    "company name", "companies name", "link", "website", "url",
    "where to buy", "where can i buy", "where to purchase",
    "name the brands", "list the brands", "list brands",
]

CASUAL_RESPONSES = {
    "hi":           "Hello! 👋 How can I help you with Sea Buckthorn today?",
    "hello":        "Hello! 👋 How can I help you with Sea Buckthorn today?",
    "hey":          "Hey there! How can I help you today?",
    "thank you":    "You're welcome! Feel free to ask anything about Sea Buckthorn. 🌿",
    "thanks":       "You're welcome! Ask me anything about Sea Buckthorn. 🌿",
    "bye":          "Goodbye! Take care! 👋",
    "goodbye":      "Goodbye! Take care! 👋",
    "good morning": "Good morning! ☀️ How can I help you with Sea Buckthorn today?",
    "good evening": "Good evening! 🌙 How can I help you with Sea Buckthorn today?",
}

STYLE_INSTRUCTIONS = {
    "brief": (
        "Give a WELL-STRUCTURED and COMPLETE answer covering ALL important points. "
        "Use numbered points or bullet points. Include reasons, examples, and explanations. "
        "brief means clear and organized, not incomplete."
    ),
    "short":    "Give a SHORT summary in 3-4 lines only. Cover only the most important point.",
    "detailed": "Give a highly DETAILED answer with numbered sections, sub-points, and thorough explanations.",
    "simple":   "Use very simple plain language. No medical jargon. Short clear sentences.",
    "normal":   "Give a clear well-structured answer with bullet points and relevant details.",
}

# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    defaults = {
        "messages":             [],
        "default_vectorstore":  None,
        "uploaded_vectorstore": None,
        "uploaded_doc_texts":   {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# AI COMPONENTS
# =============================================================================

@st.cache_resource(show_spinner="Loading AI models...")
def load_base_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return embeddings, llm

embeddings, llm = load_base_components()

# =============================================================================
# FILE LOADING
# =============================================================================

def read_file(source, filename, is_upload=False):
    """Load any supported file. Returns (docs, full_text)."""
    suffix = os.path.splitext(filename)[1].lower()
    try:
        if is_upload:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(source.read())
                path = tmp.name
        else:
            path = source

        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = json.dumps(data, indent=2)
            from langchain_core.documents import Document
            docs = [Document(page_content=text, metadata={"source": filename})]
            if is_upload: os.unlink(path)
            return docs, text

        if suffix in IMAGE_EXTS:
            text = pytesseract.image_to_string(Image.open(path))
            if is_upload: os.unlink(path)
            if not text.strip(): return [], ""
            from langchain_core.documents import Document
            docs = [Document(page_content=text, metadata={"source": filename})]
            return docs, text

        loader_map = {
            ".pdf":      lambda p: PyPDFLoader(p),
            ".docx":     lambda p: Docx2txtLoader(p),
            ".txt":      lambda p: TextLoader(p, encoding="utf-8"),
            ".csv":      lambda p: CSVLoader(p, encoding="utf-8"),
            ".xls":      lambda p: UnstructuredExcelLoader(p),
            ".xlsx":     lambda p: UnstructuredExcelLoader(p),
            ".ppt":      lambda p: UnstructuredPowerPointLoader(p),
            ".pptx":     lambda p: UnstructuredPowerPointLoader(p),
            ".md":       lambda p: UnstructuredMarkdownLoader(p),
            ".markdown": lambda p: UnstructuredMarkdownLoader(p),
        }
        if suffix not in loader_map: return [], ""

        docs      = loader_map[suffix](path).load()
        for doc in docs: doc.metadata["source"] = filename
        full_text = "\n\n".join(d.page_content for d in docs)
        if is_upload: os.unlink(path)
        return docs, full_text

    except Exception as e:
        st.sidebar.error(f"❌ {filename}: {e}")
        return [], ""


def load_data_folder():
    """Load all files from the data/ folder."""
    docs = []
    if not os.path.exists(DATA_FOLDER): return docs
    for fname in os.listdir(DATA_FOLDER):
        fpath = os.path.join(DATA_FOLDER, fname)
        if os.path.isdir(fpath): continue
        d, _ = read_file(fpath, fname, is_upload=False)
        docs.extend(d)
    return docs


def build_vectorstore(documents):
    """Split documents and build FAISS vectorstore."""
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)

# =============================================================================
# QUERY CLASSIFICATION
# =============================================================================

def classify(user_input):
    """Classify query: 'casual' | 'offtopic' | 'uploaded_doc' | 'sea_buckthorn'"""
    txt = user_input.lower().strip()
    for kw in CASUAL_KEYWORDS:
        if kw == txt or txt.startswith(kw) or txt.endswith(kw) or f" {kw} " in f" {txt} ":
            return "casual"
    if any(kw in txt for kw in OFFTOPIC_KEYWORDS):    return "offtopic"
    if any(kw in txt for kw in UPLOADED_DOC_KEYWORDS): return "uploaded_doc"
    return "sea_buckthorn"


def get_response_style(user_input):
    """Detect response style: 'short' | 'brief' | 'detailed' | 'simple' | 'normal'"""
    q = user_input.lower()
    if any(w in q for w in ["in short", "tell me in short", "shortly", "tell me shortly", "quick", "summarize", "summary"]):
        return "short"
    if any(w in q for w in ["briefly", "tell me briefly", "brief", "in brief", "tell me in brief", "give me brief"]):
        return "brief"
    if any(w in q for w in ["detail", "detailed", "elaborate", "expand", "explain", "full"]):
        return "detailed"
    if any(w in q for w in ["simple", "easy", "beginner", "basic", "plain", "simply"]):
        return "simple"
    return "normal"

# =============================================================================
# MEMORY UTILITIES
# =============================================================================

def get_history(messages):
    """Return last N messages as a readable string."""
    if not messages: return "None"
    lines = []
    for msg in messages[-MEMORY_TURNS:]:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def is_followup(user_input):
    """Return True if the question is a follow-up on the previous topic."""
    return any(kw in user_input.lower() for kw in FOLLOWUP_PHRASES)


def find_last_real_topic(messages):
    """Return the last non-followup user question — the current conversation topic."""
    for msg in reversed(messages):
        if msg["role"] != "user": continue
        q = msg["content"].strip()
        if not any(kw in q.lower() for kw in FOLLOWUP_PHRASES):
            return q
    return ""

# =============================================================================
# ANSWER FUNCTIONS
# =============================================================================

def answer_casual(question):
    """Return a short friendly reply."""
    q = question.lower().strip()
    for key, response in CASUAL_RESPONSES.items():
        if key in q: return response
    prompt = ChatPromptTemplate.from_template(
        'User said: "{question}"\nReply in 1-2 friendly sentences. Do NOT mention Sea Buckthorn.\nREPLY:'
    )
    return (prompt | llm | StrOutputParser()).invoke({"question": question})


def answer_offtopic(_question):
    """Politely redirect off-topic questions."""
    return (
        "I'm a Sea Buckthorn healthcare specialist and can't help with that. 😊\n\n"
        "Feel free to ask me anything about Sea Buckthorn — benefits, nutrition, safety, uses, and more! 🌿"
    )


def answer_uploaded_doc(question, style):
    """Answer from uploaded files only (images + documents combined)."""
    if not st.session_state.uploaded_doc_texts:
        return (
            "⚠️ No uploaded documents found.\n\n"
            "Please upload a file using the sidebar and click **'📥 Load Documents'**."
        )

    context   = ""
    filenames = list(st.session_state.uploaded_doc_texts.keys())

    for fname, text in st.session_state.uploaded_doc_texts.items():
        is_image = fname.lower().endswith(IMAGE_EXTS)
        if is_image:
            label   = f"IMAGE FILE: {fname} (OCR text)"
            content = text.strip() if text.strip() else "No readable text found."
        else:
            label   = f"DOCUMENT FILE: {fname}"
            content = (text[:4000] + "\n...[truncated]") if len(text) > 4000 else text.strip()
            content = content or "No content extracted."
        context += f"\n\n=== {label} ===\n{content}"

    uploaded_style_map = {
        "brief":    "Give a structured answer for ALL files. Use bullet points. Do NOT skip any file.",
        "short":    "For each file, give 2-3 lines summarizing the most important point.",
        "detailed": "Give a detailed breakdown for each file with numbered points.",
        "simple":   "Use simple plain language for each file. No jargon.",
        "normal":   "For each file, give a clear summary of what it contains.",
    }

    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.

The user uploaded {count} file(s): {filenames}

CONTENT FROM ALL UPLOADED FILES:
{context}

USER ASKED: {question}

INSTRUCTIONS:
{style}
- Cover EVERY file — do NOT skip any
- Clearly label each file by name
- For IMAGE files: describe OCR-extracted text
- For DOCUMENT files: summarize the content
- Use ONLY information from the files above

ANSWER:""")

    return (prompt | llm | StrOutputParser()).invoke({
        "count":     len(filenames),
        "filenames": ", ".join(filenames),
        "context":   context,
        "question":  question,
        "style":     uploaded_style_map[style],
    })


def _answer_brand_query(question):
    """Read full txt file directly for brand/company/link queries."""
    full_content = ""
    if os.path.exists(DATA_FOLDER):
        for fname in os.listdir(DATA_FOLDER):
            if fname.endswith(".txt"):
                try:
                    with open(os.path.join(DATA_FOLDER, fname), "r", encoding="utf-8") as f:
                        full_content += f"\n\n=== {fname} ===\n" + f.read()
                except Exception:
                    pass

    if not full_content:
        return "I don't have brand information in my documents."

    prompt = ChatPromptTemplate.from_template("""You are a Sea Buckthorn healthcare expert.

FULL DOCUMENT:
{content}

USER QUESTION: {question}

INSTRUCTIONS:
- List ALL brands/companies found in the document
- Brands WITH details: show name, website, products, use cases
- Brands with NO details: print name ONLY — no other text
- NEVER write "Not provided", "Not specified", or "No details"
- Copy all URLs exactly as they appear

ANSWER:""")

    return (prompt | llm | StrOutputParser()).invoke({
        "content":  full_content,
        "question": question,
    })


def answer_sea_buckthorn(question, chat_history):
    """Main RAG answer: handles follow-ups, memory, and dual vectorstore retrieval."""
    history  = get_history(chat_history)
    followup = is_followup(question)
    style    = get_response_style(question)

    # Build search query
    if followup:
        real_topic   = find_last_real_topic(chat_history)
        search_query = f"Sea Buckthorn {real_topic[:200]}" if real_topic else f"Sea Buckthorn {question}"
        topic_label  = real_topic or question
        print(f"\n🔁 Follow-up | Topic: {topic_label[:60]}")
    else:
        if any(kw in question.lower() for kw in BRAND_KEYWORDS):
            return _answer_brand_query(question)
        search_query = f"Sea Buckthorn {question}"
        topic_label  = question
        print(f"\n🔍 Fresh: {search_query[:60]}")

    # Search both vectorstores — uploaded docs labeled separately
    uploaded_chunks = []
    default_chunks  = []

    if st.session_state.uploaded_vectorstore is not None:
        uploaded_chunks = (
            st.session_state.uploaded_vectorstore
            .as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVAL_K})
            .invoke(search_query)
        )

    if st.session_state.default_vectorstore is not None:
        default_chunks = (
            st.session_state.default_vectorstore
            .as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVAL_K})
            .invoke(search_query)
        )

    # Build context — uploaded docs first, clearly labeled
    context_parts = []
    if uploaded_chunks:
        context_parts.append("=== FROM YOUR UPLOADED DOCUMENTS ===")
        for d in uploaded_chunks:
            context_parts.append(f"[{d.metadata.get('source','?')}]: {d.page_content[:500]}")

    if default_chunks:
        context_parts.append("=== FROM DEFAULT KNOWLEDGE BASE ===")
        for d in default_chunks:
            context_parts.append(f"[{d.metadata.get('source','?')}]: {d.page_content[:500]}")

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template("""You are a Sea Buckthorn healthcare expert.

CONVERSATION HISTORY:
{history}

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

RESPONSE STYLE: {style}

RULES:
1. Current topic: {topic}
2. Answer ONLY about that topic — do NOT switch topics
3. Follow RESPONSE STYLE exactly
4. If "FROM YOUR UPLOADED DOCUMENTS" section exists in context — prefer that information first
5. Use ONLY information from CONTEXT above
6. If not in context: "I don't have that information in my documents."
7. Copy URLs and links exactly as they appear

ANSWER:""")

    return (prompt | llm | StrOutputParser()).invoke({
        "history":  history,
        "context":  context,
        "question": question,
        "style":    STYLE_INSTRUCTIONS[style],
        "topic":    topic_label,
    })


# =============================================================================
# MAIN ROUTER
# =============================================================================

def get_answer(question, chat_history):
    """Route question to the correct answer function."""
    query_type = classify(question)
    style      = get_response_style(question)
    print(f"\n🏷️  Type={query_type} | Style={style} | Followup={is_followup(question)}")

    if   query_type == "casual":       return answer_casual(question)
    elif query_type == "offtopic":     return answer_offtopic(question)
    elif query_type == "uploaded_doc": return answer_uploaded_doc(question, style)
    else:                              return answer_sea_buckthorn(question, chat_history)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("#### 📄 Upload Documents")
    st.caption("PDF, DOCX, TXT, CSV, Excel, PPT, Markdown, JSON, Images (JPG/PNG/BMP/TIFF)")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf","docx","txt","csv","xls","xlsx","ppt","pptx",
              "md","json","jpg","jpeg","png","bmp","tiff","webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("📥 Load Documents", use_container_width=True):
        with st.spinner("Processing..."):
            if uploaded_files:
                all_docs, all_texts = [], {}
                for uf in uploaded_files:
                    docs, text = read_file(uf, uf.name, is_upload=True)
                    all_docs.extend(docs)
                    if text: all_texts[uf.name] = text
                if all_docs:
                    st.session_state.uploaded_vectorstore = build_vectorstore(all_docs)
                    st.session_state.uploaded_doc_texts   = all_texts
                    st.success(f"✅ Loaded: {', '.join(all_texts.keys())}")
                else:
                    st.error("Could not load files.")
            else:
                st.warning("Please select files first.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =============================================================================
# AUTO-LOAD DEFAULT KNOWLEDGE BASE
# =============================================================================

if st.session_state.default_vectorstore is None:
    with st.spinner("📚 Loading default knowledge base..."):
        default_docs = load_data_folder()
        if default_docs:
            st.session_state.default_vectorstore = build_vectorstore(default_docs)

# =============================================================================
# MAIN CHAT UI
# =============================================================================

st.title("🌿 Sea Buckthorn Healthcare Chatbot")
st.caption("Upload documents and ask anything — I answer from the right source!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything about Sea Buckthorn...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer = "Sorry, I encountered an error. Please try again."

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = get_answer(
                    question     = user_input,
                    chat_history = st.session_state.messages[:-1],
                )
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                print(f"ERROR: {e}")

    st.session_state.messages.append({"role": "assistant", "content": answer})