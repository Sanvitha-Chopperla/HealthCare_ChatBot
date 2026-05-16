

import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader,
    CSVLoader, UnstructuredExcelLoader,
    UnstructuredPowerPointLoader, UnstructuredMarkdownLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.ragas_eval import evaluate_rag
from PIL import Image
import pytesseract

# Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()

st.set_page_config(page_title="Sea Buckthorn Healthcare Chatbot", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "messages"             not in st.session_state: st.session_state.messages             = []
if "default_vectorstore"  not in st.session_state: st.session_state.default_vectorstore  = None
if "uploaded_vectorstore" not in st.session_state: st.session_state.uploaded_vectorstore = None
if "uploaded_doc_texts"   not in st.session_state: st.session_state.uploaded_doc_texts   = {}

# ─────────────────────────────────────────────────────────────────────────────
# AI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI models...")
def load_base_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return embeddings, llm

embeddings, llm = load_base_components()

# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

def read_file(source, filename, is_upload=False):
    suffix = os.path.splitext(filename)[1].lower()
    try:
        if is_upload:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(source.read())
                path = tmp.name
        else:
            path = source

        # JSON
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            text = json.dumps(data, indent=2)
            from langchain_core.documents import Document
            docs = [Document(page_content=text, metadata={"source": filename})]
            if is_upload: os.unlink(path)
            return docs, text

        # Images — OCR
        elif suffix in IMAGE_EXTS:
            image = Image.open(path)
            text  = pytesseract.image_to_string(image)
            if is_upload: os.unlink(path)
            if not text.strip():
                return [], ""
            from langchain_core.documents import Document
            docs = [Document(page_content=text, metadata={"source": filename})]
            return docs, text

        # All other formats
        elif suffix == ".pdf":   loader = PyPDFLoader(path)
        elif suffix == ".docx":  loader = Docx2txtLoader(path)
        elif suffix == ".txt":   loader = TextLoader(path, encoding="utf-8")
        elif suffix == ".csv":   loader = CSVLoader(path, encoding="utf-8")
        elif suffix in (".xls", ".xlsx"): loader = UnstructuredExcelLoader(path)
        elif suffix in (".ppt", ".pptx"): loader = UnstructuredPowerPointLoader(path)
        elif suffix in (".md", ".markdown"): loader = UnstructuredMarkdownLoader(path)
        else: return [], ""

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        full_text = "\n\n".join(d.page_content for d in docs)
        if is_upload: os.unlink(path)
        return docs, full_text

    except Exception as e:
        st.sidebar.error(f"❌ {filename}: {e}")
        return [], ""


def load_data_folder():
    docs = []
    if not os.path.exists("data"): return docs
    for fname in os.listdir("data"):
        fpath = os.path.join("data", fname)
        if os.path.isdir(fpath): continue
        d, _ = read_file(fpath, fname, is_upload=False)
        docs.extend(d)
    return docs


def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("#### 📄 Upload Documents")
    st.caption("PDF, DOCX, TXT, CSV, Excel, PPT, Markdown, JSON, Images (JPG/PNG/BMP/TIFF)")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf","docx","txt","csv","xls","xlsx","ppt","pptx",
              "md","json","jpg","jpeg","png","bmp","tiff","webp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
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

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-LOAD DEFAULT DOCS
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.default_vectorstore is None:
    with st.spinner("📚 Loading default knowledge base..."):
        default_docs = load_data_folder()
        if default_docs:
            st.session_state.default_vectorstore = build_vectorstore(default_docs)

# ─────────────────────────────────────────────────────────────────────────────
# QUERY CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

CASUAL = [
    "hi","hello","hey","hiya","howdy",
    "how are you","how r u","how are u",
    "how was your day","how is your day",
    "good morning","good evening","good afternoon","good night",
    "what's up","whats up","sup",
    "who are you","what are you",
    "thank you","thanks","thank u","thx",
    "bye","goodbye","see you","see ya",
    "ok","okay","cool","great","nice","awesome",
    "well done","good job","nice work",
]

OFFTOPIC = [
    "cricket","football","soccer","ipl","nba","sports",
    "movie","film","series","netflix","song","music","album",
    "weather","rain","temperature",
    "stock","share market","bitcoin","crypto",
    "politics","election","government","war",
    "recipe","cooking","restaurant",
    "travel","tour","vacation","trip",
    "joke","funny","meme",
    "girlfriend","boyfriend","love","marriage",
    "programming","code","software","developer",
]

UPLOADED_DOC = [
    "my uploaded document","my document","my pdf","my file","my book",
    "my image","my photo","my picture","my screenshot",
    "uploaded document","uploaded file","uploaded pdf","uploaded book",
    "uploaded image","uploaded photo","uploaded picture",
    "this document","this file","this pdf","this book",
    "this image","this photo","this picture",
    "what i uploaded","the document i uploaded",
    "summarize my","summary of my",
    "tell me about my document","tell me about my file","tell me about my pdf",
    "tell me about my image","tell me about my photo","tell me about my picture",
    "what is in my","overview of my",
    "shortly about my","brief about my",
    "from my document","from my file","from my pdf",
    "in my document","in my file","in my pdf",
    "about my uploaded",
    "what is there in","describe the image","describe the picture",
    "describe my image","describe my picture","describe my photo",
    "read the image","read the picture","what does the image say",
    "text in the image","text in the picture",
    "what is shown","tell me about the image","tell me about the picture",
]


def classify(user_input):
    txt = user_input.lower().strip()

    for kw in CASUAL:
        if kw == txt or txt.startswith(kw) or txt.endswith(kw) or f" {kw} " in f" {txt} ":
            return "casual"

    for kw in OFFTOPIC:
        if kw in txt:
            return "offtopic"

    for kw in UPLOADED_DOC:
        if kw in txt:
            return "uploaded_doc"

    return "sea_buckthorn"

# ─────────────────────────────────────────────────────────────────────────────
# HISTORY HELPER
# ─────────────────────────────────────────────────────────────────────────────

# These phrases mark a follow-up question (not a new topic)
FOLLOWUP_PHRASES = [
    # These mean "give me more/different format on the SAME topic"
    "more specific", "be specific", "be more specific",
    "elaborate", "tell me more", "more detail", "more details",
    "expand", "explain more", "go deeper", "give example", "give examples",
    "more about it", "more about that", "continue", "what else",
    "specifically", "in detail", "say more", "and also",
    "in simple", "make it simple", "explain simply",
    "tell me briefly", "tell me in brief", "tell me in short",
    "tell me shortly", "give me brief", "give me short",
]


def get_history(messages, memory_turns=10):
    """Return last N messages as readable string."""
    if not messages: return "None"
    lines = []
    for msg in messages[-memory_turns:]:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def find_last_real_topic(messages):
    """
    Walk backwards through messages to find the last REAL topic question.
    Skips follow-up questions like 'tell me briefly', 'be more specific', etc.
    """
    for msg in reversed(messages):
        if msg["role"] != "user":
            continue
        q       = msg["content"].strip()
        q_lower = q.lower()
        # Skip only if it's an explicit follow-up phrase
        is_followup = any(kw in q_lower for kw in FOLLOWUP_PHRASES)
        if not is_followup:
            return q
    return ""


def is_followup_question(user_input):
    """Check if the current question is a follow-up."""
    q_lower = user_input.lower().strip()
    return any(kw in q_lower for kw in FOLLOWUP_PHRASES)


def get_response_style(user_input):
    """Detect what style of response the user wants."""
    q = user_input.lower()
    # "in short" / "shortly" = actually short 3-4 lines
    if any(w in q for w in ["in short", "tell me in short", "shortly", "tell me shortly",
                             "quick", "summarize", "summary"]):
        return "short"
    # "brief" / "briefly" / "in brief" = structured complete answer
    if any(w in q for w in ["briefly", "tell me briefly", "brief", "in brief",
                             "tell me in brief", "give me brief", "give brief"]):
        return "brief"
    if any(w in q for w in ["detail", "detailed", "elaborate", "expand", "explain", "full"]):
        return "detailed"
    if any(w in q for w in ["simple", "easy", "beginner", "basic", "plain", "simply"]):
        return "simple"
    return "normal"

# ─────────────────────────────────────────────────────────────────────────────
# ANSWER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def answer_casual(question):
    quick = {
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
    q = question.lower().strip()
    for key, response in quick.items():
        if key in q:
            return response

    prompt = ChatPromptTemplate.from_template(
        'User said: "{question}"\n'
        "Reply in 1-2 friendly sentences only. "
        "Do NOT mention Sea Buckthorn. Just be warm and natural.\nREPLY:"
    )
    return (prompt | llm | StrOutputParser()).invoke({"question": question})


def answer_offtopic(question):
    return (
        "I'm a Sea Buckthorn healthcare specialist and can't help with that. 😊\n\n"
        "Feel free to ask me anything about Sea Buckthorn — benefits, nutrition, safety, uses, and more! 🌿"
    )


def answer_uploaded_doc(question, style):
    if not st.session_state.uploaded_doc_texts:
        return (
            "⚠️ No uploaded documents found.\n\n"
            "Please upload a file using the sidebar and click **'📥 Load Documents'**."
        )

    IMAGE_EXTS_LOWER = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
    all_filenames    = list(st.session_state.uploaded_doc_texts.keys())
    context          = ""

    # Build context from EVERY uploaded file — image or document
    for fname, text in st.session_state.uploaded_doc_texts.items():
        is_image = fname.lower().endswith(IMAGE_EXTS_LOWER)
        if is_image:
            if text.strip():
                context += f"\n\n=== IMAGE FILE: {fname} ===\nText found in image (via OCR):\n{text.strip()}"
            else:
                context += f"\n\n=== IMAGE FILE: {fname} ===\nNo readable text found in this image."
        else:
            if text.strip():
                snippet  = text[:4000] + "\n...[content truncated]" if len(text) > 4000 else text
                context += f"\n\n=== DOCUMENT FILE: {fname} ===\n{snippet.strip()}"
            else:
                context += f"\n\n=== DOCUMENT FILE: {fname} ===\nNo content could be extracted."

    filenames = ", ".join(all_filenames)

    style_instruction = {
        "brief":    "Give a structured answer covering ALL main points for each file. Use bullet points per file. Do NOT skip any file. Be organized and complete.",
        "short":    "For each file, give 2-3 lines only summarizing the most important point.",
        "detailed": "Give a detailed breakdown for each file with numbered points and thorough explanation.",
        "simple":   "Use simple plain language for each file. No jargon.",
        "normal":   "For each file, give a clear summary of what it contains.",
    }[style]

    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.

The user uploaded {count} file(s): {filenames}

CONTENT FROM ALL UPLOADED FILES:
{context}

USER ASKED: {question}

INSTRUCTIONS:
{style}

CRITICAL:
- You MUST cover EVERY file listed above — do NOT skip any
- Start each file's section with its filename clearly
- For IMAGE files: explain what text was found inside them
- For DOCUMENT files: summarize their content
- If a file has no readable content, mention that clearly

ANSWER:""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "count":     len(all_filenames),
        "filenames": filenames,
        "context":   context,
        "question":  question,
        "style":     style_instruction
    })


def answer_sea_buckthorn(question, chat_history):
    if st.session_state.default_vectorstore is None:
        return "Default knowledge base not loaded. Please add files to the data/ folder."

    history       = get_history(chat_history)
    is_followup   = is_followup_question(question)
    style         = get_response_style(question)

    # ── Build search query ─────────────────────────────────────────────────
    if is_followup:
        # Find the last REAL topic question from history
        real_topic   = find_last_real_topic(chat_history)
        search_query = f"Sea Buckthorn {real_topic[:200]}" if real_topic else f"Sea Buckthorn {question}"
        print(f"\n🔁 Follow-up | Topic: {real_topic[:60]}")
    else:
        # Brand/company/link — read full file directly
        BRAND_KEYWORDS = [
            "brand","brands","name the brand","brand name",
            "which company","which brand","who provides","who sells",
            "company name","companies name","link","website","url",
            "where to buy","where can i buy","where to purchase",
            "name the brands","list the brands","list brands"
        ]
        if any(kw in question.lower() for kw in BRAND_KEYWORDS):
            full_content = ""
            if os.path.exists("data"):
                for fname in os.listdir("data"):
                    fpath = os.path.join("data", fname)
                    if fname.endswith(".txt"):
                        try:
                            with open(fpath, "r", encoding="utf-8") as f:
                                full_content += f"\n\n=== {fname} ===\n" + f.read()
                        except: pass
            if full_content:
                brand_prompt = ChatPromptTemplate.from_template("""You are a Sea Buckthorn healthcare expert.

FULL DOCUMENT:
{content}

USER QUESTION: {question}

INSTRUCTIONS:
- List ALL brands/companies found in the document
- For brands WITH details (website, products, use cases): show all details
- For brands with NO details: print name ONLY — nothing else
- NEVER write "Not provided", "Not specified", "No details"
- Copy URLs exactly as they appear

ANSWER:""")
                return (brand_prompt | llm | StrOutputParser()).invoke({
                    "content":  full_content,
                    "question": question
                })
        search_query = f"Sea Buckthorn {question}"
        print(f"\n🔍 Fresh query: {search_query[:60]}")

    # ── Retrieve chunks — search uploaded docs first, then default ────────
    k         = 6
    all_docs  = []

    # If user has uploaded documents — search them FIRST
    if st.session_state.uploaded_vectorstore is not None:
        up_retriever = st.session_state.uploaded_vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        all_docs.extend(up_retriever.invoke(search_query))

    # Also search default knowledge base
    if st.session_state.default_vectorstore is not None:
        def_retriever = st.session_state.default_vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        all_docs.extend(def_retriever.invoke(search_query))

    context = "\n\n".join(
        f"[{d.metadata.get('source','?')}]: {d.page_content[:500]}"
        for d in all_docs
    )

    # ── Style instruction ──────────────────────────────────────────────────
    style_instruction = {
        "brief":    "Give a WELL-STRUCTURED and COMPLETE answer covering ALL important points. Use numbered points or bullet points. Include reasons, examples, and explanations. Do NOT cut the answer short — brief means clear and organized, not incomplete.",
        "short":    "Give a SHORT summary in 3-4 lines only. Cover only the most important point.",
        "detailed": "Give a highly DETAILED answer with numbered sections, sub-points, examples, and thorough explanations of every aspect.",
        "simple":   "Use very simple plain language. No medical jargon. Short clear sentences. Easy to understand.",
        "normal":   "Give a clear well-structured answer with bullet points, explanations, and relevant details.",
    }[style]

    # ── Answer prompt ──────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_template("""You are a Sea Buckthorn healthcare expert.

CONVERSATION HISTORY:
{history}

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

RESPONSE STYLE: {style}

RULES:
1. Read the CONVERSATION HISTORY carefully to understand the current topic
2. The current topic being discussed is: {topic}
3. Answer about THAT TOPIC — do NOT switch to a different topic
4. Follow the RESPONSE STYLE exactly
5. Use ONLY information from the CONTEXT
6. If not in context: "I don't have that information in my documents."
7. Copy URLs/links exactly as they appear

ANSWER:""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "history":  history,
        "context":  context,
        "question": question,
        "style":    style_instruction,
        "topic":    find_last_real_topic(chat_history) if is_followup else question
    })
# ─────────────────────────────
# 📊 SIMPLE EVALUATION (ADD HERE)
# ─────────────────────────────
def simple_evaluation(question, answer, context):
    score = {}

    score["relevance"] = 1 if any(w in answer.lower() for w in question.lower().split()) else 0.5
    score["length_score"] = min(len(answer) / 500, 1)
    score["context_usage"] = 1 if context else 0.5

    return score

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────

def get_answer(question, chat_history):
    query_type = classify(question)
    style      = get_response_style(question)

    print(f"\n🏷️  Type: {query_type} | Style: {style} | Followup: {is_followup_question(question)}")

    if   query_type == "casual":       return answer_casual(question)
    elif query_type == "offtopic":     return answer_offtopic(question)
    elif query_type == "uploaded_doc": return answer_uploaded_doc(question, style)
    else:                              return answer_sea_buckthorn(question, chat_history)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT UI
# ─────────────────────────────────────────────────────────────────────────────

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
                question=user_input,
                chat_history=st.session_state.messages[:-1]
                )

                st.markdown(answer)

                # ─────────────────────────────
                # 📊 SIMPLE EVALUATION (WORKING)
                # ─────────────────────────────
                with st.expander("📊 Evaluation", expanded=False):
                    scores = simple_evaluation(
                        user_input,
                        answer,
                        context=None  # Replace with actual context if available
                    )

                    for k, v in scores.items():
                        st.metric(k, round(v, 2))

                # ❌ REMOVE RAGAS COMPLETELY
                # (DO NOT USE run_ragas_evaluation)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})