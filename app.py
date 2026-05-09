import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
# AI COMPONENTS — load once
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
# FILE LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def read_file(source, filename, is_upload=False):
    """Load any PDF/DOCX/TXT. Returns (docs, full_text)."""
    suffix = os.path.splitext(filename)[1].lower()
    try:
        if is_upload:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(source.read())
                path = tmp.name
        else:
            path = source

        if   suffix == ".pdf":  loader = PyPDFLoader(path)
        elif suffix == ".docx": loader = Docx2txtLoader(path)
        elif suffix == ".txt":  loader = TextLoader(path, encoding="utf-8")
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
# SIDEBAR — clean, only what's needed
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    st.markdown("#### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
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
    st.markdown("#### 🧠 Memory Size")
    memory_turns = st.slider(
        "Past messages to remember",
        min_value=2, max_value=10, value=6, step=2,
        label_visibility="collapsed"
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-LOAD DEFAULT DOCS ON FIRST RUN
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.default_vectorstore is None:
    with st.spinner("📚 Loading default knowledge base..."):
        default_docs = load_data_folder()
        if default_docs:
            st.session_state.default_vectorstore = build_vectorstore(default_docs)

# ─────────────────────────────────────────────────────────────────────────────
# QUERY CLASSIFICATION — done FIRST before any LLM call
# ─────────────────────────────────────────────────────────────────────────────

# Exact casual phrases — reply friendly, no Sea Buckthorn
CASUAL = [
    "hi", "hello", "hey", "hiya", "howdy",
    "how are you", "how r u", "how are u",
    "how was your day", "how is your day",
    "good morning", "good evening", "good afternoon", "good night",
    "what's up", "whats up", "sup",
    "who are you", "what are you",
    "thank you", "thanks", "thank u", "thx",
    "bye", "goodbye", "see you", "see ya",
    "ok", "okay", "cool", "great", "nice", "awesome",
    "well done", "good job", "nice work",
]

# Off-topic subjects — redirect politely
OFFTOPIC = [
    "cricket", "football", "soccer", "ipl", "nba", "sports",
    "movie", "film", "series", "netflix", "song", "music", "album",
    "weather", "rain", "temperature",
    "stock", "share market", "bitcoin", "crypto",
    "politics", "election", "government", "war",
    "recipe", "cooking", "restaurant",
    "travel", "tour", "vacation", "trip",
    "joke", "funny", "meme",
    "girlfriend", "boyfriend", "love", "marriage",
    "programming", "code", "software", "developer",
]

# Uploaded doc phrases — answer from uploaded file only
UPLOADED_DOC = [
    "my uploaded document", "my document", "my pdf", "my file", "my book",
    "uploaded document", "uploaded file", "uploaded pdf", "uploaded book",
    "this document", "this file", "this pdf", "this book",
    "what i uploaded", "the document i uploaded",
    "summarize my", "summary of my",
    "tell me about my document", "tell me about my file", "tell me about my pdf",
    "what is in my", "overview of my",
    "shortly about my", "brief about my",
    "from my document", "from my file", "from my pdf",
    "in my document", "in my file", "in my pdf",
    "about my uploaded",
]

def classify(user_input):
    """
    Returns: 'casual' | 'offtopic' | 'uploaded_doc' | 'sea_buckthorn'
    Checks exact keyword lists — no LLM needed, zero tokens used.
    """
    txt = user_input.lower().strip()

    # Check casual first — simple greetings/thanks
    for kw in CASUAL:
        if kw == txt or txt.startswith(kw) or txt.endswith(kw) or f" {kw} " in f" {txt} ":
            return "casual"

    # Check off-topic
    for kw in OFFTOPIC:
        if kw in txt:
            return "offtopic"

    # Check uploaded doc reference
    for kw in UPLOADED_DOC:
        if kw in txt:
            return "uploaded_doc"

    # Everything else = Sea Buckthorn question
    return "sea_buckthorn"


def detect_style(txt):
    txt = txt.lower()
    if any(w in txt for w in ["short","shortly","brief","briefly","summarize","summary","quick","overview"]):
        return "short"
    if any(w in txt for w in ["detail","detailed","full","elaborate","specific","expand","explain"]):
        return "detailed"
    if any(w in txt for w in ["simple","easy","beginner","basic","plain"]):
        return "simple"
    return "normal"

# ─────────────────────────────────────────────────────────────────────────────
# HISTORY HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_history(messages, memory_turns):
    if not messages: return "None"
    lines = []
    for msg in messages[-memory_turns:]:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:250] + "..." if len(msg["content"]) > 250 else msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# ANSWER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def answer_casual(question):
    """1-2 sentence friendly reply. No Sea Buckthorn content."""
    casual_responses = {
        "hi": "Hello! 👋 How can I help you with Sea Buckthorn today?",
        "hello": "Hello! 👋 How can I help you with Sea Buckthorn today?",
        "hey": "Hey! How can I help you today?",
        "thank you": "You're welcome! Feel free to ask anything about Sea Buckthorn. 🌿",
        "thanks": "You're welcome! Feel free to ask anything about Sea Buckthorn. 🌿",
        "bye": "Goodbye! Take care! 👋",
        "goodbye": "Goodbye! Take care! 👋",
        "good morning": "Good morning! ☀️ Hope you have a great day. How can I help you with Sea Buckthorn?",
        "good evening": "Good evening! 🌙 How can I help you with Sea Buckthorn today?",
    }
    q = question.lower().strip()
    for key, response in casual_responses.items():
        if key in q:
            return response

    # For other casual phrases use a tiny LLM call
    prompt = ChatPromptTemplate.from_template(
        'The user said: "{question}"\n'
        "Reply in exactly 1-2 friendly sentences. "
        "Do NOT mention Sea Buckthorn or give any health info. "
        "Just be warm and natural.\nREPLY:"
    )
    return (prompt | llm | StrOutputParser()).invoke({"question": question})


def answer_offtopic(question):
    return (
        "I'm a Sea Buckthorn healthcare specialist and can't help with that topic. 😊\n\n"
        "Ask me anything about Sea Buckthorn — its benefits, nutrition, safety, uses, and more! 🌿"
    )


def answer_uploaded_doc(question, style):
    """Answer from uploaded docs ONLY — completely separate from data/ folder."""
    if not st.session_state.uploaded_doc_texts:
        return (
            "⚠️ No uploaded documents found.\n\n"
            "Please upload a file using the sidebar and click **'📥 Load Documents'**."
        )

    # Build context from uploaded docs full text (not retriever — avoids mixing)
    context = ""
    for fname, text in st.session_state.uploaded_doc_texts.items():
        # Take up to 3000 chars per doc
        snippet = text[:3000] + "\n...[truncated for length]" if len(text) > 3000 else text
        context += f"\n\n=== FILE: {fname} ===\n{snippet}"

    filenames = ", ".join(st.session_state.uploaded_doc_texts.keys())

    style_map = {
        "short":    "Give a SHORT summary: 5-7 bullet points. Be concise.",
        "detailed": "Give a DETAILED summary with clear sections and numbered points.",
        "simple":   "Give a SIMPLE summary in plain everyday language. No jargon.",
        "normal":   "Give a clear structured summary of the main topics.",
    }

    prompt = ChatPromptTemplate.from_template("""You are a Sea Buckthorn healthcare expert.

The user uploaded this document: {filenames}

DOCUMENT CONTENT (answer ONLY from this):
{context}

USER ASKED: {question}

INSTRUCTIONS:
{style}
- Answer ONLY from the document content above
- Do NOT use any other knowledge
- Cover the main points clearly
- If the document doesn't contain the answer, say so

YOUR ANSWER:""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "filenames": filenames,
        "context":   context,
        "question":  question,
        "style":     style_map[style]
    })


def answer_sea_buckthorn(question, chat_history, memory_turns, style):
    """Answer from default data/ folder with full conversation memory."""
    if st.session_state.default_vectorstore is None:
        return "Default knowledge base not loaded. Please add files to the data/ folder."

    history = get_history(chat_history, memory_turns)

    # Detect follow-up vs fresh question — no LLM needed
    FOLLOWUP = [
        "more specific", "be specific", "elaborate", "tell me more",
        "more detail", "expand", "explain more", "go deeper",
        "in short", "briefly", "give example", "more about it",
        "more about that", "continue", "what else", "and then",
    ]
    is_followup = any(kw in question.lower() for kw in FOLLOWUP)

    if is_followup:
        # Extract topic from last assistant message
        lines = history.split("\n")
        last_bot = next((l.replace("Assistant:", "").strip()
                         for l in reversed(lines) if l.startswith("Assistant:")), "")
        search_query = f"Sea Buckthorn {last_bot[:150]}" if last_bot else f"Sea Buckthorn {question}"
    else:
        # For brand/company/link questions — read the FULL txt file directly
        # This bypasses chunking issues entirely and guarantees all brands are found
        BRAND_KEYWORDS = [
            "brand", "brands", "name the brand", "brand name",
            "which company", "which brand", "who provides", "who sells",
            "company name", "companies name", "link", "website", "url",
            "where to buy", "where can i buy", "where to purchase",
            "name the brands", "list the brands", "list brands"
        ]
        is_brand_query = any(kw in question.lower() for kw in BRAND_KEYWORDS)

        if is_brand_query:
            # Read full txt files directly — no retriever, no chunking issues
            full_file_content = ""
            if os.path.exists("data"):
                for fname in os.listdir("data"):
                    fpath = os.path.join("data", fname)
                    if fname.endswith(".txt"):
                        try:
                            with open(fpath, "r", encoding="utf-8") as f:
                                full_file_content += f"\n\n=== {fname} ===\n" + f.read()
                        except:
                            pass
            if full_file_content:
                brand_prompt = ChatPromptTemplate.from_template("""You are a Sea Buckthorn healthcare expert.

FULL DOCUMENT CONTENT:
{content}

USER QUESTION: {question}

INSTRUCTIONS:
- Find all brand names and companies mentioned in the document
- For brands WITH full details (website, products, use cases): show all details
- For brands with NO details: print the name ONLY — no other text whatsoever
- STRICTLY FORBIDDEN to write: "Not provided", "Not specified", "No details", "not mentioned", or any similar phrase
- Copy website URLs exactly as they appear
- Do NOT add information from your own knowledge

YOUR ANSWER:""")
                brand_chain = brand_prompt | llm | StrOutputParser()
                return brand_chain.invoke({"content": full_file_content, "question": question})
            search_query = "Sea Buckthorn companies brands WellWith Vedberry"
        else:
            search_query = f"Sea Buckthorn {question}"

    print(f"\n🔍 Search: {search_query}")

    # Fetch more chunks for link/brand queries so URLs are never cut off
    LINK_KEYWORDS = [
        "link", "url", "website",
        "name the brand", "brand name", "which brand", "which company",
        "who provides", "who sells", "where to buy", "where to purchase",
        "wellwith", "vedberry", "leh berry", "himaleh", "biosash", "nutriorg",
        "patanjali", "miracle seabuck", "top 3 brand", "top 5 brand"
    ]
    k = 8 if any(kw in question.lower() for kw in LINK_KEYWORDS) else 4

    retriever = st.session_state.default_vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    docs = retriever.invoke(search_query)

    # For link queries return FULL chunk content — never truncate URLs
    if k == 8:
        context = "\n\n".join(
            f"[{d.metadata.get('source','?')}]:\n{d.page_content}" for d in docs
        )
    else:
        context = "\n\n".join(
            f"[{d.metadata.get('source','?')}]: {d.page_content[:400]}" for d in docs
        )

    style_map = {
        "short":    "Answer in 3-5 lines only. No bullet points.",
        "detailed": "Give a detailed structured answer with numbered points.",
        "simple":   "Use very simple plain language. No medical terms.",
        "normal":   "Give a clear well-structured answer with bullet points where helpful.",
    }

    prompt = ChatPromptTemplate.from_template("""You are a Sea Buckthorn healthcare expert.

CONVERSATION HISTORY:
{history}

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
{style}

IMPORTANT:
- If user says "be more specific" / "elaborate" / "tell me more":
  Look at the conversation history. Find the LAST topic discussed.
  Give more detail on THAT SAME TOPIC. Do NOT switch topics.
- Use ONLY information from the context above
- If not in context: "I don't have that information in my documents."
- If the context contains links or URLs, copy them EXACTLY as they appear — do not modify or summarize them
- For brand/company questions: list ALL brands mentioned in the context — do not skip any
- For each brand include: name, website link, products, and use cases exactly as in the context
- Do NOT add brands from your own knowledge — only from the context

ANSWER:""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "history":  history,
        "context":  context,
        "question": question,
        "style":    style_map[style]
    })

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────

def get_answer(question, chat_history, memory_turns):
    query_type = classify(question)
    style      = detect_style(question)

    print(f"\n🏷️  Type: {query_type} | Style: {style}")

    if   query_type == "casual":       return answer_casual(question)
    elif query_type == "offtopic":     return answer_offtopic(question)
    elif query_type == "uploaded_doc": return answer_uploaded_doc(question, style)
    else:                              return answer_sea_buckthorn(question, chat_history, memory_turns, style)

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
                    question     = user_input,
                    chat_history = st.session_state.messages[:-1],
                    memory_turns = memory_turns
                )
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                print(f"ERROR: {e}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
