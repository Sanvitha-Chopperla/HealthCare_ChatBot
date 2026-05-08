# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# load_dotenv()

# st.set_page_config(page_title="Sea Buckthorn Healthcare Chatbot")
# st.title("🌿 Sea Buckthorn Healthcare Chatbot")
# st.write("Ask questions about Sea Buckthorn health benefits.")


# @st.cache_resource(show_spinner="Loading Sea Buckthorn knowledge base...")
# def get_chain():
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     if not os.path.exists("vectorstore"):
#         vectorstore = create_vectorstore()  # your function
#     else:
#         vectorstore = FAISS.load_local(
#         "vectorstore",
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     retriever = vectorstore.as_retriever()

#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         temperature=0.3,
#         api_key=os.getenv("GROQ_API_KEY")
#     )

#     # FIX: renamed to 'chat_prompt' to avoid conflict with st.chat_input variable
#     chat_prompt = ChatPromptTemplate.from_template("""
# You are a healthcare assistant specializing in Sea Buckthorn.
# Use the context below to answer clearly and simply.
# If the answer is not in the context, say: 'I do not have enough information on that.'

# Context:
# {context}

# Question:
# {question}

# Answer:
# """)

#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)

#     chain = (
#         {
#             "context": retriever | format_docs,
#             "question": RunnablePassthrough()
#         }
#         | chat_prompt
#         | llm
#         | StrOutputParser()
#     )

#     return chain


# # ---- Chat UI ----

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# qa_chain = get_chain()

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # FIX: renamed from 'prompt' to 'user_input' to avoid variable conflict
# user_input = st.chat_input("Ask your question about Sea Buckthorn...")

# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     with st.chat_message("user"):
#         st.markdown(user_input)

#     # FIX: initialize answer before try block to avoid UnboundLocalError
#     answer = "Sorry, I encountered an error. Please try again."

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             try:
#                 answer = qa_chain.invoke(user_input)
#                 st.markdown(answer)
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")
#                 print(e)

#     st.session_state.messages.append({"role": "assistant", "content": answer})

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------- LOAD ENV ---------------- #

load_dotenv()

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Sea Buckthorn Healthcare Chatbot")

st.title("🌿 Sea Buckthorn Healthcare Chatbot")

st.write("Ask questions about Sea Buckthorn health benefits.")

# ---------------- LOAD CHATBOT ---------------- #

@st.cache_resource(show_spinner="Loading Sea Buckthorn knowledge base...")
def get_chain():

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS Vector Database
    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retriever
    retriever = vectorstore.as_retriever()

    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Prompt
    chat_prompt = ChatPromptTemplate.from_template("""
You are a healthcare assistant specializing in Sea Buckthorn.

Use the context below to answer clearly and simply.

If the answer is not available in the context,
say:
'I do not have enough information on that.'

Context:
{context}

Question:
{question}

Answer:
""")

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL Chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | chat_prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ---------------- SESSION STATE ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- LOAD QA CHAIN ---------------- #

qa_chain = get_chain()

# ---------------- DISPLAY OLD MESSAGES ---------------- #

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ---------------- #

user_input = st.chat_input(
    "Ask your question about Sea Buckthorn..."
)

# ---------------- CHAT PROCESS ---------------- #

if user_input:

    # Store User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Default Error Message
    answer = "Sorry, I encountered an error. Please try again."

    # Assistant Response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer = qa_chain.invoke(user_input)

                st.markdown(answer)

            except Exception as e:

                st.error(f"Error: {str(e)}")

                print(e)

    # Store Assistant Message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )