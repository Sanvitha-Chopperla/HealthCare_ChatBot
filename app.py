# import streamlit as st

# from utils.chatbot import load_chatbot

# st.set_page_config(
#     page_title="Sea Buckthorn Healthcare Chatbot"
# )

# st.title("🌿 Sea Buckthorn Healthcare Chatbot")

# st.write(
#     "Ask questions about Sea Buckthorn health benefits."
# )

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# qa_chain = load_chatbot()

# # Display old messages
# for message in st.session_state.messages:

#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         prompt = st.chat_input("Ask your question...")

# if prompt:

#     # Store user message
#     st.session_state.messages.append({
#         "role": "user",
#         "content": prompt
#     })

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Assistant response
#     with st.chat_message("assistant"):

#         with st.spinner("Thinking..."):

#             response = qa_chain.invoke({"input": prompt})

#             st.markdown(response)

#     # Store assistant response
#     st.session_state.messages.append({
#         "role": "assistant",
#          "content": response
#     })

# import streamlit as st
# from utils.chatbot import load_chatbot

# st.set_page_config(page_title="Sea Buckthorn Healthcare Chatbot")

# st.title("🌿 Sea Buckthorn Healthcare Chatbot")
# st.write("Ask questions about Sea Buckthorn health benefits.")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Load chatbot once and cache it
# @st.cache_resource
# def get_chain():
#     return load_chatbot()

# qa_chain = get_chain()

# # Display old messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # ✅ chat_input must be OUTSIDE the loop
# prompt = st.chat_input("Ask your question about Sea Buckthorn...")

# if prompt:

#     # Store and show user message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Get and show assistant response
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             result = qa_chain.invoke({"input": prompt})
#             # ✅ extract the answer from result dict
#             answer = result["answer"]
#             st.markdown(answer)

#     # Store assistant response
#     st.session_state.messages.append({"role": "assistant", "content": answer})

# import streamlit as st
# from utils.chatbot import load_chatbot

# st.set_page_config(page_title="Sea Buckthorn Healthcare Chatbot")

# st.title("🌿 Sea Buckthorn Healthcare Chatbot")
# st.write("Ask questions about Sea Buckthorn health benefits.")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# @st.cache_resource
# def get_chain():
#     return load_chatbot()

# qa_chain = get_chain()

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# prompt = st.chat_input("Ask your question about Sea Buckthorn...")

# if prompt:

#     st.session_state.messages.append({"role": "user", "content": prompt})

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):

#             # ✅ chain now returns plain string directly
#             answer = qa_chain.invoke(prompt)
#             st.markdown(answer)

#     st.session_state.messages.append({"role": "assistant", "content": answer})

# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# load_dotenv()
# print(os.getenv("GOOGLE_API_KEY"))

# st.set_page_config(page_title="Sea Buckthorn Healthcare Chatbot")
# st.title("🌿 Sea Buckthorn Healthcare Chatbot")
# st.write("Ask questions about Sea Buckthorn health benefits.")

# # ---- Load everything directly here (no cache, no import from utils) ----

# # @st.cache_resource(show_spinner="Loading Sea Buckthorn knowledge base...")
# def get_chain():
#     print("Before embeddings")

#     embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
#     print("Before FAISS load")
#     vectorstore = FAISS.load_local(
#         "vectorstore",
#         embeddings,
#         allow_dangerous_deserialization=True
#     )
#     print("FAISS loaded")

#     retriever = vectorstore.as_retriever()
#     print("Gemini loading...")

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         temperature=0.3
#     )

#     prompt = ChatPromptTemplate.from_template("""
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
#         | prompt
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

# prompt = st.chat_input("Ask your question about Sea Buckthorn...")

# if prompt:

#     st.session_state.messages.append({"role": "user", "content": prompt})

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             try:
#                 answer = qa_chain.invoke(prompt)
#                 st.markdown(answer)
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")
#                 print(e)

#     st.session_state.messages.append({"role": "assistant", "content": answer})

# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# load_dotenv()

# st.set_page_config(page_title="Sea Buckthorn Healthcare Chatbot")
# st.title("🌿 Sea Buckthorn Healthcare Chatbot")
# st.write("Ask questions about Sea Buckthorn health benefits.")

# # FIX 1: Uncomment @st.cache_resource so this only runs ONCE
# @st.cache_resource(show_spinner="Loading Sea Buckthorn knowledge base...")
# def get_chain():
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
#     vectorstore = FAISS.load_local(
#         "vectorstore",
#         embeddings,
#         allow_dangerous_deserialization=True
#     )
#     retriever = vectorstore.as_retriever()

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         temperature=0.3
#     )

#     # FIX 2: Renamed variable from 'prompt' to 'chat_prompt'
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

# user_input = st.chat_input("Ask your question about Sea Buckthorn...")  # FIX 2: renamed

# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     with st.chat_message("user"):
#         st.markdown(user_input)

#     # FIX 3: Define answer before try block to avoid UnboundLocalError
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

load_dotenv()

st.set_page_config(page_title="Sea Buckthorn Healthcare Chatbot")
st.title("🌿 Sea Buckthorn Healthcare Chatbot")
st.write("Ask questions about Sea Buckthorn health benefits.")


@st.cache_resource(show_spinner="Loading Sea Buckthorn knowledge base...")
def get_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # FIX: renamed to 'chat_prompt' to avoid conflict with st.chat_input variable
    chat_prompt = ChatPromptTemplate.from_template("""
You are a healthcare assistant specializing in Sea Buckthorn.
Use the context below to answer clearly and simply.
If the answer is not in the context, say: 'I do not have enough information on that.'

Context:
{context}

Question:
{question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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


# ---- Chat UI ----

if "messages" not in st.session_state:
    st.session_state.messages = []

qa_chain = get_chain()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# FIX: renamed from 'prompt' to 'user_input' to avoid variable conflict
user_input = st.chat_input("Ask your question about Sea Buckthorn...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # FIX: initialize answer before try block to avoid UnboundLocalError
    answer = "Sorry, I encountered an error. Please try again."

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = qa_chain.invoke(user_input)
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                print(e)

    st.session_state.messages.append({"role": "assistant", "content": answer})