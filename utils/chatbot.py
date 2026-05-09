# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# load_dotenv()


# def load_chatbot():

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     vectorstore = FAISS.load_local(
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

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def load_chatbot():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    chat_prompt = ChatPromptTemplate.from_template("""
You are a professional healthcare assistant specializing in Sea Buckthorn (Hippophae rhamnoides).

Your job is to answer the user's question using ONLY the context provided below.

Instructions:
- Give a clear, structured, and detailed answer
- Use bullet points or numbered lists where helpful
- If specific data (like vitamins, nutrients, benefits) is available in the context, include it
- Do NOT make up information that is not in the context
- If the answer is not found in the context, say: "I do not have enough information on that topic."

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