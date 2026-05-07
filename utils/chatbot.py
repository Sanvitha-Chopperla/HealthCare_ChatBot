# from dotenv import load_dotenv

# from langchain_openai import ChatOpenAI

# from langchain_community.vectorstores import FAISS

# from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain.chains import RetrievalQA

# from utils.prompt import PROMPT

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

#     retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 3}
#     )

#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0.3
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={
#             "prompt": PROMPT
#         }
#     )

#     return qa_chain

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain

# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
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

#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0.3
#     )

#     prompt = ChatPromptTemplate.from_template("""
# You are a healthcare assistant for Sea Buckthorn.

# Use the context to answer clearly and simply.

# Context:
# {context}
# Question:
# {input}
# """)

#     doc_chain = create_stuff_documents_chain(llm, prompt)

#     chain = create_retrieval_chain(retriever, doc_chain)

#     return chainimport os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

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

#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0.3
#     )

#     prompt = ChatPromptTemplate.from_template("""
# You are a healthcare assistant specializing in Sea Buckthorn.
# Use the context below to answer clearly and simply.
# If the answer is not in the context, say: 'I do not have enough information on that.'

# Context:
# {context}

# Question:
# {input}

# Answer:
# """)

#     doc_chain = create_stuff_documents_chain(llm, prompt)
#     chain = create_retrieval_chain(retriever, doc_chain)

#     return chain
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

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
# {input}

# Answer:
# """)

#     doc_chain = create_stuff_documents_chain(llm, prompt)
#     chain = create_retrieval_chain(retriever, doc_chain)

#     return chain

# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
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

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
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

#     # Using pure LCEL — no langchain.chains imports at all
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

    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

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