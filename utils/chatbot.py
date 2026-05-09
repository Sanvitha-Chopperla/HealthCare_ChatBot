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

#     retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 6}
#     )

#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         temperature=0.3,
#         api_key=os.getenv("GROQ_API_KEY")
#     )

#     chat_prompt = ChatPromptTemplate.from_template("""
# You are a professional healthcare assistant specializing in Sea Buckthorn (Hippophae rhamnoides).

# Your job is to answer the user's question using ONLY the context provided below.

# Instructions:
# - Give a clear, structured, and detailed answer
# - Use bullet points or numbered lists where helpful
# - If specific data (like vitamins, nutrients, benefits) is available in the context, include it
# - Do NOT make up information that is not in the context
# - If the answer is not found in the context, say: "I do not have enough information on that topic."

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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

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

    # ── Step 1: Query rewriter ──────────────────────────────────────────────
    # Converts casual/vague questions into proper search queries
    # so the retriever finds the RIGHT chunks from your documents.

    rewrite_prompt = ChatPromptTemplate.from_template("""
You are a query rewriting assistant for a Sea Buckthorn healthcare chatbot.

Your job is to convert the user's casual or vague question into a 
clear, specific search query about Sea Buckthorn.

Rules:
- Always keep the topic related to Sea Buckthorn
- Make it a proper factual question
- Keep it short (one sentence)
- Do NOT answer the question — only rewrite it

Examples:
User: "tell me in short about it"         → "What is Sea Buckthorn? Give a brief overview."
User: "make it simple"                    → "Explain Sea Buckthorn simply for a beginner."
User: "what about vitamins"               → "What vitamins are found in Sea Buckthorn?"
User: "is it good for me"                 → "What are the health benefits of Sea Buckthorn?"
User: "side effects?"                     → "What are the side effects of Sea Buckthorn?"
User: "how to use it"                     → "How is Sea Buckthorn used or consumed?"
User: "what is sea buckthorn"             → "What is Sea Buckthorn and what are its properties?"

Now rewrite this user question:
"{question}"

Rewritten query:
""")

    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # ── Step 2: Answer prompt ───────────────────────────────────────────────
    answer_prompt = ChatPromptTemplate.from_template("""
You are a friendly and professional healthcare assistant specializing in Sea Buckthorn.

Answer the user's question using ONLY the context provided below.

Instructions:
- Match your answer style to how the user asked:
  * If they asked for "short" or "simple" → give a brief 3-5 line answer
  * If they asked for "detailed" or "explain" → give a full structured answer with bullet points
  * If they asked a specific question → answer it directly and clearly
- Use simple, easy-to-understand language
- Include specific facts (vitamins, nutrients, benefits) if available in the context
- Do NOT make up information not in the context
- If the context does not contain the answer, say: "I don't have that specific information in my documents."

Context:
{context}

User's Original Question:
{question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ── Full RAG Chain with Query Rewriting ─────────────────────────────────
    def rag_with_rewrite(question):
        # Step 1: Rewrite the query for better retrieval
        rewritten = rewrite_chain.invoke({"question": question})
        print(f"\n🔄 Original : {question}")
        print(f"✏️  Rewritten: {rewritten}\n")

        # Step 2: Retrieve relevant docs using rewritten query
        docs = retriever.invoke(rewritten)
        context = format_docs(docs)

        # Step 3: Answer using original question + retrieved context
        answer = answer_prompt | llm | StrOutputParser()
        return answer.invoke({"context": context, "question": question})

    return RunnableLambda(rag_with_rewrite)