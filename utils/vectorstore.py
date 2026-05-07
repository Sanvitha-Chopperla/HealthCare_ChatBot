import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from utils.loader import load_documents
load_dotenv()

# def create_vectorstore():

#     documents = load_documents()

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50
#     )

#     docs = text_splitter.split_documents(documents)

#     embeddings=GoogleGenerativeAIEmbeddings(
#     model="text-embedding-004",
#     google_api_key=os.getenv("GOOGLE_API_KEY")
#     )
#     vectorstore = FAISS.from_documents(
#         docs,
#         embeddings
#     )

#     vectorstore.save_local("vectorstore")

#     print("Vector Database Created Successfully")
# def create_vectorstore():

#     documents = load_documents()

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50
#     )

#     docs = text_splitter.split_documents(documents)

#     embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     vectorstore = FAISS.from_documents(
#         docs,
#         embeddings
#     )

#     vectorstore.save_local("vectorstore")

#     print("Vector Database Created Successfully")

# if __name__ == "__main__":
#     create_vectorstore()

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.loader import load_documents

load_dotenv()


def create_vectorstore():

    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    vectorstore.save_local("vectorstore")

    print("Vector Database Created Successfully")


if __name__ == "__main__":
    create_vectorstore()

