# from langchain_community.document_loaders import TextLoader

# def load_documents():

#     loader = TextLoader("data/sea_buckthorn.txt")

#     documents = loader.load()

#     return documents

from langchain_community.document_loaders import TextLoader


def load_documents():
    loader = TextLoader("data/sea_buckthorn.txt")
    documents = loader.load()
    return documents