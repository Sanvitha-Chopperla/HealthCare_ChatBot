import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)


def load_documents():
    """
    Loads all .txt, .pdf, and .docx files from the data/ folder.
    Just drop any document or book into data/ and it will be included automatically.
    """

    all_documents = []
    data_folder = "data"

    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"'{data_folder}' folder not found. Please create it and add your documents."
        )

    files_found = os.listdir(data_folder)
    if not files_found:
        raise ValueError(
            f"No files found in '{data_folder}'. Please add your documents/books."
        )

    print(f"\n📂 Loading files from '{data_folder}' folder...")

    for filename in files_found:
        filepath = os.path.join(data_folder, filename)

        # Skip subfolders
        if os.path.isdir(filepath):
            continue

        try:
            if filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
                all_documents.extend(docs)
                print(f"  ✅ Loaded TXT : {filename} ({len(docs)} document(s))")

            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                all_documents.extend(docs)
                print(f"  ✅ Loaded PDF : {filename} ({len(docs)} page(s))")

            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
                docs = loader.load()
                all_documents.extend(docs)
                print(f"  ✅ Loaded DOCX: {filename} ({len(docs)} document(s))")

            else:
                print(f"  ⚠️  Skipped    : {filename} (unsupported format)")

        except Exception as e:
            print(f"  ❌ Failed to load {filename}: {str(e)}")

    print(f"\n✅ Total documents loaded: {len(all_documents)}\n")

    if not all_documents:
        raise ValueError("No documents were loaded. Check your files in the data/ folder.")

    return all_documents