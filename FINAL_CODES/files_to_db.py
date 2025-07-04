import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# -------------- EMBEDDING MODEL ----------------
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs, 
    encode_kwargs=encode_kwargs 
)
# -----------------------------------------------

# -------------------- PATHS --------------------
current_dir = os.getcwd()
db_dir = os.path.join(current_dir, "db")
idx_name = "FAISS_metadata"
# -----------------------------------------------
        
def text_file_extract(file_name):
    
    file_path = os.path.join(current_dir, "uploads", file_name)

    loader = TextLoader(file_path, encoding="utf-8")
    data = loader.load()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    chunks = text_splitter.split_documents(data)

    for chunk in chunks:
        chunk.metadata = {'source': file_name}

    faissdb = FAISS.load_local(
        folder_path=db_dir,
        embeddings=embeddings,
        index_name=idx_name,
        allow_dangerous_deserialization=True,
    )

    faissdb.add_documents(chunks)

    faissdb.save_local(folder_path=db_dir, index_name=idx_name)

def pdf_file_extract(file_name):

    file_path = os.path.join(current_dir, "uploads", file_name)

    loader = PyMuPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10,
        length_function=len,
    )

    chunks = text_splitter.split_documents(data)

    for chunk in chunks:
        chunk.metadata = {'source': file_name}

    faissdb = FAISS.load_local(
        folder_path=db_dir,
        embeddings=embeddings,
        index_name=idx_name,
        allow_dangerous_deserialization=True,
    )

    faissdb.add_documents(chunks)

    faissdb.save_local(folder_path=db_dir, index_name=idx_name)