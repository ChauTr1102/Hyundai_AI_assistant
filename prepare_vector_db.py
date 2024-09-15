from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from sentence_transformers import SentenceTransformer
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from io import BytesIO
from gpt4all import GPT4All
pdf_data_path = r"data"
vector_db_path = "vectorstores/db_faiss"

# Ham 1. Tao ra vector DB tu 1 doan text


def create_db_from_files(pdf_data_path):
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embedding_model = OpenAIEmbeddings(openai_api_key= "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")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db



# Streamlit interface

create_db_from_files(pdf_data_path)
