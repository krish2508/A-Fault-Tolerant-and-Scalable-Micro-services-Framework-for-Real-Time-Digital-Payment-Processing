from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from pdf_loader import documents

import os 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(documents)

# print(split_documents)
# print("Total chunks:", len(split_documents))
# breakpoint()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)
# breakpoint()
vector_store = FAISS.from_documents(split_documents, embeddings)

