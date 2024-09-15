import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Dict, Any
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


def vectorstore():
    print("Ingestion...")

    loader = TextLoader("./mbcet_website_data.txt")
    document = loader.load()

    print("splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="llama3")

    print("ingesting..")

    # PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    library = FAISS.from_documents(texts, embeddings)

    print("finish")

    return library
