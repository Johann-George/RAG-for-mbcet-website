import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Ingestion...")


    # Use a custom loader that specifies UTF-8 encoding
    # class CustomTextLoader(TextLoader):
    #     def load(self):
    #         with open(self.file_path, 'r', encoding='utf-8') as file:
    #             return file.read()


    # Create an instance of the custom loader
    loader = TextLoader("C:/Users/johan/PycharmProjects/VectorDB_Intro/mbcet_website_data.txt")
    document = loader.load()

    print("splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="llama3")

    print("ingesting..")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])

    print("finish")
