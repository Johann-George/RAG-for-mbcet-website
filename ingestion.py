from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

load_dotenv()

if __name__ == "__main__":
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

    library.save_local("faiss_index")

    print("finish")
