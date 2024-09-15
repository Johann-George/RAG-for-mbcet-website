import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm(query: str, vectorstore: any, chat_history: List[Dict[str, Any]] = []):
    embeddings = OllamaEmbeddings(model='llama3')
    llm = ChatOllama(model='llama3')

    # vectorStore = PineconeVectorStore(
    #     index_name=os.environ['INDEX_NAME'], embedding=embeddings
    # )

    # retriever = vectorstore.as_retriever()

    retriever = vectorstore.as_retriever()

    template = """
    Act as a receptionist for Mar Baselios College of engineering and technology and use the following pieces of context
    to answer the questions at the end. If you don't know the answer just say that you don't know,
    don't try to make up an answer. Always say "thanks for asking!" at the end of the answer.
    
   {context}
    
    Question: {question}
    
    Helpful answer:"""

    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    result = rag_chain.invoke({
        "chat_history": chat_history,
        "context": history_aware_retriever,
        "question": query,
        "input": query
    })
    return result


if __name__ == "__main__":
    res = run_llm(query="Who is E-yantra team?")
    print(res.content)
