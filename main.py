import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving")

    embeddings = OllamaEmbeddings(model='llama3')
    llm = ChatOllama(model='llama3')

    query = "What are the various awards won by the college?"
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    vectorStore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrieval_chain = create_retrieval_chain(
    #     retriever=vectorStore.as_retriever(), combine_docs_chain=combine_docs_chain
    # )
    #
    # result = retrieval_chain.invoke(input={"input": query})
    # print(result)

    template = """
    Act as a receptionist for Mar Baselios College of engineering and technology and use the following pieces of context
    to answer the questions at the end. If you don't know the answer just say that you don't know,
    don't try to make up an answer. Always say "thanks for asking!" at the end of the answer.
    
    {context}
    
    Question: {question}
    
    Helpful answer:"""

    custom_rag_propmt = PromptTemplate.from_template(template)

    rag_chain = (
            {"context": vectorStore.as_retriever() | format_docs, "question": RunnablePassthrough()}
            | custom_rag_propmt
            | llm
    )

    res = rag_chain.invoke(query)
    print(res.content)
