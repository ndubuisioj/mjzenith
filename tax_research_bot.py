import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Ensure API key is configured
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set your OPENAI_API_KEY inside GitHub Codespaces Secrets.")

def setup_tax_knowledge_base():
    print("Parsing tax documents...")
    # Creates directory if it doesn't exist; drop your IRS PDFs or tax texts here
    os.makedirs("knowledge_base", exist_ok=True)
    
    loader = PyPDFDirectoryLoader("knowledge_base/")
    docs = loader.load()
    
    if not docs:
        print("⚠️ 'knowledge_base' folder is empty. Please add tax PDFs or documents.")
        return None

    # Split documents into readable fragments for the AI
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embed and store texts in a local vector database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def query_tax_ai(retriever, query_text):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # System prompt ensuring strict adherence to provided source tax documentation
    system_prompt = (
        "You are an expert tax research assistant. Use the following pieces of retrieved tax context "
        "to answer the question. If you do not know the answer, say that you cannot find it in the "
        "provided source material. Do not make up facts.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print(f"\nEvaluating Query: '{query_text}'")
    response = rag_chain.invoke({"input": query_text})
    
    print("\n=== AI Tax Research Answer ===")
    print(response["answer"])
    print("\n=== Sources Referenced ===")
    for doc in response["context"]:
        print(f"- {doc.metadata.get('source', 'Unknown Source')} (Page {doc.metadata.get('page', 'N/A')})")

if __name__ == "__main__":
    retriever = setup_tax_knowledge_base()
    if retriever:
        # Example research query
        query = "What are the rules regarding capital gains tax exclusions on primary residence sales?"
        query_tax_ai(retriever, query)
