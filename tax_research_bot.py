import os

KNOWLEDGE_BASE_DIR = "knowledge_base"
DEFAULT_QUERY = (
    "What are the rules regarding capital gains tax exclusions on primary "
    "residence sales?"
)
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"


def require_langchain_dependencies():
    """Load LangChain packages only when the bot actually needs them."""
    try:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:  # pragma: no cover - environment-specific dependency issue
        raise RuntimeError(
            "LangChain dependencies are not available in this environment. "
            "Install the project requirements and use a Python version compatible "
            "with the LangChain version in requirements.txt."
        ) from exc

    return {
        "create_retrieval_chain": create_retrieval_chain,
        "create_stuff_documents_chain": create_stuff_documents_chain,
        "PyPDFDirectoryLoader": PyPDFDirectoryLoader,
        "Chroma": Chroma,
        "ChatPromptTemplate": ChatPromptTemplate,
        "ChatOpenAI": ChatOpenAI,
        "OpenAIEmbeddings": OpenAIEmbeddings,
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
    }


def require_openai_api_key():
    """Ensure the OpenAI API key is configured before the app runs."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "Please set your OPENAI_API_KEY inside GitHub Codespaces Secrets."
        )


def format_sources(documents):
    """Return a readable list of citation strings for retrieved documents."""
    formatted_sources = []
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source", "Unknown Source")
        page = metadata.get("page", "N/A")
        formatted_sources.append(f"- {source} (Page {page})")
    return formatted_sources


def setup_tax_knowledge_base():
    """Create the vector database used to answer tax questions from local documents."""
    deps = require_langchain_dependencies()
    print("Parsing tax documents...")
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

    loader = deps["PyPDFDirectoryLoader"](KNOWLEDGE_BASE_DIR)
    docs = loader.load()

    if not docs:
        print("⚠️ 'knowledge_base' folder is empty. Please add tax PDFs or documents.")
        return None

    text_splitter = deps["RecursiveCharacterTextSplitter"](
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(docs)

    embeddings = deps["OpenAIEmbeddings"](model=EMBEDDING_MODEL)
    vectorstore = deps["Chroma"].from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def query_tax_ai(retriever, query_text):
    """Run a retrieval-augmented generation search for a tax question."""
    deps = require_langchain_dependencies()
    llm = deps["ChatOpenAI"](model=CHAT_MODEL, temperature=0)

    system_prompt = (
        "You are an expert tax research assistant. Use the following pieces of "
        "retrieved tax context to answer the question. If you do not know the "
        "answer, say that you cannot find it in the provided source material. "
        "Do not make up facts.\n\nContext:\n{context}"
    )

    prompt = deps["ChatPromptTemplate"].from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = deps["create_stuff_documents_chain"](llm, prompt)
    rag_chain = deps["create_retrieval_chain"](retriever, question_answer_chain)

    print(f"\nEvaluating Query: '{query_text}'")
    response = rag_chain.invoke({"input": query_text})

    print("\n=== AI Tax Research Answer ===")
    print(response["answer"])
    print("\n=== Sources Referenced ===")
    for source in format_sources(response.get("context", [])):
        print(source)

    return response


def main():
    """Entry point for the tax research bot."""
    require_openai_api_key()
    retriever = setup_tax_knowledge_base()
    if retriever:
        query_tax_ai(retriever, DEFAULT_QUERY)


if __name__ == "__main__":
    main()
