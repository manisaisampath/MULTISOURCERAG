from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

llm = None
vectorstore = None

persist_directory = Path(__file__).parent / "resources/vectorstore"


def database():
    global llm, vectorstore

    if vectorstore is None:
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma(
            collection_name="real_estate",
            embedding_function=embedding,
            persist_directory=str(persist_directory)
        )

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )


def processurl(urls):
    global vectorstore

    print("Initializing database...")
    database()

    print("Loading documents from URLs...")

    loader = UnstructuredURLLoader(
        urls=urls,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    docs = loader.load()
    print("Documents loaded:", len(docs))

    if not docs:
        raise ValueError("No documents loaded from URLs.")

    print("Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)
    print("Chunks created:", len(chunks))

    if not chunks:
        raise ValueError("No chunks created from documents.")

    print("Adding documents to vectorstore...")

    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vectorstore.add_documents(chunks, ids=uuids)

    print("Documents successfully added.")


def generate(query):
    global vectorstore

    if vectorstore is None:
        raise Exception("Vectorstore not initialized. Please process URLs first.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found.", []

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the context below.

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": query
    })

    return answer, sources
