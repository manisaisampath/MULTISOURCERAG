from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
    print("Initializing database and vectorstore...")
    database()
    print("Resetting vectorstore...")
    vectorstore.delete_collection()
    vectorstore = None
    database()


    print("Loading documents from URLs...")
    loader = UnstructuredURLLoader(
        urls=urls,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
    )

    doc = loader.load()

    print("Splitting documents...")
    chunk = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200
    )
 
    data = chunk.split_documents(doc)

    ("Adding documents to vectorstore...")
    uuids = [str(uuid4()) for _ in range(len(data))]
    vectorstore.add_documents(data, ids=uuids)



def generate(query):
    if not vectorstore:
        raise Exception("Vectorstore not initialized.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Combine document contents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 3: Extract unique source URLs
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


if __name__ == "__main__":

    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html",
        "https://www.investopedia.com/terms/r/realestate.asp"
    ]


    processurl(urls)

    answer, sources = generate(
    "how many times FED lowered the interest rate in 2024"
)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:\n")
    for src in sources:
        print(src)

