import os
import shutil

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain.chains import RetrievalQA

load_dotenv()


class RAGEngine:

    def __init__(self):

        # Embedding model
        self.embeddings = OllamaEmbeddings(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("EMBEDDING_MODEL")
        )

        # LLM
        self.llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("LLM_MODEL")
        )

        # Vector database
        self.db = Chroma(
            persist_directory=os.getenv("CHROMA_DB_PATH"),
            embedding_function=self.embeddings
        )

    def ingest_document(self, file_path, filename):

        print(f"\nLoading: {filename}")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
        )

        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["source"] = filename

        self.db.add_documents(chunks)

        return {
            "chunks": len(chunks),
            "filename": filename
        }

    def query(self, question):

        print("Searching documents...")

        retriever = self.db.as_retriever(
            search_kwargs={
                "k": int(os.getenv("TOP_K_RESULTS", 3))
            }
        )

        print("Creating QA chain...")

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )

        print("Generating answer...")

        result = qa.invoke(
            {"query": question}
        )

        sources = []

        for doc in result["source_documents"]:
            src = doc.metadata.get("source", "Unknown")

            if src not in sources:
                sources.append(src)

        return {
            "answer": result["result"],
            "sources": sources
        }

    def reset(self):

        db_path = os.getenv("CHROMA_DB_PATH")

        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        self.__init__()

        return {
            "status": "Database cleared"
        }