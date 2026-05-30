from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_ollama import OllamaEmbeddings, ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()


class RAGEngine:
    def __init__(self):

        # Set up the embedding model (converts text to vectors)
        self.embeddings = OllamaEmbeddings(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("EMBEDDING_MODEL")
        )

        # Set up the LLM (answers questions)
        self.llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("LLM_MODEL"),
            temperature=0
        )

        # Set up the text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
        )

        # Set up the vector database
        self.db = Chroma(
            persist_directory=os.getenv("CHROMA_DB_PATH"),
            embedding_function=self.embeddings
        )

    def ingest_document(self, path, filename):

        # Pick the right loader based on file extension
        ext = filename.rsplit(".", 1)[-1].lower()

        loaders = {
            "pdf": PyPDFLoader,
            "docx": Docx2txtLoader,
            "txt": TextLoader
        }

        loader = loaders.get(ext, TextLoader)(path)

        # Load and split the document
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)

        # Tag each chunk with the source filename
        for chunk in chunks:
            chunk.metadata["source"] = filename

        # Store the chunks in the vector database
        self.db.add_documents(chunks)

        return {
            "chunks": len(chunks),
            "filename": filename
        }

    def query(self, question):
print("Searching documents...")
        # Find the most relevant chunks
        retriever = self.db.as_retriever(
            search_kwargs={
                "k": int(os.getenv("TOP_K_RESULTS", 3))
            }
        )
print("Creating QA chain...")
        # Build the question-answering chain
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )

        # Get the answer
        result = qa.invoke({"query": question})

        sources = list({
            d.metadata.get("source", "")
            for d in result["source_documents"]
        })

        return {
            "response": result["result"],
            "sources": sources
        }

    def reset(self):
        import shutil

        db_path = os.getenv("CHROMA_DB_PATH")

        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        self.__init__()

        return {
            "status": "cleared"
        }