from rag_engine import RAGEngine

print("Creating engine...")
rag = RAGEngine()

print("Asking question...")

response = rag.query(
    "What is this document about?"
)

print("Answer:")
print(response)