from rag_engine import RAGEngine

rag = RAGEngine()

print("EU Regulatory Chatbot Ready")
print("Type exit to quit")

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    answer = rag.query(question)

    print("\nBot:", answer)