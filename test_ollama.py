from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="mistral"
)

response = llm.invoke("What is the capital of India?")

print(response.content)