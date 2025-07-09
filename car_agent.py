import os
from getpass import getpass
from rag_agent import CarManualRAGAgent

if __name__ == "__main__":
    api_key = getpass("Enter your OpenAI API Key: ")
    agent = CarManualRAGAgent(api_key=api_key, pdf_path="./car_manual.pdf")
    print("\n--- Car Manual RAG Agent is Ready! ---")
    print("Ask questions about your car manual. Type 'exit' to quit.")
    while True:
        query = input("\nYour Question: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = agent.ask(query)
        print("Answer:", answer)