import os
from getpass import getpass

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# V-- THIS IS THE CORRECTED IMPORT --V
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# --- 1. SET UP YOUR OPENAI API KEY ---
os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API Key: ")

# --- 2. LOAD AND PROCESS THE DOCUMENT ---
pdf_path = "./car_manual.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

if not documents:
    print("Could not load documents from PDF. Please check the file path and content.")
    exit()

print(f"Loaded {len(documents)} pages from the PDF.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)
docs = text_splitter.split_documents(documents)

print(f"Split the document into {len(docs)} chunks.")

# --- 3. CREATE EMBEDDINGS AND IN-MEMORY VECTOR STORE ---
embeddings = OpenAIEmbeddings()

print("Creating in-memory vector store... This may take a moment.")
try:
    # This line now works because of the corrected import
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    print("Vector store created successfully.")
except Exception as e:
    print(f"An error occurred while creating the vector store: {e}")
    exit()

# --- 4. BUILD THE CONVERSATIONAL RAG CHAIN ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

print("\n--- Car Manual RAG Agent is Ready! ---")
print("Ask questions about your car manual. Type 'exit' to quit.")

# --- 5. INTERACTIVE CHAT LOOP ---
chat_history = []

while True:
    query = input("\nYour Question: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    result = qa_chain({"question": query, "chat_history": chat_history})

    print("Answer:", result["answer"])

    chat_history.append((query, result["answer"]))