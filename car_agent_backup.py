import os
from getpass import getpass

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# --- 1. SET UP YOUR OPENAI API KEY ---
# This line is for running in environments like Google Colab or a terminal.
# If you are using a .env file, you would use: from dotenv import load_dotenv; load_dotenv()
os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API Key: ")

# --- 2. LOAD AND PROCESS THE DOCUMENT ---

# Load the PDF document
pdf_path = "./car_manual.pdf"  # Make sure this path is correct
loader = PyPDFLoader(pdf_path)
documents = loader.load()

if not documents:
    print("Could not load documents from PDF. Please check the file path and content.")
    exit()

print(f"Loaded {len(documents)} pages from the PDF.")

# Split the document into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # The max number of characters in a chunk
    chunk_overlap=100,  # The number of characters to overlap between chunks
    length_function=len
)
docs = text_splitter.split_documents(documents)

print(f"Split the document into {len(docs)} chunks.")

# --- 3. CREATE EMBEDDINGS AND VECTOR STORE ---

# Create embeddings for the document chunks
# This will convert the text chunks into numerical vectors
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store from the document chunks and their embeddings
# This allows for efficient similarity searching
print("Creating vector store... This may take a moment.")
try:
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("Vector store created successfully.")
except Exception as e:
    print(f"An error occurred while creating the vector store: {e}")
    exit()

# --- 4. BUILD THE CONVERSATIONAL RAG CHAIN ---

# Initialize the LLM we'll use for answering questions
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create the conversational RAG chain.
# This chain will handle retrieving relevant documents, and then generating an answer,
# while also managing the conversation history.
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True  # Optionally return the source documents
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

    # The chain takes the user's question and the chat history as input
    result = qa_chain({"question": query, "chat_history": chat_history})

    # Print the answer
    print("Answer:", result["answer"])

    # Optionally, print the source documents used for the answer
    # for i, doc in enumerate(result['source_documents']):
    #     print(f"\n--- Source Document {i+1} (from page {doc.metadata['page']}) ---")
    #     print(doc.page_content)

    # Update the chat history with the new question and answer
    chat_history.append((query, result["answer"]))