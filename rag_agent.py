import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

class CarManualRAGAgent:
    def __init__(self, api_key: str, pdf_path: str = "./car_manual.pdf"):
        os.environ["OPENAI_API_KEY"] = api_key
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            raise ValueError("Could not load documents from PDF. Please check the file path and content.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        self.chat_history = []

    def ask(self, question: str) -> str:
        result = self.qa_chain({"question": question, "chat_history": self.chat_history})
        self.chat_history.append((question, result["answer"]))
        return result["answer"] 