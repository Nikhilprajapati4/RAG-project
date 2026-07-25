from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

FILE_PATH = r"D:\Projects\RAG Chatbot\Data\transformer.pdf"

loader = PyPDFLoader(FILE_PATH)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model ,
                                     persist_directory="Chroma_vector_store")