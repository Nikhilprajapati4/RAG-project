from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

FILE_PATH = r"D:\Projects\RAG Chatbot\Data\transformer.pdf"

loader = PyPDFLoader(FILE_PATH)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(len(chunks))



