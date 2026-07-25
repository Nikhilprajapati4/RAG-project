#_____from pdf___________

from langchain_community.document_loaders import PyPDFLoader

FILE_PATH = r"D:\Projects\RAG Chatbot\Data\transformer.pdf"

loader = PyPDFLoader(FILE_PATH)

documents = loader.load()


# ____from text_____________

from langchain_community.document_loaders import TextLoader

FILE_PATH = r"D:\Projects\RAG Chatbot\Data\transformer.pdf"

loader = TextLoader(FILE_PATH)

documents = loader.load()


#___from webpage______

from langchain_community.document_loaders import WebBaseLoader

URL = "website"

loader = WebBaseLoader(URL)

documents = loader.load()