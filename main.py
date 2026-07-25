from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv()


prompt = ChatPromptTemplate.from_messages([("system" , """you are a good AI Use only provided context to answer me.
                                              if you  not find in context than say Sorry I Dont know !!! """),
                                              ("human" , """ context : {context} 
                                              
                                              Question : {question}""")])

embedding_model = MistralAIEmbeddings(model="mistral-embed")
vectorstore = Chroma(persist_directory="Chroma_vector_store" 
                     , embedding_function=embedding_model)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm= ChatMistralAI(model = "mistral-small-2506")


print("__________________________________Rag is created___________________________________")
print("press 0 for exit")

while True :
    quary = input("Ask as want : ")
    if quary == "0" :
        break

    docs = retriever.invoke(quary)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context" : context , "question" : quary})

    response = llm.invoke(final_prompt)

    print(f"\n AI : {response.content}")