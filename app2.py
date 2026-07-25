import os
import tempfile

import streamlit as st

from dotenv import load_dotenv

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.prompts import (
    ChatPromptTemplate
)


# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="RAG QnA",
    page_icon="🤖",
    layout="wide"
)


st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


st.markdown(
    """
    <style>

    /* =================================
       MAIN APP: BLACK + RED GRADIENT
    ================================= */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(180, 0, 0, 0.35),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #050505 0%,
                #120000 45%,
                #000000 100%
            );

        color: #FFFFFF;
    }


    /* =================================
       SIDEBAR
    ================================= */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #180000 0%,
                #080808 55%,
                #000000 100%
            ) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }


    /* =================================
       TOP HEADER BAR
    ================================= */

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0.85) !important;
    }


    /* =================================
       BOTTOM AREA
    ================================= */

    [data-testid="stBottom"] {
        background: #050505 !important;
    }


    /* =================================
       FOOTER
    ================================= */

    footer {
        background: #050505 !important;
    }


    /* =================================
       TEXT COLORS
    ================================= */

    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }

    p, label, span {
        color: #F5F5F5 !important;
    }


    /* =================================
       CHAT MESSAGE TEXT
    ================================= */

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 10px;
    }


    /* =================================
       CHAT INPUT
    ================================= */

    [data-testid="stChatInput"] {
        background: #111111 !important;
        border: 1px solid #8B0000 !important;
        border-radius: 12px;
    }


    /* Chat input text */

    [data-testid="stChatInput"] textarea {
        color: #FFFFFF !important;
    }


    /* =================================
       BUTTONS
    ================================= */

    .stButton > button {
        background: linear-gradient(
            135deg,
            #8B0000,
            #D00000
        );

        color: #FFFFFF !important;

        border: none;

        border-radius: 8px;

        font-weight: bold;
    }


    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #D00000,
            #FF1A1A
        );

        color: #FFFFFF !important;
    }


    /* =================================
       FILE UPLOADER
    ================================= */

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);

        border: 1px solid #8B0000;

        border-radius: 10px;

        padding: 10px;
    }


    /* =================================
       SELECT / RADIO TEXT
    ================================= */

    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }


    /* =================================
       EXPANDER
    ================================= */

    [data-testid="stExpander"] {
        background: rgba(100, 0, 0, 0.15);

        border: 1px solid #660000;

        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# =====================================================
# Title
# =====================================================

st.title("🤖RAG QnA")


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.header("⚙️ Select Document Mode")

    mode = st.radio(
        "Choose an option:",
        [
            "📚 Transformer",
            "📤 Upload New PDF"
        ]
    )


# =====================================================
# Initialize Session State
# =====================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = None


if "current_mode" not in st.session_state:

    st.session_state.current_mode = None


if "current_file" not in st.session_state:

    st.session_state.current_file = None


# =====================================================
# Load Embedding Model
# =====================================================

@st.cache_resource
def load_embedding_model():

    return MistralAIEmbeddings(
        model="mistral-embed"
    )


# =====================================================
# Load LLM
# =====================================================

@st.cache_resource
def load_llm():

    return ChatMistralAI(
        model="mistral-small-2506"
    )


embedding_model = load_embedding_model()

llm = load_llm()


# =====================================================
# Text Splitter
# =====================================================

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=50

)


# =====================================================
# Prompt
# =====================================================

prompt = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            """
            You are a helpful document question-answering assistant.

            Answer the question using ONLY the provided context.

            If the answer is not available in the context, say:

            Sorry, I don't know based on the provided document.

            Do not use outside knowledge.
            """

        ),

        (

            "human",

            """
            Context:

            {context}


            Question:

            {question}
            """

        )

    ]

)


# =====================================================
# DEFAULT DOCUMENT MODE
# =====================================================

if mode == "📚 Transformer":

    st.info(
        "📚 You are chatting with the default Transformer document."
    )


    # If mode changes, clear chat
    if st.session_state.current_mode != mode:

        st.session_state.messages = []

        st.session_state.current_mode = mode


    # Load existing Chroma database

    @st.cache_resource
    def load_default_vectorstore():

        vectorstore = Chroma(

            persist_directory="Chroma_vector_store",

            embedding_function=embedding_model

        )

        return vectorstore


    try:

        vectorstore = load_default_vectorstore()

        st.session_state.vectorstore = vectorstore

        st.success(
            "✅ Default document is ready!"
        )

    except Exception as e:

        st.error(
            "Default vector database not found."
        )

        st.info(
            "Please create the Chroma database first."
        )

        st.stop()


# =====================================================
# UPLOAD NEW PDF MODE
# =====================================================

elif mode == "📤 Upload New PDF":

    st.info(
        "📤 Upload a PDF and chat with that document."
    )


    uploaded_file = st.file_uploader(

        "Upload your PDF",

        type=["pdf"]

    )


    if uploaded_file is not None:


        # Detect new file

        if (

            st.session_state.current_file

            != uploaded_file.name

        ):

            with st.spinner(
                "Processing your PDF..."
            ):


                # Save uploaded file temporarily

                with tempfile.NamedTemporaryFile(

                    delete=False,

                    suffix=".pdf"

                ) as temp_file:


                    temp_file.write(

                        uploaded_file.getbuffer()

                    )

                    temp_file_path = temp_file.name


                # Load PDF

                loader = PyPDFLoader(

                    temp_file_path

                )

                documents = loader.load()


                # Split documents

                chunks = splitter.split_documents(

                    documents

                )


                # Create new vector database

                vectorstore = Chroma.from_documents(

                    documents=chunks,

                    embedding=embedding_model

                )


                # Store in session state

                st.session_state.vectorstore = vectorstore


                st.session_state.current_file = (

                    uploaded_file.name

                )


                # Clear old chat

                st.session_state.messages = []


                st.success(

                    f"✅ {uploaded_file.name} is ready!"

                )


                st.info(

                    f"📄 Created {len(chunks)} chunks."

                )


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(

        message["role"]

    ):

        st.markdown(

            message["content"]

        )


# =====================================================
# CHAT INPUT
# =====================================================

user_query = st.chat_input(

    "Ask a question about the document..."

)


# =====================================================
# QUESTION ANSWERING
# =====================================================

if user_query:


    if st.session_state.vectorstore is None:

        st.warning(

            "Please select or upload a document first."

        )

        st.stop()


    # Display user message

    with st.chat_message("user"):

        st.markdown(

            user_query

        )


    st.session_state.messages.append(

        {

            "role": "user",

            "content": user_query

        }

    )


    # Create retriever

    retriever = (

        st.session_state.vectorstore

        .as_retriever(

            search_type="mmr",

            search_kwargs={

                "k": 3,

                "fetch_k": 10,

                "lambda_mult": 0.5

            }

        )

    )


    # Retrieve relevant documents

    with st.spinner(

        "Searching the document..."

    ):

        docs = retriever.invoke(

            user_query

        )


    # Create context

    context = "\n\n".join(

        [

            doc.page_content

            for doc in docs

        ]

    )


    # Create prompt

    final_prompt = prompt.invoke(

        {

            "context": context,

            "question": user_query

        }

    )


    # Generate response

    with st.spinner(

        "Generating answer..."

    ):

        response = llm.invoke(

            final_prompt

        )


    answer = response.content


    # Display answer

    with st.chat_message(

        "assistant"

    ):

        st.markdown(

            answer

        )


    # Save assistant message

    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": answer

        }

    )


    # Show context

    with st.expander(

        "📚 View Retrieved Context"

    ):

        for i, doc in enumerate(docs):

            st.markdown(

                f"### Chunk {i + 1}"

            )

            st.write(

                doc.page_content

            )

            st.divider()