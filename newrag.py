import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(
    page_title="LLM-first + Casual Detection Chatbot",
    layout="centered"
)

st.title("Chatbot with RAG doc")


# -------------------------
# Document loading
# -------------------------
DOCS_PATH = Path("documents")

def load_documents():
    docs = []
    for file in DOCS_PATH.glob("*"):
        if file.is_file() and file.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        elif file.is_file() and file.suffix == ".txt":
            docs.extend(TextLoader(str(file)).load())
    return docs

# -------------------------
# Build RAG chain (cached)
# -------------------------
@st.cache_resource(show_spinner="🔨 Building vector database...")
def build_rag():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    splits = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm_for_rag = Ollama(model="llama3", temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_for_rag,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return qa_chain, retriever

qa_chain, retriever = build_rag()
llm = Ollama(model="llama3", temperature=0)  # LLM-first

# -------------------------
# Chat state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Casual intent detection via LLM
# -------------------------
def is_casual_intent(user_input):
    """
    Returns True if the user_input is casual/small-talk and NOT a document question.
    """
    prompt = (
        f"Decide if the following user input is casual/small talk and NOT a question "
        f"related to any documents. Return True or False ONLY.\n\nUser input: '{user_input}'"
    )
    response = llm.generate([prompt])
    answer = response.generations[0][0].text.lower()
    return "true" in answer

# -------------------------
# Get response function
# -------------------------
def get_response(user_input):
    sources = []

    # Step 1: Check casual intent
    if is_casual_intent(user_input):
        response = llm.generate([user_input])
        answer = response.generations[0][0].text
        return answer, sources

    # Step 2: Check for document relevance
    try:
        docs = retriever.get_relevant_documents(user_input)
    except AttributeError:
        docs = retriever.vectorstore.similarity_search(user_input, k=4)

    # Step 3: Use RAG if documents found, else fallback to LLM
    if docs:
        result = qa_chain({"question": user_input})
        answer = result["answer"]
        sources = result["source_documents"]
    else:
        response = llm.generate([user_input])
        answer = response.generations[0][0].text

    return answer, sources

# -------------------------
# User input
# -------------------------
user_input = st.chat_input("Ask anything…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            answer, sources = get_response(user_input)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    if sources:
        with st.expander("📚 Sources"):
            for doc in sources:
                st.write(f"**File:** {doc.metadata.get('source', 'Unknown')}")
                if "page" in doc.metadata:
                    st.write(f"**Page:** {doc.metadata['page']}")
