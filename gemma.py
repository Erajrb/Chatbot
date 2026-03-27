import streamlit as st
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

# -----------------------------
# Initialize memory
# -----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
memory = st.session_state.memory

# -----------------------------
# Initialize chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Gemma Chatbot with Memory")

# -----------------------------
# Display previous messages
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # -----------------------------
    # Prompt template
    # -----------------------------
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Respond clearly to user queries."),
        ("user", "Question: {question}")
    ])

    # -----------------------------
    # LLM and output parser
    # -----------------------------
    llm = Ollama(model="gemma3:4b")
    output_parser = StrOutputParser()

    # -----------------------------
    # Build chain with memory
    # -----------------------------
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_parser=output_parser,
        memory=memory
    )

    # -----------------------------
    # Run chain
    # -----------------------------
    response = chain.run(user_input)

    # -----------------------------
    # Store assistant response
    # -----------------------------
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
