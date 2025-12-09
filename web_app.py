import streamlit as st
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
# Using your specific "Classic" imports
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
st.set_page_config(page_title="Textbook AI", page_icon="📚")
st.title("📚 Chat with your Textbook")

# NEW (Cloud Safe)
# Streamlit will automatically load this from its secret vault
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # Fallback for local testing (optional, or just set it in your terminal)
    st.error("Missing API Key in Secrets!")

# 2. CACHED RESOURCE LOADING
# We use @st.cache_resource so it doesn't reload the database every time you type
@st.cache_resource
def load_rag_pipeline():
    # A. Setup Embeddings (New Model)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # B. Load Existing Database
    # We assume you ran rag_app.py once already to build this folder!
    if not os.path.exists("./chroma_db_data"):
        st.error("❌ Database not found! Please run 'rag_app.py' first to ingest the PDF.")
        return None
        
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db_data"
    )
    retriever = vectorstore.as_retriever()
    
    # C. Setup LLM (The New Flash Model)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    
    # D. Create Chain
    system_prompt = (
        "You are a helpful student assistant. "
        "Use the retrieved context to answer the question. "
        "If you don't know, say so. Keep answers concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Load the brain
rag_chain = load_rag_pipeline()

# 3. CHAT INTERFACE
# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the PDF..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        if rag_chain:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                
                # Save assistant response
                st.session_state.messages.append({"role": "assistant", "content": answer})