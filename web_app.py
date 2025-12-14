import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
# We use the "Classic" chains that match your environment
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
st.set_page_config(page_title="Autism Resource AI", page_icon="🧩")
st.title("🧩 Autism Resource Assistant")
st.markdown("Ask me about resources, visual schedules, or support groups!")

# 1. SETUP API KEY (Cloud-Safe)
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ Missing API Key! Make sure you have a .streamlit/secrets.toml file locally.")

# 2. LOAD THE BRAIN (Cached)
@st.cache_resource
def load_rag_pipeline():
    # A. Setup Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # B. Load Database
    if not os.path.exists("./chroma_db_data"):
        return None
        
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db_data"
    )
    retriever = vectorstore.as_retriever()
    
    # C. Setup LLM (Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    
    # D. Define the Prompt
    system_prompt = (
        "You are a compassionate and helpful assistant connecting people to autism resources. "
        "Use the retrieved context to answer the question. "
        "If the user asks for a specific type of resource (like visual schedules), "
        "list the options available in the context. "
        "If you don't know, say so gently."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # E. Build the Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Load the pipeline
rag_chain = load_rag_pipeline()

if rag_chain is None:
    st.error("❌ Database not found! Please run 'rag_app.py' locally to build the database, then push the 'chroma_db_data' folder to GitHub.")

# 3. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- REACT TO USER INPUT ---
if prompt := st.chat_input("Ex: Where can I find visual schedules?"):
    # 1. Show User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        if rag_chain:
            with st.spinner("Searching resources..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    source_documents = response["context"]

                    # Display Answer
                    st.markdown(answer)
                    
                    # Display Sources (The upgraded JSON version)
                    with st.expander("📚 Recommended Resources"):
                        unique_sources = set()
                        
                        for doc in source_documents:
                            # In your JSON script, we saved the Name as "source" and Website as "url"
                            name = doc.metadata.get("source", "Unknown Resource")
                            url = doc.metadata.get("url", "")
                            
                            # Create a nice markdown link if the URL exists
                            if url and url != "N/A":
                                citation = f"[{name}]({url})"
                            else:
                                citation = name
                                
                            unique_sources.add(citation)
                        
                        # Display the list
                        if unique_sources:
                            for source in unique_sources:
                                st.markdown(f"- {source}")
                        else:
                            st.markdown("_No specific resources cited._")
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
