import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
st.set_page_config(page_title="KAI: Kind AI", page_icon="🌿")

# 1. SETUP API KEY
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ Missing API Key! Make sure you have a .streamlit/secrets.toml file locally.")

# --- SESSION STATE INITIALIZATION ---
if "intro_complete" not in st.session_state:
    st.session_state.intro_complete = False
if "onboarding_complete" not in st.session_state:
    st.session_state.onboarding_complete = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "age_context" not in st.session_state:
    st.session_state.age_context = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. LOAD THE BRAIN (Cached)
@st.cache_resource
def load_rag_pipeline():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    if not os.path.exists("./chroma_db_data"):
        return None
        
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db_data"
    )
    retriever = vectorstore.as_retriever()
    
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    
    # --- KAI PERSONA PROMPT ---
    system_prompt = (
        "You are KAI (Kind AI), a guide for everyday life. "
        "Your core philosophy:\n"
        "1. Kind: Be a calm, non-judgmental presence.\n"
        "2. Assistive: Focus on practical help, not diagnosis.\n"
        "3. Intelligent: Understand context and provide meaningful guidance.\n\n"
        "You are not here to control, correct, or 'fix' anyone. "
        "Offer step-by-step support with daily routines. "
        "Give simple, literal explanations of complex information. "
        "Provide calm prompts if the user seems overwhelmed.\n\n"
        "CONTEXT: You are speaking to a {role}. "
        "The individual is {age} years old. Adjust your language accordingly.\n\n"
        "RETRIEVED KNOWLEDGE:\n"
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

rag_chain = load_rag_pipeline()

# --- 3. INTRO SCREEN (The Brand Story) ---
if not st.session_state.intro_complete:
    st.title("🌿 KAI: Kind AI")
    st.subheader("A guide for everyday life.")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### **K**ind")
        st.markdown("### **A**ssistive")
        st.markdown("### **I**ntelligent")
    with col2:
        st.markdown("A calm, non-judgmental presence.")
        st.markdown("Focused on practical help, not diagnosis.")
        st.markdown("Able to understand context and provide meaningful guidance.")
    
    st.markdown("---")
    st.markdown(
        """
        **KAI is not here to control, correct, or “fix” anyone.** We act as a guide offering:
        * Step-by-step support with daily routines
        * Simple, literal explanations of complex information
        * Calm prompts when you feel overwhelmed
        """
    )
    st.markdown("---")
    
    if st.button("Begin Journey", type="primary"):
        st.session_state.intro_complete = True
        st.rerun()

# --- 4. ONBOARDING FLOW ---
elif not st.session_state.onboarding_complete:
    st.title("Settings")
    st.markdown("To help KAI guide you better, please select an option:")
    
    if st.session_state.user_role is None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("I am an Autistic Adult"):
                st.session_state.user_role = "Autistic Adult"
                st.rerun()
        with col2:
            if st.button("I am a Caregiver"):
                st.session_state.user_role = "Caregiver"
                st.rerun()

    else:
        role = st.session_state.user_role
        if role == "Autistic Adult":
            st.subheader("How old are you?")
        else:
            st.subheader("How old is the person you care for?")
            
        age_input = st.number_input("Age", min_value=1, max_value=120, value=18)
        
        if st.button("Start Chat"):
            st.session_state.age_context = age_input
            st.session_state.onboarding_complete = True
            st.rerun()

# --- 5. MAIN CHAT INTERFACE ---
else:
    # Minimal Header
    st.title("🌿 KAI")
    
    # Sidebar Context
    with st.sidebar:
        st.header("Context")
        st.info(f"**Role:** {st.session_state.user_role}\n\n**Age:** {st.session_state.age_context}")
        if st.button("Reset KAI"):
            st.session_state.clear()
            st.rerun()

    if rag_chain is None:
        st.error("❌ Database not found! Please run 'rag_app.py' locally first.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you today?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            if rag_chain:
                with st.spinner("Thinking gently..."):
                    try:
                        response = rag_chain.invoke({
                            "input": prompt,
                            "role": st.session_state.user_role,
                            "age": str(st.session_state.age_context)
                        })
                        
                        answer = response["answer"]
                        source_documents = response["context"]

                        st.markdown(answer)
                        
                        with st.expander("📚 Helpful Resources"):
                            unique_sources = set()
                            for doc in source_documents:
                                name = doc.metadata.get("source", "Unknown Resource")
                                url = doc.metadata.get("url", "")
                                
                                if url and url != "N/A":
                                    citation = f"[{name}]({url})"
                                else:
                                    citation = name
                                unique_sources.add(citation)
                            
                            if unique_sources:
                                for source in unique_sources:
                                    st.markdown(f"- {source}")
                            else:
                                st.markdown("_No specific resources cited._")
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
