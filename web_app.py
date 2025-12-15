import streamlit as st
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
st.set_page_config(page_title="KAI: Kind AI", page_icon="kai_logo.png", layout="wide")

# --- BRANDING & CSS ---
st.markdown("""
    <style>
    /* 1. GLOBAL TEXT & BACKGROUNDS */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Global Text Color - Deep Navy for AAA Compliance */
    h1, h2, h3, h4, p, li, .stMarkdown, .stCaption {
        color: #0E2A3A !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* 2. HEADER & FOOTER CLEANUP */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    div[data-testid="stBottom"] {
        background-color: #FFFFFF !important;
        border-top: 1px solid #F0F6F8;
    }
    
    /* 3. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #F4F8FA;
        border-right: 1px solid #E1EFFF;
    }
    
    /* 4. PRIMARY BUTTONS */
    div.stButton > button:first-child {
        background-color: #1F455C !important; 
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child p {
        color: #FFFFFF !important; 
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #0E2A3A !important;
        transform: translateY(-1px);
    }

    /* 5. SUGGESTION BUTTONS */
    div[data-testid="column"] button {
        background-color: #FFFFFF !important; 
        border: 2px solid #E1EFFF !important; 
        height: auto !important;
        padding: 15px !important;
        text-align: left !important;
    }
    div[data-testid="column"] button p {
        color: #0E2A3A !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    div[data-testid="column"] button:hover {
        border-color: #1F455C !important;
        background-color: #F4F8FA !important;
        transform: translateY(-2px);
    }
    
    /* 6. CHAT INPUT STYLING */
    div[data-testid="stChatInput"] {
        background-color: transparent !important;
        border-color: transparent !important; 
    }
    .stChatInput > div {
        background-color: transparent !important;
        border: none !important; 
        box-shadow: none !important;
    }
    textarea[data-testid="stChatInputTextArea"] {
        background-color: #F4F8FA !important;
        color: #0E2A3A !important;
        caret-color: #0E2A3A;
        border: 2px solid #E1EFFF !important;
        border-radius: 25px !important;
        padding: 12px 20px !important;
    }
    textarea[data-testid="stChatInputTextArea"]:focus {
        border-color: #1F455C !important;
        box-shadow: 0 0 0 3px rgba(31, 69, 92, 0.15) !important;
        outline: none !important;
    }
    button[data-testid="stChatInputSubmitButton"] {
        background-color: transparent !important;
        color: #1F455C !important;
        border: none !important;
    }
    
    /* 7. AVATAR COLORS (The Fix) */
    
    /* User Avatar (Was Red -> Now Calm Blue-Grey) */
    div[data-testid="stChatMessageAvatarUser"] {
        background-color: #5A7080 !important;
    }
    
    /* Assistant Avatar (Was Orange -> Now Brand Teal) */
    div[data-testid="stChatMessageAvatarAssistant"] {
        background-color: #2C5E7A !important;
    }
    
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 1. SETUP API KEY
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- SESSION STATE ---
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

# 2. LOAD BRAIN
@st.cache_resource
def load_rag_pipeline():
    if os.environ.get("GOOGLE_API_KEY") is None:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    if not os.path.exists("./chroma_db_data"):
        return None
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_data")
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    
    system_prompt = (
        "You are KAI (Kind AI), a guide for everyday life. "
        "Your core philosophy:\n"
        "1. Kind: Be a calm, non-judgmental presence.\n"
        "2. Assistive: Focus on practical help, not diagnosis.\n"
        "3. Intelligent: Understand context and provide meaningful guidance.\n\n"
        "CONTEXT: You are speaking to a {role}. "
        "The individual is {age} years old. Adjust your language accordingly.\n\n"
        "RETRIEVED KNOWLEDGE:\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

rag_chain = load_rag_pipeline()

# --- HELPER: GENERATE RESPONSE ---
def generate_response(prompt_text):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking gently..."):
                try:
                    response = rag_chain.invoke({
                        "input": prompt_text,
                        "role": st.session_state.user_role,
                        "age": str(st.session_state.age_context)
                    })
                    answer = response["answer"]
                    source_documents = response["context"]
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": source_documents})
                except Exception as e:
                    st.error(f"Error: {e}")

# --- 3. INTRO SCREEN ---
if not st.session_state.intro_complete:
    st.write("") 
    st.write("")
    col1, col_spacer, col2 = st.columns([1, 0.2, 1.5])
    with col1:
        try:
            st.image("kai_logo.png", use_container_width=True)
        except:
            st.header("🌿 KAI")
    with col2:
        st.markdown("<h1 style='text-align: left; margin-bottom: 0;'>KAI: Kind AI</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left; margin-top: 0; font-weight: 400;'>A guide for everyday life.</h3>", unsafe_allow_html=True)
        intro_list_html = """
        <div class="kai-list">
            • <strong>Kind:</strong> A calm, non-judgmental presence.<br>
            • <strong>Assistive:</strong> Focused on practical help.<br>
            • <strong>Intelligent:</strong> Meaningful guidance.
        </div>
        """
        st.markdown(intro_list_html, unsafe_allow_html=True)
        st.write("") 
        st.write("") 
        if st.button("Begin your journey", type="primary"):
            st.session_state.intro_complete = True
            st.rerun()

# --- 4. ONBOARDING ---
elif not st.session_state.onboarding_complete:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h2 style='text-align: center;'>Getting started</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>To help KAI guide you better, please select an option:</p>", unsafe_allow_html=True)
        st.write("") 
        if st.session_state.user_role is None:
            b1, b2 = st.columns(2)
            with b1:
                if st.button("I am an Autistic Adult", use_container_width=True):
                    st.session_state.user_role = "Autistic Adult"
                    st.rerun()
            with b2:
                if st.button("I am a Caregiver", use_container_width=True):
                    st.session_state.user_role = "Caregiver"
                    st.rerun()
        else:
            role = st.session_state.user_role
            if role == "Autistic Adult":
                st.markdown("<h3 style='text-align: center;'>How old are you?</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='text-align: center;'>How old is the person you care for?</h3>", unsafe_allow_html=True)
            age_input = st.number_input("Age", min_value=1, max_value=120, value=18, label_visibility="collapsed")
            st.write("")
            if st.button("Start Chat", type="primary", use_container_width=True):
                st.session_state.age_context = age_input
                st.session_state.onboarding_complete = True
                st.rerun()

# --- 5. MAIN CHAT INTERFACE ---
else:
    with st.sidebar:
        try:
            st.image("kai_logo.png", width=80)
        except:
            st.write("🌿")
        st.markdown("### Context")
        st.info(f"**Role:** {st.session_state.user_role}\n\n**Age:** {st.session_state.age_context}")
        st.write("")
        if st.button("Reset KAI"):
            st.session_state.clear()
            st.rerun()

    if not st.session_state.messages:
        st.markdown("## Hello. How can I guide you today?")
        st.write("")
        st.markdown("<p style='color: #0E2A3A; opacity: 0.9; font-weight: 600;'>Here are a few ways I can help:</p>", unsafe_allow_html=True)
        
        if st.session_state.user_role == "Autistic Adult":
            suggestions = [
                "Help me create a calm morning routine.",
                "How can I explain my sensory needs to friends?",
                "What are some tips for handling burnout?",
            ]
        else:
            suggestions = [
                "Suggest sensory-friendly activities for a child.",
                "How can I support them during a meltdown?",
                "Help me prepare for an IEP meeting.",
            ]
        
        col1, col2, col3 = st.columns(3)
        if col1.button(suggestions[0]):
            generate_response(suggestions[0])
            st.rerun()
        if col2.button(suggestions[1]):
            generate_response(suggestions[1])
            st.rerun()
        if col3.button(suggestions[2]):
            generate_response(suggestions[2])
            st.rerun()

    if rag_chain is None:
        st.error("❌ Database missing. Please check your setup.")
        st.stop()

    # --- CHAT DISPLAY (Standard Icons, Custom CSS Colors) ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["role"] == "assistant":
                 with st.expander("📚 Helpful Resources"):
                    unique_sources = set()
                    for doc in message["sources"]:
                        name = doc.metadata.get("source", "Unknown Resource")
                        url = doc.metadata.get("url", "")
                        if url and url != "N/A":
                            unique_sources.add(f"[{name}]({url})")
                        else:
                            unique_sources.add(name)
                    if unique_sources:
                        for source in unique_sources:
                            st.markdown(f"- {source}")
                    else:
                        st.markdown("_No specific resources cited._")

    if prompt := st.chat_input("Ask about routines, resources, or support..."):
        generate_response(prompt)
        st.rerun()
