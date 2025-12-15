import streamlit as st
import os
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
    /* 1. FORCE LIGHT THEME & BACKGROUNDS */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* 2. REMOVE BLACK BARS (Header & Footer) */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    div[data-testid="stBottom"] {
        background-color: #FFFFFF !important;
        border-top: 1px solid #F0F6F8;
    }
    
    /* 3. TEXT STYLING */
    h1, h2, h3, h4, p, li, .stMarkdown, .stCaption {
        color: #4A7A94 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* 4. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #F0F6F8;
        border-right: 1px solid #E1EFFF;
    }
    
    /* 5. BUTTON STYLING */
    div.stButton > button:first-child {
        background-color: #8ABCCE !important; 
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-size: 18px !important;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #79A8B8 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    
    /* 6. CHAT INPUT STYLING */
    .stChatInput {
        padding-bottom: 15px;
    }
    .stChatInput textarea {
        background-color: #F0F6F8 !important;
        color: #4A7A94 !important;
        border: 1px solid #E1EFFF !important;
        border-radius: 20px !important;
        padding: 10px 15px;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.03);
    }
    .stChatInput textarea:focus {
        border-color: #8ABCCE !important;
        box-shadow: 0 0 0 2px rgba(138, 188, 206, 0.25) !important;
    }
    .stChatInput button {
        color: #8ABCCE !important;
    }
    
    /* 7. LIST STYLING */
    .kai-list {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #4A7A94;
        margin-top: 10px;
    }
    .kai-list strong {
        font-weight: 700;
        color: #2E5E74;
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
        
        st.markdown("""
        <div class="kai-list">
            • <strong>Kind:</strong> A calm, non-judgmental presence.<br>
            • <strong>
