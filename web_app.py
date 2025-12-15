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
    /* 1. FORCE LIGHT THEME & BACKGROUNDS */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* 2. REMOVE BLACK BARS */
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
    
    /* 5. PRIMARY BUTTON STYLING (Begin Journey / Start Chat) */
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

    /* 6. SUGGESTION BUTTONS (The new "Cards") */
    /* We target buttons inside the main chat area specifically */
    div[data-testid="column"] button {
        background-color: #F0F6F8 !important; /* Light Grey-Blue */
        color: #4A7A94 !important;
        border: 1px solid #E1EFFF !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        height: auto !important;
        padding: 15px !important;
        text-align: left !important;
        white-space: normal !important; /* Allow text to wrap */
    }
    div[data-testid="column"] button:hover {
        border-color: #8ABCCE !important;
        background-color: #FFFFFF !important;
        color: #8ABCCE !important;
        transform: translateY(-2px);
    }
    
    /* 7. CHAT INPUT STYLING (Double Border Fix) */
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
        background-color: #F0F6F8 !important;
        color: #4A7A94 !important;
        border: 1px solid #E1EFFF !important;
        border-radius: 25px !important;
        padding: 12px 20px !important;
        box-shadow: none !important;
    }
    textarea[data-testid="stChatInputTextArea"]:focus {
        border-color: #8ABCCE !important;
        box-shadow: 0 0 0 3px rgba(138, 188, 206, 0.2) !important;
        outline: none !important;
    }
    button[data-testid="stChatInputSubmitButton"] {
        background-color: transparent !important;
        color: #8ABCCE !important;
        border: none !important;
        box-shadow: none !important;
    }
    button[data-testid="stChatInputSubmitButton"]:hover {
        color: #4A7A94 !important;
    }
    
    /* 8. LIST STYLING */
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

# --- HELPER: GENERATE RESPONSE ---
# We use this function so both BUTTONS and CHAT INPUT can trigger the same logic
def generate_response(prompt_text):
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    
    # 2. Generate Assistant Response
    if rag_chain:
        # Create a placeholder for the "Thinking..." state
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
                    
                    # Add to history (so it persists)
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
    # Sidebar
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

    # --- CHAT HISTORY ---
    # We display the messages manually
    if not st.session_state.messages:
        st.markdown("## Hello. How can I guide you today?")
        
        # --- SUGGESTED PROMPTS (Only when history is empty) ---
        st.write("")
        st.markdown("<p style='color: #4A7A94; opacity: 0.8;'>Here are a few ways I can help:</p>", unsafe_allow_html=True)
        
        # Dynamic Questions based on Role
        if st.session_state.user_role == "Autistic Adult":
            suggestions = [
                "Help me create a calm morning routine.",
                "How can I explain my sensory needs to friends?",
                "What are some tips for handling burnout?",
            ]
        else: # Caregiver
            suggestions = [
                "Suggest sensory-friendly activities for a child.",
                "How can I support them during a meltdown?",
                "Help me prepare for an IEP meeting.",
            ]
        
        # Display as Columns
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
        # -------------------------------------------------------

    if rag_chain is None:
        st.error("❌ Database missing. Please check your setup.")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if they exist in this message
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

    # --- INPUT ---
    if prompt := st.chat_input("Ask about routines, resources, or support..."):
        generate_response(prompt)
        st.rerun()
