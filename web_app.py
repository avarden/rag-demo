import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION & STYLING ---
# Set page config using the new logo
st.set_page_config(page_title="KAI: Kind AI", page_icon="kai_logo.png", layout="centered")

# --- CUSTOM CSS FOR MOCKUP THEME ---
# This injects CSS to override Streamlit's defaults to match your calm blue/white mockup
st.markdown("""
    <style>
    /* Ensure main background is white */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Text Colors - A calming dark blue/grey */
    h1, h2, h3, p, li, .stMarkdown {
        color: #4A7A94 !important;
        text-align: center; /* Default center alignment for intro */
    }

    /* Primary Button Styling (The boat color) */
    div.stButton > button:first-child {
        background-color: #8ABCCE !important; /* Calm blue */
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 15px 30px;
        font-size: 20px !important;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    /* Hover effect for button */
    div.stButton > button:first-child:hover {
        background-color: #79A8B8 !important;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }

    /* Centering Images */
    div[data-testid="stImage"] > img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Custom styling for the bullet list to center the block but left-align text */
    .kai-list-wrapper {
         display: flex;
         justify-content: center;
         margin-top: 20px;
         margin-bottom: 30px;
    }
    .kai-list {
        text-align: left;
        display: inline-block;
        font-size: 1.2rem;
        line-height: 1.6;
        color: #4A7A94;
    }
    .kai-list strong {
        font-weight: bold;
    }

    /* Adjustments for chat interface to keep it clean */
    .stChatMessage {
        text-align: left !important;
    }
    div[data-testid="stChatMessageContent"] p {
        text-align: left !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 1. SETUP API KEY
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # Fallback for local testing if secrets not found
    # os.environ["GOOGLE_API_KEY"] = "YOUR_KEY_HERE" # Uncomment for quick local test
    pass

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
    # Check for API key before trying to load
    if os.environ.get("GOOGLE_API_KEY") is None:
        return None

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

# --- 3. INTRO SCREEN (Mockup Design) ---
if not st.session_state.intro_complete:
    # 1. Logo
    try:
        st.image("kai_logo.png", width=180)
    except FileNotFoundError:
        st.warning("Please save your logo image as 'kai_logo.png' in the project folder.")
        st.title("🌿 KAI")

    # 2. Headline
    st.markdown("<h1>KAI: Kind AI - A guide for everyday life.</h1>", unsafe_allow_html=True)
    
    # 3. Simplified Bullet Points (Using custom HTML for precise centering)
    st.markdown("""
        <div class="kai-list-wrapper">
            <div class="kai-list">
                • <strong>Kind:</strong> A calm, non-judgmental presence.<br>
                • <strong>Assistive:</strong> Focused on practical help.<br>
                • <strong>Intelligent:</strong> Meaningful guidance.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 4. Button (Begin your journey)
    # We use columns to ensure the button stays centered and doesn't stretch too wide
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # use_container_width=True makes the button fill the column width
        if st.button("Begin your journey", type="primary", use_container_width=True):
            st.session_state.intro_complete = True
            st.rerun()

# --- 4. ONBOARDING FLOW (Simplified Styling) ---
elif not st.session_state.onboarding_complete:
    st.markdown("<h2>Getting started</h2>", unsafe_allow_html=True)
    st.markdown("<p>To help KAI guide you better, please select an option:</p>", unsafe_allow_html=True)
    st.write("") # Spacing

    if st.session_state.user_role is None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("I am an Autistic Adult", use_container_width=True):
                st.session_state.user_role = "Autistic Adult"
                st.rerun()
        with col2:
            if st.button("I am a Caregiver", use_container_width=True):
                st.session_state.user_role = "Caregiver"
                st.rerun()

    else:
        role = st.session_state.user_role
        if role == "Autistic Adult":
            st.markdown("<h3>How old are you?</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3>How old is the person you care for?</h3>", unsafe_allow_html=True)
            
        # Number input doesn't need custom styling, it looks okay by default
        col_a, col_b, col_c = st.columns([1,1,1])
        with col_b:
             age_input = st.number_input("Age", min_value=1, max_value=120, value=18, label_visibility="collapsed")
        
        st.write("")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Start Chat", type="primary", use_container_width=True):
                st.session_state.age_context = age_input
                st.session_state.onboarding_complete = True
                st.rerun()

# --- 5. MAIN CHAT INTERFACE ---
else:
    # Minimal Header with Logo
    col1, col2 = st.columns([1, 5])
    with col1:
         try:
            st.image("kai_logo.png", width=60)
         except:
             st.write("🌿")
    with col2:
        # Using HTML to left align the title in the chat view
        st.markdown("<h2 style='text-align: left; margin-top: 10px;'>KAI</h2>", unsafe_allow_html=True)

    # Sidebar Context
    with st.sidebar:
        st.header("Context")
        st.markdown(f"**Role:** {st.session_state.user_role}")
        st.markdown(f"**Age:** {st.session_state.age_context}")
        st.markdown("---")
        if st.button("Reset KAI", type="secondary"):
            st.session_state.clear()
            st.rerun()

    # API Key Check
    if os.environ.get("GOOGLE_API_KEY") is None:
         st.error("❌ API Key not found. Please check your secrets.toml file.")
         st.stop()

    if rag_chain is None:
        st.error("❌ Database not found! Please run 'rag_app.py' locally first.")
        st.stop()

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
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
                        
                        # Citations (Left aligned by default in chat)
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
