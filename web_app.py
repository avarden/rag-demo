import streamlit as st
import os
import json
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
    h1, h2, h3, h4, p, li, .stMarkdown, .stCaption, label {
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

    /* 5. SUGGESTION BUTTONS (Outline Style) */
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
    
    /* 7. AVATAR COLORS */
    div[data-testid="stChatMessageAvatarUser"] {
        background-color: #5A7080 !important;
    }
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
if "user_location" not in st.session_state:
    st.session_state.user_location = None
if "age_context" not in st.session_state:
    st.session_state.age_context = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# NEW: Store dynamic suggestions
if "current_suggestions" not in st.session_state:
    st.session_state.current_suggestions = []

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
    
    # --- UPDATED PROMPT: ADDED "SUGGESTIONS" LOGIC ---
    system_prompt = (
        "You are KAI (Kind AI), a guide for everyday life. "
        "Your core philosophy:\n"
        "1. Kind: Be a calm, non-judgmental presence.\n"
        "2. Assistive: Focus on practical help, not diagnosis.\n"
        "3. Intelligent: Understand context and provide meaningful guidance.\n\n"
        "CONTEXT: You are speaking to a {role} located in {location}. "
        "The individual is {age} years old.\n\n"
        "--- INSTRUCTIONS FOR RESPONSE STYLE ---\n"
        "IF SPEAKING TO AN 'Autistic Adult':\n"
        "1. Use simple, short sentences.\n"
        "2. Use literal language only (NO metaphors, idioms, or sarcasm).\n"
        "3. If giving instructions, use a numbered list with a MAXIMUM of 7 steps.\n\n"
        "IF SPEAKING TO A 'Caregiver':\n"
        "1. Be supportive, empathetic, and detailed.\n\n"
        "--- FOLLOW-UP INSTRUCTION (CRITICAL) ---\n"
        "At the very end of your response, on a new line, provide exactly 3 short, relevant follow-up options for the user to click.\n"
        "Format them strictly like this:\n"
        "SUGGESTIONS: Option 1 | Option 2 | Option 3\n\n"
        "RETRIEVED KNOWLEDGE:\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

rag_chain = load_rag_pipeline()

# --- HELPER: GET LOCATIONS ---
def get_locations_from_file():
    locations = set()
    if os.path.exists("resources.json"):
        try:
            with open("resources.json", "r") as f:
                data = json.load(f)
                for item in data:
                    city = item.get("city", "N/A")
                    country = item.get("country", "N/A")
                    if city != "N/A" and country != "N/A":
                        locations.add(f"{city}, {country}")
                    elif country != "N/A":
                        locations.add(country)
        except:
            pass
    sorted_locs = sorted(list(locations))
    sorted_locs.append("Other / International")
    return sorted_locs

# --- HELPER: GENERATE RESPONSE & PARSE SUGGESTIONS ---
def generate_response(prompt_text):
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    
    # 2. Clear previous suggestions immediately
    st.session_state.current_suggestions = []
    
    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking gently..."):
                try:
                    response = rag_chain.invoke({
                        "input": prompt_text,
                        "role": st.session_state.user_role,
                        "location": st.session_state.user_location, 
                        "age": str(st.session_state.age_context)
                    })
                    raw_answer = response["answer"]
                    source_documents = response["context"]
                    
                    # --- PARSING LOGIC ---
                    if "SUGGESTIONS:" in raw_answer:
                        parts = raw_answer.split("SUGGESTIONS:")
                        clean_answer = parts[0].strip()
                        suggestions_text = parts[1].strip()
                        
                        # Split by pipe '|' and clean up
                        new_suggestions = [s.strip() for s in suggestions_text.split("|")]
                        st.session_state.current_suggestions = new_suggestions
                    else:
                        clean_answer = raw_answer
                        st.session_state.current_suggestions = []
                    
                    # Display Answer
                    st.markdown(clean_answer)
                    
                    # Display Sources
                    if source_documents:
                         with st.expander("📚 Helpful Resources"):
                            unique_sources = set()
                            for doc in source_documents:
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
                                
                    # Save to history (clean version)
                    st.session_state.messages.append({"role": "assistant", "content": clean_answer, "sources": source_documents})
                    
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
        
        # STEP 1: ROLE
        if st.session_state.user_role is None:
            st.markdown("<p style='text-align: center;'>To help KAI guide you better, please select an option:</p>", unsafe_allow_html=True)
            st.write("") 
            b1, b2 = st.columns(2)
            with b1:
                if st.button("I am an Autistic Adult", use_container_width=True):
                    st.session_state.user_role = "Autistic Adult"
                    st.rerun()
            with b2:
                if st.button("I am a Caregiver", use_container_width=True):
                    st.session_state.user_role = "Caregiver"
                    st.rerun()
        
        # STEP 2: LOCATION
        elif st.session_state.user_location is None:
            st.markdown("<h3 style='text-align: center;'>Where are you located?</h3>", unsafe_allow_html=True)
            location_options = get_locations_from_file()
            selected_loc = st.selectbox("Select Location", location_options, index=None, placeholder="Choose a location...")
            st.write("")
            if st.button("Next", type="primary", use_container_width=True):
                if selected_loc:
                    st.session_state.user_location = selected_loc
                    st.rerun()
                else:
                    st.warning("Please select a location (or 'Other').")

        # STEP 3: AGE
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
        st.info(f"**Role:** {st.session_state.user_role}\n\n**Loc:** {st.session_state.user_location}\n\n**Age:** {st.session_state.age_context}")
        st.write("")
        if st.button("Reset KAI"):
            st.session_state.clear()
            st.rerun()

    # --- DISPLAY MESSAGES ---
    if not st.session_state.messages:
        st.markdown("## Hello. How can I guide you today?")
        # Set default suggestions if history is empty
        if not st.session_state.current_suggestions:
            if st.session_state.user_role == "Autistic Adult":
                st.session_state.current_suggestions = [
                    "Help me create a calm morning routine.",
                    "How can I explain my sensory needs?",
                    "What are some tips for burnout?",
                ]
            else:
                st.session_state.current_suggestions = [
                    "Suggest sensory-friendly activities.",
                    "How can I support them during a meltdown?",
                    "Help me prepare for an IEP meeting.",
                ]

    if rag_chain is None:
        st.error("❌ Database missing. Please check your setup.")
        st.stop()

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

    # --- DYNAMIC SUGGESTION BUTTONS ---
    # We display these just above the chat input
    if st.session_state.current_suggestions:
        st.write("")
        st.markdown("<p style='color: #0E2A3A; opacity: 0.9; font-weight: 600;'>Suggested next steps:</p>", unsafe_allow_html=True)
        
        # Determine columns (handle 1, 2, or 3 suggestions safely)
        cols = st.columns(len(st.session_state.current_suggestions))
        
        for i, suggestion in enumerate(st.session_state.current_suggestions):
            if cols[i].button(suggestion, key=f"sugg_{i}"):
                generate_response(suggestion)
                st.rerun()

    if prompt := st.chat_input("Ask about routines, resources, or support..."):
        generate_response(prompt)
        st.rerun()
