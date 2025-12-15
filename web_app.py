import streamlit as st
import os
import json
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
st.set_page_config(page_title="KAI: Kind AI", page_icon="kai_logo.png", layout="wide")

# --- BRANDING & CSS ---
st.markdown("""
    <style>
    /* 1. GLOBAL TEXT & BACKGROUNDS */
    .stApp { background-color: #FFFFFF; }
    h1, h2, h3, h4, p, li, .stMarkdown, .stCaption, label { color: #0E2A3A !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    header[data-testid="stHeader"] { background-color: #FFFFFF !important; }
    div[data-testid="stBottom"] { background-color: #FFFFFF !important; border-top: 1px solid #F0F6F8; }
    section[data-testid="stSidebar"] { background-color: #F4F8FA; border-right: 1px solid #E1EFFF; }
    
    /* BUTTONS */
    div.stButton > button:first-child { background-color: #1F455C !important; border: none; border-radius: 10px; padding: 0.5rem 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.2s ease; }
    div.stButton > button:first-child p { color: #FFFFFF !important; font-size: 18px !important; font-weight: 600 !important; }
    div.stButton > button:first-child:hover { background-color: #0E2A3A !important; transform: translateY(-1px); }

    /* SUGGESTION BUTTONS */
    div[data-testid="column"] button { background-color: #FFFFFF !important; border: 2px solid #E1EFFF !important; height: auto !important; padding: 15px !important; text-align: left !important; }
    div[data-testid="column"] button p { color: #0E2A3A !important; font-size: 16px !important; font-weight: 500 !important; }
    div[data-testid="column"] button:hover { border-color: #1F455C !important; background-color: #F4F8FA !important; transform: translateY(-2px); }
    
    /* INPUT & AVATARS */
    div[data-testid="stChatInput"] { background-color: transparent !important; border-color: transparent !important; }
    .stChatInput > div { background-color: transparent !important; border: none !important; box-shadow: none !important; }
    textarea[data-testid="stChatInputTextArea"] { background-color: #F4F8FA !important; color: #0E2A3A !important; caret-color: #0E2A3A; border: 2px solid #E1EFFF !important; border-radius: 25px !important; padding: 12px 20px !important; }
    textarea[data-testid="stChatInputTextArea"]:focus { border-color: #1F455C !important; box-shadow: 0 0 0 3px rgba(31, 69, 92, 0.15) !important; outline: none !important; }
    div[data-testid="stChatMessageAvatarUser"] { background-color: #5A7080 !important; }
    div[data-testid="stChatMessageAvatarAssistant"] { background-color: #2C5E7A !important; }
    
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 1. SETUP API KEY
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- SESSION STATE ---
if "intro_complete" not in st.session_state: st.session_state.intro_complete = False
if "onboarding_complete" not in st.session_state: st.session_state.onboarding_complete = False
if "user_role" not in st.session_state: st.session_state.user_role = None
if "user_location" not in st.session_state: st.session_state.user_location = None
if "age_context" not in st.session_state: st.session_state.age_context = None
if "messages" not in st.session_state: st.session_state.messages = []
if "current_suggestions" not in st.session_state: st.session_state.current_suggestions = []

# 2. LOAD BRAIN (High Efficiency)
@st.cache_resource
def load_rag_pipeline():
    if os.environ.get("GOOGLE_API_KEY") is None: return None
    if not os.path.exists("./chroma_db_data"): return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_data")
        
        # --- SPEED OPTIMIZATION 1: REDUCE RETRIEVAL ---
        # Fetch only top 2 documents. 
        # This keeps the context focused and reduces the data sent to the LLM.
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # Using Flash Lite for max speed
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite", temperature=0)
        
        system_prompt = (
            "You are KAI (Kind AI), a guide for everyday life. "
            "Your core philosophy:\n"
            "1. Kind: Be a calm, non-judgmental presence.\n"
            "2. Assistive: Focus on practical help, not diagnosis.\n"
            "3. Intelligent: Understand context and provide meaningful guidance.\n\n"
            "CONTEXT: You are speaking to a {role} located in {location}. "
            "The individual is {age} years old.\n\n"
            "--- SPECIAL MODE: MATCHMAKING & CONNECTION ---\n"
            "IF the user asks to 'find friends' or 'connect with caregivers', OR if the chat history shows we are currently building a profile:\n"
            "1. Adopt a warm, welcoming 'Matchmaker' persona.\n"
            "2. Continue the interview process naturally. Ask ONE question at a time.\n"
            "3. Acknowledge their previous answer before asking the next one.\n"
            "4. Good questions for Autistic Adults: Special interests/Hobbies? Preferred communication (Text vs Voice)? Sensory needs for meetups?\n"
            "5. Good questions for Caregivers: Age of person supported? Main challenges? Looking for emotional support or practical tips?\n\n"
            "--- INSTRUCTIONS FOR RESPONSE STYLE ---\n"
            "IF SPEAKING TO AN 'Autistic Adult':\n"
            "1. Use simple, short sentences.\n"
            "2. Use literal language only (NO metaphors, idioms, or sarcasm).\n"
            "3. If giving instructions, use a numbered list with a MAXIMUM of 7 steps.\n\n"
            "IF SPEAKING TO A 'Caregiver':\n"
            "1. Be supportive, empathetic, and detailed.\n\n"
            "--- FOLLOW-UP INSTRUCTION ---\n"
            "At the very end of your response, on a new line, provide exactly 3 short, relevant follow-up options for the user to click.\n"
            "Format them strictly like this:\n"
            "SUGGESTIONS: Option 1 | Option 2 | Option 3\n\n"
            "RETRIEVED KNOWLEDGE:\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain
        
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        return None

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
                    if city != "N/A" and country != "N/A": locations.add(f"{city}, {country}")
                    elif country != "N/A": locations.add(country)
        except: pass
    sorted_locs = sorted(list(locations))
    sorted_locs.append("Other / International")
    return sorted_locs

# --- HELPER: GENERATE RESPONSE (Optimized History) ---
def generate_response(prompt_text):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.session_state.current_suggestions = [] 
    
    # --- SPEED OPTIMIZATION 2: TIGHTER HISTORY ---
    # Only keep the last 4 messages (2 exchanges). 
    # This keeps it fast while remembering the immediate previous question.
    recent_messages = st.session_state.messages[-4:] 
    
    chat_history = []
    for msg in recent_messages[:-1]:
        if msg["role"] == "user": chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant": chat_history.append(AIMessage(content=msg["content"]))

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking gently..."):
                max_retries = 3
                success = False
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        response = rag_chain.invoke({
                            "input": prompt_text,
                            "chat_history": chat_history,
                            "role": st.session_state.user_role,
                            "location": st.session_state.user_location, 
                            "age": str(st.session_state.age_context)
                        })
                        
                        raw_answer = response["answer"]
                        source_documents = response["context"]
                        
                        if "SUGGESTIONS:" in raw_answer:
                            parts = raw_answer.split("SUGGESTIONS:")
                            clean_answer = parts[0].strip()
                            suggestions_text = parts[1].strip()
                            st.session_state.current_suggestions = [s.strip() for s in suggestions_text.split("|")]
                        else:
                            clean_answer = raw_answer
                            st.session_state.current_suggestions = []
                        
                        st.markdown(clean_answer)
                        
                        if source_documents:
                             with st.expander("📚 Helpful Resources"):
                                unique_sources = set()
                                for doc in source_documents:
                                    name = doc.metadata.get("source", "Unknown Resource")
                                    url = doc.metadata.get("url", "")
                                    if url and url != "N/A": unique_sources.add(f"[{name}]({url})")
                                    else: unique_sources.add(name)
                                for source in unique_sources: st.markdown(f"- {source}")
                                    
                        st.session_state.messages.append({"role": "assistant", "content": clean_answer, "sources": source_documents})
                        success = True
                        break 
                        
                    except Exception as e:
                        last_error = e
                        if "429" in str(e):
                            time.sleep(2) 
                            continue 
                        else:
                            st.error(f"Error: {e}")
                            break 
                
                if success:
                    st.rerun()
                elif last_error and "429" in str(last_error):
                     st.error("KAI is busy (Rate Limit). Please wait 1 minute.")

# --- 3. INTRO SCREEN ---
if not st.session_state.intro_complete:
    st.write("") 
    st.write("")
    col1, col_spacer, col2 = st.columns([1, 0.2, 1.5])
    with col1:
        try: st.image("kai_logo.png", use_container_width=True)
        except: st.header("🌿 KAI")
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
        elif st.session_state.user_location is None:
            st.markdown("<h3 style='text-align: center;'>Where are you located?</h3>", unsafe_allow_html=True)
            location_options = get_locations_from_file()
            selected_loc = st.selectbox("Select Location", location_options, index=None, placeholder="Choose a location...")
            st.write("")
            if st.button("Next", type="primary", use_container_width=True):
                if selected_loc:
                    st.session_state.user_location = selected_loc
                    st.rerun()
                else: st.warning("Please select a location.")
        else:
            role = st.session_state.user_role
            if role == "Autistic Adult": st.markdown("<h3 style='text-align: center;'>How old are you?</h3>", unsafe_allow_html=True)
            else: st.markdown("<h3 style='text-align: center;'>How old is the person you care for?</h3>", unsafe_allow_html=True)
            age_input = st.number_input("Age", min_value=1, max_value=120, value=18, label_visibility="collapsed")
            st.write("")
            if st.button("Start Chat", type="primary", use_container_width=True):
                st.session_state.age_context = age_input
                st.session_state.onboarding_complete = True
                st.rerun()

# --- 5. MAIN CHAT INTERFACE ---
else:
    with st.sidebar:
        try: st.image("kai_logo.png", width=80)
        except: st.write("🌿")
        st.markdown("### Context")
        st.info(f"**Role:** {st.session_state.user_role}\n\n**Loc:** {st.session_state.user_location}\n\n**Age:** {st.session_state.age_context}")
        st.write("")
        connect_label = "Find friends" if st.session_state.user_role == "Autistic Adult" else "Connect with other caregivers"
        if st.button(connect_label, type="primary"):
            generate_response(f"I would like to {connect_label.lower()}. Please guide me through the process.")
        st.write("")
        if st.button("Reset KAI"):
            st.session_state.clear()
            st.rerun()

    if rag_chain is None:
        if not os.path.exists("./chroma_db_data"):
            st.error("❌ Database missing. Please run 'rag_app.py' locally.")
        else:
            st.error("❌ Connection Error: Check API Key or Model Name.")
        st.stop()

    if not st.session_state.messages:
        st.markdown("## Hello. How can I guide you today?")
        if not st.session_state.current_suggestions:
            if st.session_state.user_role == "Autistic Adult":
                st.session_state.current_suggestions = ["Help me create a calm morning routine.", "How can I explain my sensory needs?", "Tips for burnout?"]
            else:
                st.session_state.current_suggestions = ["Sensory-friendly activities.", "Support during a meltdown?", "Prepare for IEP meeting."]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["role"] == "assistant":
                 with st.expander("📚 Helpful Resources"):
                    unique_sources = set()
                    for doc in message["sources"]:
                        name = doc.metadata.get("source", "Unknown Resource")
                        url = doc.metadata.get("url", "")
                        if url and url != "N/A": unique_sources.add(f"[{name}]({url})")
                        else: unique_sources.add(name)
                    for source in unique_sources: st.markdown(f"- {source}")

    if st.session_state.current_suggestions:
        st.write("")
        st.markdown("<p style='color: #0E2A3A; opacity: 0.9; font-weight: 600;'>Suggested next steps:</p>", unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.current_suggestions))
        for i, suggestion in enumerate(st.session_state.current_suggestions):
            if cols[i].button(suggestion, key=f"sugg_{i}"): generate_response(suggestion)

    if prompt := st.chat_input("Ask about routines, resources, or support..."):
        generate_response(prompt)
