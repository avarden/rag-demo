import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
st.set_page_config(page_title="Autism Resource AI", page_icon="🧩")

# 1. SETUP API KEY
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ Missing API Key! Make sure you have a .streamlit/secrets.toml file locally.")

# --- SESSION STATE INITIALIZATION ---
# We use these variables to track where the user is in the flow
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
    
    # Setup LLM
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    
    # --- DYNAMIC SYSTEM PROMPT ---
    # We leave placeholders {role} and {age} that we will fill in at runtime
    system_prompt = (
        "You are a compassionate and helpful assistant connecting people to autism resources. "
        "CONTEXT ABOUT USER: You are speaking to a {role}. "
        "The autistic individual is {age} years old. "
        "Adjust your tone and recommendations to be appropriate for this age group and role. "
        "Use the retrieved context to answer the question. "
        "If the user asks for a specific type of resource (like visual schedules), "
        "list the options available in the context. "
        "If you don't know, say so gently."
        "\n\n"
        "RETRIEVED KNOWLEDGE:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # Build Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

rag_chain = load_rag_pipeline()

# --- 3. ONBOARDING FLOW ---
if not st.session_state.onboarding_complete:
    st.title("🧩 Welcome")
    st.markdown("To provide the best resources, please tell us a bit about yourself.")
    
    # Step 1: Choose Role
    if st.session_state.user_role is None:
        st.subheader("I am...")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("An Autistic Adult"):
                st.session_state.user_role = "Autistic Adult"
                st.rerun() # Refresh to show next step
        
        with col2:
            if st.button("A Caregiver"):
                st.session_state.user_role = "Caregiver"
                st.rerun()

    # Step 2: Choose Age
    else:
        role = st.session_state.user_role
        
        if role == "Autistic Adult":
            st.subheader("How old are you?")
        else:
            st.subheader("How old is the person you care for?")
            
        # Age Input
        age_input = st.number_input("Age", min_value=1, max_value=120, value=18)
        
        if st.button("Start Chatting"):
            st.session_state.age_context = age_input
            st.session_state.onboarding_complete = True
            st.rerun()

# --- 4. MAIN CHAT INTERFACE ---
else:
    # Header with Context Badge
    st.title("🧩 Autism Resource Assistant")
    st.caption(f"Context: {st.session_state.user_role} | Age: {st.session_state.age_context}")
    
    # Reset Button (in sidebar)
    with st.sidebar:
        st.write("Current Settings:")
        st.info(f"Role: {st.session_state.user_role}\n\nAge: {st.session_state.age_context}")
        if st.button("Reset / Start Over"):
            st.session_state.clear()
            st.rerun()

    # Database Check
    if rag_chain is None:
        st.error("❌ Database not found! Please run 'rag_app.py' locally first.")

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to Input
    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            if rag_chain:
                with st.spinner("Searching resources..."):
                    try:
                        # --- CRITICAL: PASSING THE CONTEXT ---
                        # We pass the role and age into the prompt variables we defined earlier
                        response = rag_chain.invoke({
                            "input": prompt,
                            "role": st.session_state.user_role,
                            "age": str(st.session_state.age_context)
                        })
                        
                        answer = response["answer"]
                        source_documents = response["context"]

                        st.markdown(answer)
                        
                        # Sources Logic
                        with st.expander("📚 Recommended Resources"):
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
