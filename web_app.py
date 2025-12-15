# --- HELPER: GENERATE RESPONSE & PARSE SUGGESTIONS ---
def generate_response(prompt_text):
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.session_state.current_suggestions = [] # Clear old suggestions
    
    # 2. Build History for Memory
    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking gently..."):
                try:
                    # Run the AI
                    response = rag_chain.invoke({
                        "input": prompt_text,
                        "chat_history": chat_history,
                        "role": st.session_state.user_role,
                        "location": st.session_state.user_location, 
                        "age": str(st.session_state.age_context)
                    })
                    raw_answer = response["answer"]
                    source_documents = response["context"]
                    
                    # Parse Suggestions
                    if "SUGGESTIONS:" in raw_answer:
                        parts = raw_answer.split("SUGGESTIONS:")
                        clean_answer = parts[0].strip()
                        suggestions_text = parts[1].strip()
                        new_suggestions = [s.strip() for s in suggestions_text.split("|")]
                        st.session_state.current_suggestions = new_suggestions
                    else:
                        clean_answer = raw_answer
                        st.session_state.current_suggestions = []
                    
                    # Display Answer
                    st.markdown(clean_answer)
                    
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
                                
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": clean_answer, "sources": source_documents})
                    
                    # --- SUCCESS? NOW WE RERUN ---
                    # Only rerun if everything worked perfectly.
                    st.rerun()
                    
                except Exception as e:
                    # --- FAILURE? SHOW ERROR ---
                    # Do NOT rerun. Let the error stay on screen so we can read it.
                    st.error(f"An error occurred: {e}")
