"""
Graph RAG - Streamlit Frontend
"""
import streamlit as st
from api_client import GraphRAGClient
import time


# Page config
st.set_page_config(
    page_title="UET Graph RAG",
    page_icon="🎓",
    layout="wide"
)

# Initialize API client
if "client" not in st.session_state:
    st.session_state.client = GraphRAGClient()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar
with st.sidebar:
    st.title("🎓 UET Graph RAG")
    st.markdown("---")
    
    # Connection status
    st.subheader("Connection Status")
    if st.session_state.client.check_health():
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Offline")
        st.info("Start backend with: `python main.py`")
    
    st.markdown("---")
    
    # Info
    st.subheader("About")
    st.markdown("""
    This system answers questions about **UET Lahore** departments, programs, and faculty using:
    - **Graph Database** (Neo4j)
    - **Vector Search** (Embeddings)
    - **LLM** (Gemini/Ollama/vLLM)
    """)
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main chat area
st.title("💬 Chat with UET Prospectus")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            
            # Status badge
            if metadata.get("is_department_related"):
                st.caption("✅ Department-related question")
            else:
                st.caption("⚠️ Out of scope")
            
            # Sources
            if metadata.get("sources"):
                with st.expander("📚 Sources"):
                    for i, source in enumerate(metadata["sources"], 1):
                        st.code(f"{i}. Chunk ID: {source}", language=None)

# Chat input
if prompt := st.chat_input("Ask about UET departments, programs, or faculty..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.client.send_message(prompt)
                
                # Display answer
                st.markdown(response["answer"])
                
                # Show metadata
                if response.get("is_department_related"):
                    st.caption("✅ Department-related question")
                else:
                    st.caption("⚠️ Out of scope")
                
                # Sources
                if response.get("sources"):
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(response["sources"], 1):
                            st.code(f"{i}. Chunk ID: {source}", language=None)
                
                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "metadata": {
                        "is_department_related": response.get("is_department_related"),
                        "sources": response.get("sources", [])
                    }
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure the backend is running: `python main.py`")
