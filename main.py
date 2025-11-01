"""
Main Entry Point for Post Discharge Medical AI Assistant

This module serves as the entry point for the multi-agent medical assistant system.
Uses Streamlit for the UI.
"""

import os
import sys
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging first
from utils.logger_config import setup_logging, get_logger
setup_logging()
import logfire

logger = get_logger(__name__)

# Import Streamlit
import streamlit as st

# Import agents and modules
from agents.receptionist_agent import ReceptionistAgent
from agents.clinical_agent import ClinicalAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from rag.vectorstore import get_vectorstore, initialize_vectorstore
from rag.embeddings import get_embedding_model
from rag.loader import build_vectorstore_from_pdf


# Initial greeting message
INITIAL_GREETING = "Hello! I'm here to help with your post-discharge care. Could you please tell me your name so I can look up your discharge report?"


@st.cache_resource
def initialize_agents():
    """
    Initialize all agents for the multi-agent system.
    
    Returns:
        Dictionary of initialized agents
    """
    logger.info("Initializing agents and resources...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Initializing agents and resources...")
    
    # Initialize embedding model
    logger.info("Initializing embedding model...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Initializing embedding model...")
    get_embedding_model()
    
    # Check if vectorstore exists, if not build it from PDF
    logger.info("Checking vectorstore...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Checking vectorstore...")
    vectorstore = get_vectorstore()
    if not vectorstore:
        logger.info("Vectorstore not found. Building from PDF...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Vectorstore not found. Building from PDF...")
        pdf_path = settings.NEPHROLOGY_PDF_PATH
        if os.path.exists(pdf_path):
            success = build_vectorstore_from_pdf(pdf_path, force_rebuild=False)
            if success:
                logger.info("Vectorstore built successfully")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Vectorstore built successfully")
                vectorstore = get_vectorstore()
            else:
                logger.warning("Failed to build vectorstore. RAG may not work properly.")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] Failed to build vectorstore. RAG may not work properly.")
        else:
            logger.warning(f"PDF file not found: {pdf_path}. RAG may not work properly.")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] PDF file not found: {pdf_path}. RAG may not work properly.")
    
    # Initialize LLM (only for answer generation - minimize API calls)
    logger.info("Initializing LLM (Gemini)...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Initializing LLM (Gemini)...")
    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found. LLM functionality may be limited.")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] GEMINI_API_KEY not found. LLM functionality may be limited.")
        llm = None
    else:
        llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.LLM_TEMPERATURE
        )
    
    if not llm:
        logger.error("LLM not initialized. Cannot create agents.")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] LLM not initialized. Cannot create agents.")
        return None
    
    # Initialize clinical agent first (needed for receptionist routing)
    logger.info("Initializing clinical agent...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Initializing clinical agent...")
    clinical_agent = ClinicalAgent(llm, vectorstore)
    
    # Initialize receptionist agent with reference to clinical agent
    logger.info("Initializing receptionist agent...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Initializing receptionist agent...")
    receptionist_agent = ReceptionistAgent(llm, clinical_agent=clinical_agent)
    
    logger.info("All agents initialized successfully")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] All agents initialized successfully")
    
    # Return agents dictionary
    return {
        "receptionist": receptionist_agent,
        "clinical": clinical_agent
    }


def main():
    """
    Main Streamlit application.
    """
    st.set_page_config(
        page_title="Post Discharge Medical AI Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    # Title
    st.title("🏥 Post Discharge Medical AI Assistant")
    st.markdown("**Your personal healthcare assistant for post-discharge care**")
    
    # Medical Disclaimers
    st.warning("⚠️ **Important Medical Disclaimers:**")
    st.markdown("""
    - **This is an AI assistant for educational purposes only**
    - **Always consult healthcare professionals for medical advice**
    - This assistant provides information only, not medical diagnosis or treatment recommendations
    - For emergencies, seek immediate medical attention or call emergency services (911/999)
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check API key
        if not settings.GEMINI_API_KEY:
            st.warning("⚠️ GEMINI_API_KEY not found. LLM functionality may be limited.")
        else:
            st.success("✅ API Key configured")
        
        st.markdown("---")
        st.markdown("### 📋 Features")
        st.markdown("""
        - Retrieve discharge reports
        - Answer medication questions
        - Explain discharge instructions
        - Provide medical information
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ Important Medical Disclaimers")
        st.warning("**This is an AI assistant for educational purposes only**")
        st.warning("**Always consult healthcare professionals for medical advice**")
        st.info("This assistant provides information only, not medical diagnosis. For emergencies, seek immediate medical attention.")
        
        st.markdown("---")
    
    # Initialize agents (cached)
    agents = initialize_agents()
    
    if not agents:
        st.error("❌ Failed to initialize agents. Please check your configuration and logs.")
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add initial greeting message
        st.session_state.messages.append({
            "role": "assistant",
            "content": INITIAL_GREETING,
            "metadata": {"action": "initial_greeting"}
        })
    if "current_patient" not in st.session_state:
        st.session_state.current_patient = None
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = agents["receptionist"]
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Clear conversation button in sidebar (after agents initialization)
    with st.sidebar:
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            # Add initial greeting message after clearing
            st.session_state.messages.append({
                "role": "assistant",
                "content": INITIAL_GREETING,
                "metadata": {"action": "initial_greeting"}
            })
            st.session_state.current_patient = None
            st.session_state.current_agent = agents["receptionist"]
            st.session_state.conversation_history = []
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message and message["metadata"]:
                metadata = message["metadata"]
                if metadata.get("patient_found"):
                    st.success("✅ Patient found in records")
                elif metadata.get("emergency"):
                    st.error("🚨 Emergency alert detected")
                
                # Display RAG sources if available
                citations = metadata.get("citations", [])
                if citations and len(citations) > 0:
                    st.markdown("---")
                    st.markdown("### 📚 RAG Sources")
                    for i, cite in enumerate(citations, 1):
                        citation_text = cite.get('citation', 'Unknown source') or 'Unknown source'
                        distance_score = cite.get('distance_score', 'N/A')
                        excerpt = cite.get('excerpt', '')
                        with st.expander(f"{i}. {citation_text}", expanded=False):
                            st.markdown(f"**Distance Score:** {distance_score}")
                            if excerpt:
                                st.markdown(f"**Excerpt:** {excerpt}")
                
                # Display web search sources if available
                web_sources = metadata.get("web_sources", [])
                if web_sources and len(web_sources) > 0:
                    st.markdown("---")
                    st.markdown("### 📚 Web Search Sources")
                    for i, source in enumerate(web_sources, 1):
                        title = source.get('title', 'Source') or 'Untitled Source'
                        url = source.get('url', '') or '#'
                        snippet = source.get('snippet', '')
                        with st.expander(f"{i}. {title}", expanded=False):
                            if url and url != '#':
                                st.markdown(f"**URL:** [{url}]({url})")
                            if snippet:
                                st.markdown(f"**Summary:** {snippet}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your post-discharge care..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Determine if routing to clinical agent is needed
        current_agent = st.session_state.current_agent
        agent_type = "receptionist" if current_agent == agents["receptionist"] else "clinical"
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{agent_type.upper()} AGENT] [INFO] Processing message: {prompt[:50]}...")
        
        # Use LLM-based routing if we're currently on receptionist agent
        patient_context = None
        if current_agent == agents["receptionist"] and hasattr(current_agent, 'should_route_to_clinical'):
            if current_agent.should_route_to_clinical(
                prompt, 
                patient_name=st.session_state.current_patient,
                conversation_history=st.session_state.conversation_history
            ):
                # Get patient context before routing
                if st.session_state.current_patient and hasattr(current_agent, 'get_patient_context'):
                    patient_context = current_agent.get_patient_context(st.session_state.current_patient)
                
                current_agent = agents["clinical"]
                st.session_state.current_agent = current_agent
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ROUTING] Routing from RECEPTIONIST AGENT to CLINICAL AGENT")
                # Add system message about routing
                with st.chat_message("assistant"):
                    st.info("🔄 Connecting you with the clinical specialist...")
        
        # Process message through agent
        # Wrap long-running operation with spinner OUTSIDE of chat_message to avoid blur
        response_text = ""
        metadata = {}
        with st.spinner("Processing..."):
            try:
                # Prepare invoke parameters
                invoke_params = {
                    "message": prompt,
                    "patient_name": st.session_state.current_patient,
                    "conversation_history": st.session_state.conversation_history
                }
                # Add patient_context if available (for clinical agent)
                if patient_context and current_agent == agents["clinical"]:
                    invoke_params["patient_context"] = patient_context
                
                response = current_agent.invoke(**invoke_params)
                
                # Extract response content
                if isinstance(response, dict):
                    response_text = response.get("response", str(response))
                    metadata = response.get("metadata", {})
                    
                    # Update patient name if found
                    if response.get("patient_name"):
                        st.session_state.current_patient = response.get("patient_name")
                else:
                    response_text = str(response)
                    metadata = {}
                
                # Add to conversation history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "metadata": metadata
                })
                
                # Update conversation history for context
                st.session_state.conversation_history.append({
                    "user": prompt,
                    "assistant": response_text
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                logger.error(f"Error processing message: {e}", exc_info=True)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Error processing message: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                response_text = error_msg
                metadata = {}
        
        # Display response AFTER processing is complete
        with st.chat_message("assistant"):
            if isinstance(response_text, str) and response_text.startswith("Sorry, I encountered an error"):
                st.error(response_text)
            else:
                st.markdown(response_text)
                
                # Display RAG sources if available
                citations = metadata.get("citations", [])
                if citations and len(citations) > 0:
                    st.markdown("---")
                    st.markdown("### 📚 RAG Sources")
                    for i, cite in enumerate(citations, 1):
                        citation_text = cite.get('citation', 'Unknown source') or 'Unknown source'
                        distance_score = cite.get('distance_score', 'N/A')
                        excerpt = cite.get('excerpt', '')
                        with st.expander(f"{i}. {citation_text}", expanded=False):
                            st.markdown(f"**Distance Score:** {distance_score}")
                            if excerpt:
                                st.markdown(f"**Excerpt:** {excerpt}")
                
                # Display web search sources if available
                web_sources = metadata.get("web_sources", [])
                if web_sources and len(web_sources) > 0:
                    st.markdown("---")
                    st.markdown("### 📚 Web Search Sources")
                    for i, source in enumerate(web_sources, 1):
                        title = source.get('title', 'Source') or 'Untitled Source'
                        url = source.get('url', '') or '#'
                        snippet = source.get('snippet', '')
                        with st.expander(f"{i}. {title}", expanded=False):
                            if url and url != '#':
                                st.markdown(f"**URL:** [{url}]({url})")
                            if snippet:
                                st.markdown(f"**Summary:** {snippet}")


if __name__ == "__main__":
    # Validate API key (only needed for LLM, not embeddings)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("Missing GEMINI_API_KEY! LLM functionality will be limited.")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] Missing GEMINI_API_KEY! LLM functionality will be limited.")
    
    main()
