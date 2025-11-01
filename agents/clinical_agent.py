"""
Clinical Agent for Post Discharge Medical AI Assistant

This agent handles clinical questions, medical advice, and complex medical queries.
Uses RAG to retrieve relevant medical information from knowledge base.
"""

from typing import Optional, Dict, Any, List
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from rag.vectorstore import get_vectorstore
from tools.rag_retrieval_tool import rag_retrieval, retrieve_medical_knowledge
from tools.web_search_tool import web_search
from config.settings import settings
from utils.logger_config import get_logger
import logfire
from datetime import datetime

logger = get_logger(__name__)


class ClinicalAgent:
    """
    Clinical agent that handles medical questions and clinical queries.
    
    Responsibilities:
    - Answer clinical questions using RAG
    - Provide medication information and guidance
    - Explain discharge instructions
    - Analyze symptoms (non-diagnostic)
    - Retrieve relevant medical knowledge from vectorstore
    """
    
    def __init__(self, llm: BaseChatModel, vectorstore: Optional[Chroma] = None):
        """
        Initialize the clinical agent.
        
        Args:
            llm: Language model instance for the agent
            vectorstore: ChromaDB vectorstore instance for RAG
        """
        self.llm = llm
        self.vectorstore = vectorstore or get_vectorstore()
        self.agent = None
        self.tools = []
        
        # Initialize tools
        self._initialize_tools()
        
        # Create agent with tools and system prompt
        self._create_agent()
    
    
    def _initialize_tools(self):
        """Initialize tools for the clinical agent."""
        self.tools = [
            rag_retrieval,  # RAG retrieval from ChromaDB
            web_search,  # Web search for additional info
        ]
    
    
    def _create_agent(self):
        """Create the agent with system prompt and tools."""
        # Create agent using new LangChain 0.2+ API
        # create_react_agent requires a prompt parameter
        try:
            # Try to pull the default ReAct prompt from LangChain Hub
            try:
                prompt = hub.pull("hwchase17/react")
            except Exception:
                # If hub pull fails, create a basic ReAct-style prompt template
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful clinical assistant. Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Always emphasize that this is information, not medical diagnosis. Be clear, empathetic, and professional."""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ])
            
            agent = create_react_agent(self.llm, self.tools, prompt=prompt)
            self.agent = AgentExecutor(agent=agent, tools=self.tools, verbose=False)
        except Exception as e:
            logger.warning(f"Failed to create agent with new API: {e}. Using LLM directly.")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [WARNING] Failed to create agent: {e}")
            self.agent = None
    
    
    def invoke(self, message: str, patient_name: Optional[str] = None, 
               patient_context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Process a message through the clinical agent.
        
        Args:
            message: User message
            patient_name: Optional patient name for context
            patient_context: Optional patient context (discharge report, etc.)
            **kwargs: Additional context
            
        Returns:
            Agent response with metadata and citations
        """
        # Check for emergency keywords
        if self.check_emergency_keywords(message):
            logger.warning(f"Emergency keywords detected in message: {message[:50]}...")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [WARNING] Emergency keywords detected: {message[:50]}...")
            return {
                "response": "🚨 EMERGENCY ALERT: Please seek immediate medical attention or call emergency services (911/999). This assistant provides information only and cannot diagnose or treat emergencies.",
                "agent_type": "clinical",
                "emergency": True,
                "metadata": {"action": "emergency_alert"}
            }
        
        logger.info(f"Processing clinical question: {message[:50]}...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [INFO] Processing clinical question: {message[:50]}...")
        
        # Use RAG to answer clinical questions
        rag_result = self.rag_answer(message, patient_context)
        
        # Format response with citations
        answer = rag_result.get("answer", "")
        citations = rag_result.get("citations", [])
        web_sources = rag_result.get("web_sources", [])
        
        # Citations and web sources are displayed separately in UI (not inline)
        
        return {
            "response": answer,
            "agent_type": "clinical",
            "patient_name": patient_name,
            "metadata": {
                "used_rag": rag_result.get("has_rag_context", False),
                "sources_count": len(citations),
                "citations": citations,
                "web_sources": web_sources
            }
        }
    
    
    def rag_answer(self, query: str, patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Answer clinical question using RAG with citations.
        
        Args:
            query: User query
            patient_context: Optional patient context to enhance query
            
        Returns:
            Dict with answer and citations
        """
        # Enhance query with patient context if available
        enhanced_query = query
        if patient_context:
            diagnosis = patient_context.get('primary_diagnosis', '')
            medications = ', '.join(patient_context.get('medications', []))
            context_str = f"Patient context: Diagnosis - {diagnosis}, Medications - {medications}. "
            enhanced_query = context_str + query
        
        # Retrieve relevant medical knowledge from ChromaDB
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [TOOL] Calling RAG retrieval: {enhanced_query[:50]}...")
        retrieved_docs = retrieve_medical_knowledge(enhanced_query, k=5)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [TOOL] RAG retrieval completed: {len(retrieved_docs)} results")
        
        # Check relevance threshold using distance_score directly
        # Compare query with 5 closest chunks - if ALL 5 have distance_score > 0.5, use RAG
        # Otherwise, if at least one has distance_score <= 0.5, use web search
        RAG_DISTANCE_THRESHOLD = 0.8
        use_rag = False
        if retrieved_docs:
            # Collect distance scores for all chunks
            distance_scores = []
            chunk_details = []  # Store details for logfire logging
            
            for i, doc in enumerate(retrieved_docs):
                distance_score = float(doc.get('score', float('inf')))
                distance_scores.append(distance_score)
                
                # Prepare chunk details for logfire
                text = doc.get('text', doc.get('page_content', ''))
                citation = doc.get('citation', 'Unknown source')
                metadata = doc.get('metadata', {})
                
                chunk_details.append({
                    "chunk_index": i + 1,
                    "distance_score": round(distance_score, 4),
                    "citation": citation,
                    "source": metadata.get('source', 'Unknown'),
                    "page": metadata.get('page', 'N/A'),
                    "content_preview": text[:300] + "..." if len(text) > 300 else text,
                    "content_length": len(text),
                    "above_threshold": distance_score > RAG_DISTANCE_THRESHOLD
                })
            
            # Log all 5 closest chunks to logfire for observability
            logfire.info(
                "RAG retrieval: 5 closest chunks analyzed",
                query=enhanced_query[:200],
                total_chunks=len(retrieved_docs),
                chunks=chunk_details,
                threshold=RAG_DISTANCE_THRESHOLD,
                distance_scores=[round(s, 4) for s in distance_scores]
            )
            
            # Check if ALL 5 closest chunks have distance_score > 0.5
            all_above_threshold = all(score > RAG_DISTANCE_THRESHOLD for score in distance_scores)
            
            if all_above_threshold:
                use_rag = True
                logger.info(f"All {len(distance_scores)} closest chunks have distance_score > {RAG_DISTANCE_THRESHOLD} (scores: {[f'{s:.3f}' for s in distance_scores]}), using RAG")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [INFO] All {len(distance_scores)} closest chunks have distance_score > {RAG_DISTANCE_THRESHOLD} (scores: {[f'{s:.3f}' for s in distance_scores]}), using RAG")
                
                # Log decision to logfire
                logfire.info(
                    "RAG threshold check: All chunks above threshold, using RAG",
                    query=enhanced_query[:200],
                    decision="rag",
                    chunks_analyzed=len(retrieved_docs),
                    all_distance_scores=[round(s, 4) for s in distance_scores],
                    threshold=RAG_DISTANCE_THRESHOLD
                )
            else:
                logger.info(f"At least one chunk has distance_score <= {RAG_DISTANCE_THRESHOLD} (scores: {[f'{s:.3f}' for s in distance_scores]}), switching to web search")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [INFO] At least one chunk has distance_score <= {RAG_DISTANCE_THRESHOLD} (scores: {[f'{s:.3f}' for s in distance_scores]}), using web search instead")
                
                # Log decision to logfire
                logfire.info(
                    "RAG threshold check: At least one chunk below threshold, using web search",
                    query=enhanced_query[:200],
                    decision="web_search",
                    chunks_analyzed=len(retrieved_docs),
                    all_distance_scores=[round(s, 4) for s in distance_scores],
                    threshold=RAG_DISTANCE_THRESHOLD,
                    chunks_below_threshold=sum(1 for s in distance_scores if s <= RAG_DISTANCE_THRESHOLD)
                )
        
        # Format retrieved context with citations
        context_parts = []
        citations = []
        web_sources = []  # Initialize web_sources at the start
        
        if retrieved_docs and use_rag:
            for i, doc in enumerate(retrieved_docs, 1):
                text = doc.get('text', doc.get('page_content', ''))
                citation = doc.get('citation', 'Unknown source')
                # Use distance_score directly
                distance_score = float(doc.get('score', 0.0))
                
                context_parts.append(f"[Reference {i}] {text}")
                citations.append({
                    "reference": i,
                    "citation": citation,
                    "distance_score": round(distance_score, 4),
                    "excerpt": text[:200] + "..." if len(text) > 200 else text
                })
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        # Use LLM to generate answer (minimize Gemini API calls)
        # If RAG results are below threshold or no RAG results, use web search
        if context and use_rag:
            # Build comprehensive prompt with RAG context
            prompt = f"""You are a clinical assistant providing medical information based on retrieved knowledge.

Retrieved Medical Knowledge:
{context}

Patient Question: {query}

Instructions:
1. Answer the question using the retrieved medical knowledge above.
2. Reference specific sources using [Reference X] notation.
3. If the retrieved information doesn't fully answer the question, acknowledge limitations.
4. Always emphasize this is information, not medical diagnosis.
5. If patient context is provided, incorporate it into your answer where relevant.
6. Be clear, empathetic, and professional.

Your answer:"""
        else:
            # No RAG context or RAG score < 0.5 - use web search as fallback
            logger.info("RAG score below threshold or no RAG context found, using web search")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [TOOL] RAG score below threshold, calling web_search tool")
            web_results = ""
            
            if settings.USE_WEB_SEARCH:
                # Call SerpAPI web search and get structured results with fetched content
                try:
                    from tools.web_search_tool import _serpapi_search
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [TOOL] Calling web_search: {query[:50]}...")
                    # Get structured results with fetched content from URLs
                    structured_results = _serpapi_search(query, max_results=5, return_structured=True, fetch_content=True)
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [TOOL] web_search completed - found {len(structured_results) if structured_results else 0} results")
                    
                    if structured_results:
                        # Format for prompt with full content
                        web_results = f"Web search results for '{query}':\n\n"
                        for i, result in enumerate(structured_results, 1):
                            source_num = f"[Source {i}]"
                            
                            web_results += f"{source_num}\n"
                            web_results += f"Title: {result.get('title', '')}\n"
                            web_results += f"URL: {result.get('url', '')}\n"
                            
                            # Use fetched content if available, otherwise use snippet
                            content = result.get('content', result.get('snippet', ''))
                            if content:
                                web_results += f"Content: {content}\n"
                            
                            web_results += "\n"
                        
                        # Collect sources for display
                        web_sources = [
                            {
                                "title": result.get("title", ""),
                                "url": result.get("url", ""),
                                "snippet": result.get("snippet", ""),
                                "content_preview": result.get("content", result.get("snippet", ""))[:200] if result.get("content") else result.get("snippet", "")
                            }
                            for result in structured_results
                        ]
                    else:
                        web_results = "No web search results found."
                        web_sources = []  # Ensure web_sources is initialized
                except Exception as e:
                    logger.error(f"Error in web search: {e}")
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [ERROR] Error in web search: {e}")
                    web_results = "Web search temporarily unavailable."
                    web_sources = []  # Ensure web_sources is initialized
            else:
                web_results = "Web search is disabled."
                web_sources = []  # Ensure web_sources is initialized
            
            prompt = f"""You are a clinical assistant. Use the following web search results to answer the question.

Web Search Results:
{web_results}

Patient Question: {query}

Instructions:
1. Answer the question comprehensively using the content from the web search results above.
2. When referencing information, cite the source using [Source X] notation where X is the source number.
3. Acknowledge that this information should be verified with healthcare providers.
4. Always emphasize this is information, not medical diagnosis.
5. Be clear, empathetic, and professional.
6. Do NOT use any tools - only use the web search results provided above.

Your answer (with source citations using [Source X] format):"""
        
        # Generate response using LLM directly (bypass agent executor to prevent tool calls)
        # Use direct LLM call for both RAG and web search since we already have the context in the prompt
        logger.info("Using direct LLM call (bypassing agent executor) to prevent tool calls and parsing errors")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [INFO] Using direct LLM call (bypassing agent executor)")
        answer = self._generate_answer_direct(prompt)
        
        return {
            "answer": answer,
            "citations": citations,
            "sources_count": len(citations),
            "has_rag_context": len(context_parts) > 0,
            "web_sources": web_sources
        }
    
    def _generate_answer_direct(self, prompt: str) -> str:
        """Generate answer directly using LLM, bypassing agent executor to prevent tool calls."""
        try:
            # Use LLM directly without agent executor - this prevents any tool calls and parsing errors
            logger.info("Calling LLM directly (no agent executor)")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [INFO] Calling LLM directly (no agent executor)")
            
            # Create a simple HumanMessage instead of raw prompt for better compatibility
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                # Try to extract content from response object
                return str(response)
        except ValueError as ve:
            # Handle parsing errors specifically
            error_msg = str(ve)
            logger.error(f"Output parsing error in LLM response: {error_msg}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [ERROR] Output parsing error: {error_msg[:200]}")
            
            # Try a fallback - invoke with just the prompt as string
            try:
                logger.info("Retrying with direct prompt string")
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    return response.content
                return str(response)
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return "I apologize, but I encountered an error while generating a response. Please try again or consult your healthcare provider."
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [ERROR] Error generating answer: {e}")
            logfire.error("Error generating answer", error=str(e), error_type=type(e).__name__)
            return "I apologize, but I encountered an error while generating a response. Please try again or consult your healthcare provider."
    
    
    def check_emergency_keywords(self, message: str) -> bool:
        """
        Check if message contains emergency keywords.
        
        Args:
            message: User message
            
        Returns:
            True if emergency keywords detected
        """
        emergency_keywords = [
            "chest pain", "difficulty breathing", "severe pain",
            "unconscious", "severe bleeding", "stroke", "heart attack"
        ]
        message_lower = message.lower()
        # Note: Could implement more sophisticated detection in future
        return any(keyword in message_lower for keyword in emergency_keywords)

