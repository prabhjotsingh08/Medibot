"""
Clinical Agent for Post Discharge Medical AI Assistant

This agent handles clinical questions, medical advice, and complex medical queries.
Uses RAG or Web Search tools based on query relevance.
"""

from typing import Optional, Dict, Any, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from rag.vectorstore import get_vectorstore
from tools.rag_retrieval_tool import retrieve_medical_knowledge
from tools.web_search_tool import _serpapi_search
from config.settings import settings
from utils.logger_config import get_logger
import logfire
from datetime import datetime

logger = get_logger(__name__)


class ClinicalAgent:
    """
    Clinical agent that handles medical questions and clinical queries.
    
    Responsibilities:
    - Answer clinical questions using RAG or Web Search
    - Provide medication information and guidance
    - Explain discharge instructions
    - Analyze symptoms (non-diagnostic)
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
        self.RAG_DISTANCE_THRESHOLD = 1.8
    
    
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
        logger.info(f"Processing clinical question: {message[:50]}...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [INFO] Processing clinical question: {message[:50]}...")
        
        # Get answer using RAG or Web Search
        result = self._get_answer(message, patient_context)
        
        return {
            "response": result["answer"],
            "agent_type": "clinical",
            "patient_name": patient_name,
            "metadata": {
                "tool_used": result["tool_used"],
                "used_rag": result["tool_used"] == "rag",
                "sources_count": len(result.get("citations", [])),
                "citations": result.get("citations", []),
                "web_sources": result.get("web_sources", [])
            }
        }
    
    
    def _get_answer(self, query: str, patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get answer using RAG or Web Search tool based on relevance.
        
        Args:
            query: User query
            patient_context: Optional patient context to enhance query
            
        Returns:
            Dict with answer, tool_used, citations/web_sources
        """
        # Build enhanced query with patient context
        enhanced_query = self._build_enhanced_query(query, patient_context)
        
        # Retrieve RAG chunks
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [TOOL] Calling RAG retrieval: {enhanced_query[:50]}...")
        retrieved_docs = retrieve_medical_knowledge(enhanced_query, k=5)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [TOOL] RAG retrieval completed: {len(retrieved_docs)} results")
        
        # Decide which tool to use based on query type and RAG relevance
        tool_decision = self._decide_tool(query, retrieved_docs)
        
        if tool_decision == "rag":
            return self._answer_with_rag(query, enhanced_query, retrieved_docs, patient_context)
        else:
            return self._answer_with_web_search(query, enhanced_query, patient_context)
    
    
    def _build_enhanced_query(self, query: str, patient_context: Optional[Dict[str, Any]] = None) -> str:
        """Build enhanced query with patient context."""
        if not patient_context:
            return query
        
        diagnosis = patient_context.get('primary_diagnosis', '')
        medications = ', '.join(patient_context.get('medications', []))
        
        if diagnosis or medications:
            context_str = f"Patient context: Diagnosis - {diagnosis}, Medications - {medications}. "
            return context_str + query
        
        return query
    
    
    def _decide_tool(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Decide whether to use RAG or Web Search tool.
        
        Decision logic:
        1. If query is research/non-medical -> Web Search
        2. If no RAG docs retrieved -> Web Search
        3. If any RAG chunk has distance_score > threshold -> Web Search
        4. Otherwise -> RAG
        
        Args:
            query: Original user query
            retrieved_docs: Retrieved RAG documents
            
        Returns:
            "rag" or "web_search"
        """
        # Check if query is research or non-medical
        if self._is_research_or_non_medical_query(query):
            logger.info(f"Query detected as research/non-medical, using web search")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [INFO] Query detected as research/non-medical, using web search")
            logfire.info(
                "Tool decision: Research/non-medical query detected",
                query=query[:200],
                decision="web_search",
                reason="research_or_non_medical"
            )
            return "web_search"
        
        # Check if RAG docs were retrieved
        if not retrieved_docs:
            logger.info("No RAG documents retrieved, using web search")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [INFO] No RAG documents retrieved, using web search")
            logfire.info(
                "Tool decision: No RAG documents found",
                query=query[:200],
                decision="web_search",
                reason="no_rag_docs"
            )
            return "web_search"
        
        # Check RAG relevance scores
        distance_scores = []
        chunk_details = []
        
        for i, doc in enumerate(retrieved_docs):
            distance_score = float(doc.get('score', float('inf')))
            distance_scores.append(distance_score)
            
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
                "above_threshold": distance_score > self.RAG_DISTANCE_THRESHOLD
            })
        
        # Log all chunks for observability
        logfire.info(
            "RAG retrieval: 5 closest chunks analyzed",
            query=query[:200],
            total_chunks=len(retrieved_docs),
            chunks=chunk_details,
            threshold=self.RAG_DISTANCE_THRESHOLD,
            distance_scores=[round(s, 4) for s in distance_scores]
        )
        
        # Check if any chunk exceeds threshold
        if any(score > self.RAG_DISTANCE_THRESHOLD for score in distance_scores):
            logger.info(f"At least one chunk has distance_score > {self.RAG_DISTANCE_THRESHOLD} (scores: {[f'{s:.3f}' for s in distance_scores]}), using web search")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [INFO] At least one chunk above threshold, using web search")
            
            logfire.info(
                "Tool decision: RAG chunks above threshold",
                query=query[:200],
                decision="web_search",
                chunks_analyzed=len(retrieved_docs),
                all_distance_scores=[round(s, 4) for s in distance_scores],
                threshold=self.RAG_DISTANCE_THRESHOLD,
                chunks_above_threshold=sum(1 for s in distance_scores if s > self.RAG_DISTANCE_THRESHOLD)
            )
            return "web_search"
        
        # All chunks are relevant, use RAG
        logger.info(f"All chunks have distance_score <= {self.RAG_DISTANCE_THRESHOLD} (scores: {[f'{s:.3f}' for s in distance_scores]}), using RAG")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [INFO] All chunks below threshold, using RAG")
        
        logfire.info(
            "Tool decision: RAG chunks below threshold",
            query=query[:200],
            decision="rag",
            chunks_analyzed=len(retrieved_docs),
            all_distance_scores=[round(s, 4) for s in distance_scores],
            threshold=self.RAG_DISTANCE_THRESHOLD
        )
        return "rag"
    
    
    def _answer_with_rag(self, original_query: str, enhanced_query: str, 
                         retrieved_docs: List[Dict[str, Any]], 
                         patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate answer using RAG tool.
        
        Args:
            original_query: Original user query
            enhanced_query: Query enhanced with patient context
            retrieved_docs: Retrieved RAG documents
            patient_context: Optional patient context
            
        Returns:
            Dict with answer, tool_used="rag", citations
        """
        # Format RAG context and citations
        context_parts = []
        citations = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            text = doc.get('text', doc.get('page_content', ''))
            citation = doc.get('citation', 'Unknown source')
            distance_score = float(doc.get('score', 0.0))
            
            context_parts.append(f"[Reference {i}] {text}")
            citations.append({
                "reference": i,
                "citation": citation,
                "distance_score": round(distance_score, 4),
                "excerpt": text
            })
        
        rag_context = "\n\n".join(context_parts)
        
        # Build prompt with query, patient context, and RAG chunks
        prompt = self._build_rag_prompt(original_query, rag_context, patient_context)
        
        # Generate answer
        answer = self._generate_llm_response(prompt)
        
        return {
            "answer": answer,
            "tool_used": "rag",
            "citations": citations,
            "web_sources": []
        }
    
    
    def _answer_with_web_search(self, original_query: str, enhanced_query: str,
                                patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate answer using Web Search tool.
        
        Args:
            original_query: Original user query
            enhanced_query: Query enhanced with patient context
            patient_context: Optional patient context
            
        Returns:
            Dict with answer, tool_used="web_search", web_sources
        """
        web_sources = []
        web_results_text = ""
        
        if not settings.USE_WEB_SEARCH:
            logger.warning("Web search is disabled in settings")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [WARNING] Web search is disabled")
            web_results_text = "Web search is disabled."
        else:
            try:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [TOOL] Calling web_search: {original_query[:50]}...")
                
                # Call web search with content fetching
                structured_results = _serpapi_search(
                    original_query, 
                    max_results=5, 
                    return_structured=True, 
                    fetch_content=True
                )
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [TOOL] web_search completed - found {len(structured_results) if structured_results else 0} results")
                
                if structured_results:
                    # Format web results for prompt
                    web_results_text = f"Web search results for '{original_query}':\n\n"
                    for i, result in enumerate(structured_results, 1):
                        web_results_text += f"[Source {i}]\n"
                        web_results_text += f"Title: {result.get('title', '')}\n"
                        web_results_text += f"URL: {result.get('url', '')}\n"
                        
                        content = result.get('content', result.get('snippet', ''))
                        if content:
                            web_results_text += f"Content: {content}\n"
                        web_results_text += "\n"
                    
                    # Collect sources for metadata
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
                    web_results_text = "No web search results found."
                    
            except Exception as e:
                logger.error(f"Error in web search: {e}")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [ERROR] Error in web search: {e}")
                web_results_text = "Web search temporarily unavailable."
        
        # Build prompt with query, patient context, and web results
        prompt = self._build_web_search_prompt(original_query, web_results_text, patient_context)
        
        # Generate answer
        answer = self._generate_llm_response(prompt)
        
        return {
            "answer": answer,
            "tool_used": "web_search",
            "citations": [],
            "web_sources": web_sources
        }
    
    
    def _build_rag_prompt(self, query: str, rag_context: str, 
                         patient_context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for RAG-based answer."""
        patient_info = ""
        if patient_context:
            diagnosis = patient_context.get('primary_diagnosis', '')
            medications = ', '.join(patient_context.get('medications', []))
            if diagnosis or medications:
                patient_info = f"\nPatient Context:\n- Diagnosis: {diagnosis}\n- Medications: {medications}\n"
        
        return f"""You are a clinical assistant providing medical information based on retrieved knowledge.

Retrieved Medical Knowledge:
{rag_context}
{patient_info}
Patient Question: {query}

Instructions:
1. Answer the question using the retrieved medical knowledge above.
2. Reference specific sources using [Reference X] notation.
3. If patient context is provided, incorporate it into your answer where relevant.
4. If the retrieved information doesn't fully answer the question, acknowledge limitations.
5. Always emphasize this is information, not medical diagnosis.
6. Be clear, empathetic, and professional.

Your answer:"""
    
    
    def _build_web_search_prompt(self, query: str, web_results: str,
                                 patient_context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for web search-based answer."""
        patient_info = ""
        if patient_context:
            diagnosis = patient_context.get('primary_diagnosis', '')
            medications = ', '.join(patient_context.get('medications', []))
            if diagnosis or medications:
                patient_info = f"\nPatient Context:\n- Diagnosis: {diagnosis}\n- Medications: {medications}\n"
        
        return f"""You are a clinical assistant. Use the following web search results to answer the question.

Web Search Results:
{web_results}
{patient_info}
Patient Question: {query}

Instructions:
1. Answer the question comprehensively using the content from the web search results above.
2. When referencing information, cite the source using [Source X] notation.
3. If patient context is provided, incorporate it into your answer where relevant.
4. Acknowledge that this information should be verified with healthcare providers.
5. Always emphasize this is information, not medical diagnosis.
6. Be clear, empathetic, and professional.

Your answer (with source citations using [Source X] format):"""
    
    
    def _generate_llm_response(self, prompt: str) -> str:
        """
        Generate response using LLM directly (no agent executor).
        
        Args:
            prompt: Formatted prompt with context
            
        Returns:
            Generated answer string
        """
        try:
            logger.info("Calling LLM directly")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [INFO] Calling LLM directly")
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CLINICAL AGENT] [ERROR] Error generating answer: {e}")
            logfire.error("Error generating answer", error=str(e), error_type=type(e).__name__)
            return "I apologize, but I encountered an error while generating a response. Please try again or consult your healthcare provider."
    
    
    def _is_research_or_non_medical_query(self, query: str) -> bool:
        """
        Check if query is about research or unrelated to medical science.
        
        Args:
            query: User query
            
        Returns:
            True if query is about research or non-medical topics
        """
        query_lower = query.lower()
        
        # Research-related keywords
        research_keywords = [
            "research", "study", "studies", "clinical trial", "clinical trials",
            "publication", "paper", "journal", "article", "findings",
            "experiment", "experimental", "hypothesis", "methodology",
            "data analysis", "statistical analysis", "meta-analysis",
            "systematic review", "literature review", "peer review"
        ]
        
        # Non-medical topics that should use web search
        non_medical_keywords = [
            "cooking", "recipe", "weather", "sports", "entertainment",
            "movie", "music", "travel", "vacation", "hotel", "restaurant",
            "shopping", "fashion", "technology news", "politics",
            "stock market", "cryptocurrency", "bitcoin", "investment"
        ]
        
        has_research_keyword = any(keyword in query_lower for keyword in research_keywords)
        has_non_medical_keyword = any(keyword in query_lower for keyword in non_medical_keywords)
        
        return has_research_keyword or has_non_medical_keyword