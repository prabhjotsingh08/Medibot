"""
Receptionist Agent for Post Discharge Medical AI Assistant

This agent handles initial patient queries, routing, and basic information retrieval.
"""

from typing import Optional, Dict, Any, List
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain import hub
from tools.patient_report_tool import fetch_patient_report
from utils.logger_config import get_logger
import logfire
from datetime import datetime

logger = get_logger(__name__)


class ReceptionistAgent:
    """
    Receptionist agent that handles initial patient interactions.
    
    Responsibilities:
    - Patient lookup and identification
    - Basic information retrieval
    - Routing to clinical agent when needed
    - Appointment scheduling queries
    """
    
    def __init__(self, llm: BaseChatModel, clinical_agent=None):
        """
        Initialize the receptionist agent.
        
        Args:
            llm: Language model instance for the agent
            clinical_agent: Optional reference to clinical agent for routing
        """
        self.llm = llm
        self.agent = None
        self.tools = []
        self.clinical_agent = clinical_agent  # Reference to clinical agent for routing
        
        # Initialize tools
        self._initialize_tools()
        
        # Create agent with tools and system prompt
        self._create_agent()
    
    
    def _initialize_tools(self):
        """Initialize tools for the receptionist agent."""
        self.tools = [
            fetch_patient_report,
        ]
    
    
    def _create_agent(self):
        """Create the agent with system prompt and tools."""
        # Create agent using new LangChain 0.2+ API
        # create_react_agent requires a prompt parameter
        try:
            # Try to pull the default ReAct prompt from LangChain Hub
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful receptionist assistant for a post-discharge medical care system. Answer the following questions as best you can. You have access to the following tools:

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

Your role is to help patients identify themselves, look up their discharge reports, and route clinical questions to the appropriate agent. Be friendly, professional, and helpful."""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ])
            
            agent = create_react_agent(self.llm, self.tools, prompt=prompt)
            self.agent = AgentExecutor(agent=agent, tools=self.tools, verbose=False)
        except Exception as e:
            logger.warning(f"Failed to create agent with new API: {e}. Using LLM directly.")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [RECEPTIONIST AGENT] [WARNING] Failed to create agent: {e}")
            self.agent = None
    
    
    def invoke(self, message: str, patient_name: Optional[str] = None, 
               conversation_history: Optional[List] = None, **kwargs) -> Dict[str, Any]:
        """
        Process a message through the receptionist agent.
        Handles asking for patient name, fetching data, and routing.
        
        Args:
            message: User message
            patient_name: Optional patient name for context
            conversation_history: Optional conversation history
            **kwargs: Additional context
            
        Returns:
            Agent response with metadata
        """
        # First, check if routing to clinical agent is needed (using LLM)
        # This check should happen BEFORE patient name extraction to avoid
        # repeating discharge reports for clinical queries
        if self.clinical_agent:
            should_route = self.should_route_to_clinical(
                message, 
                patient_name=patient_name, 
                conversation_history=conversation_history
            )
            if should_route:
                logger.info(f"Routing to clinical agent for message: {message[:50]}...")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [RECEPTIONIST AGENT] [ROUTING] Routing to CLINICAL AGENT for: {message[:50]}...")
                # Get patient context if we have a name (use session state if available)
                patient_context = None
                if patient_name:
                    patient_context = self.get_patient_context(patient_name)
                elif conversation_history:
                    # Try to extract patient name from conversation history
                    for entry in reversed(conversation_history):
                        if isinstance(entry, dict) and entry.get("patient_name"):
                            patient_name = entry.get("patient_name")
                            patient_context = self.get_patient_context(patient_name)
                            break
                
                # Route to clinical agent
                return self.clinical_agent.invoke(
                    message=message,
                    patient_name=patient_name,
                    patient_context=patient_context,
                    conversation_history=conversation_history,
                    **kwargs
                )
        
        # Extract patient name from message if not provided (only if not routing to clinical)
        if not patient_name:
            patient_name = self._extract_patient_name(message)
        
        # If we have a patient name, try to fetch their data
        if patient_name:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [RECEPTIONIST AGENT] [TOOL] Calling fetch_patient_report for: {patient_name}")
            patient_context = self.get_patient_context(patient_name)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [RECEPTIONIST AGENT] [TOOL] fetch_patient_report completed")
            if patient_context:
                # Patient found - provide summary and ask follow-up
                response_text = self._format_patient_summary(patient_context)
                response_text += "\n\nHow can I help you today? You can ask about medications, discharge instructions, or any other questions."
                return {
                    "response": response_text,
                    "agent_type": "receptionist",
                    "patient_name": patient_name,
                    "metadata": {"patient_found": True}
                }
            else:
                # Patient not found
                return {
                    "response": f"I couldn't find a discharge report for '{patient_name}'. Please check the name and try again, or type your name if you haven't already.",
                    "agent_type": "receptionist",
                    "metadata": {"patient_found": False}
                }
        
        # Ask for patient name if not provided
        return {
            "response": "Hello! I'm here to help with your post-discharge care. Could you please tell me your name so I can look up your discharge report?",
            "agent_type": "receptionist",
            "metadata": {"action": "ask_name"}
        }
    
    def _extract_patient_name(self, message: str) -> Optional[str]:
        """Extract patient name from message using patterns."""
        from utils.helpers import extract_patient_name
        
        name = extract_patient_name(message)
        if name:
            logger.info(f"Extracted patient name: {name}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [RECEPTIONIST AGENT] [INFO] Extracted patient name: {name}")
        return name
    
    def _format_patient_summary(self, patient_context: Dict[str, Any]) -> str:
        """Format patient summary for display."""
        summary = f"Found your discharge report, {patient_context.get('patient_name', 'Patient')}!\n\n"
        summary += f"**Discharge Date:** {patient_context.get('discharge_date', 'N/A')}\n"
        summary += f"**Primary Diagnosis:** {patient_context.get('primary_diagnosis', 'N/A')}\n"
        summary += f"\n**Medications:**\n"
        for med in patient_context.get('medications', []):
            summary += f"  - {med}\n"
        summary += f"\n**Follow-up:** {patient_context.get('follow_up', 'N/A')}\n"
        return summary
    
    
    def should_route_to_clinical(self, message: str, patient_name: Optional[str] = None, 
                                  conversation_history: Optional[List] = None) -> bool:
        """
        Determine if a message should be routed to clinical agent using LLM.
        
        Args:
            message: User message
            patient_name: Optional patient name for context
            conversation_history: Optional conversation history
            
        Returns:
            True if should route to clinical agent
        """
        # Use LLM to determine if this is a clinical query
        routing_prompt = f"""You are a medical assistant router. Your task is to determine if a patient's message should be handled by a clinical specialist or the receptionist.

RECEPTIONIST handles:
- Patient identification ("I am John Smith")
- Basic information requests about discharge reports
- Appointment scheduling questions

CLINICAL SPECIALIST handles:
- Medical symptoms (nausea, vomiting, pain, fever, etc.)
- Medication questions and concerns
- Health condition questions
- Treatment-related questions
- Questions about what to do regarding health issues
- Any medical advice or guidance needs

Patient Message: "{message}"

Respond with ONLY one word: "CLINICAL" or "RECEPTIONIST"

Your response:"""

        try:
            response = self.llm.invoke(routing_prompt)
            if hasattr(response, 'content'):
                result = response.content.strip().upper()
            else:
                result = str(response).strip().upper()
            
            # Check if LLM says to route to clinical
            is_clinical = "CLINICAL" in result
            logger.info(f"LLM routing decision: {result} -> Route to clinical: {is_clinical}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [RECEPTIONIST AGENT] [ROUTING] LLM decision: {result} -> Route to clinical: {is_clinical}")
            return is_clinical
        except Exception as e:
            logger.error(f"Error in LLM routing decision: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [RECEPTIONIST AGENT] [ERROR] Error in LLM routing: {e}")
            # Fallback: if message seems like it could be clinical (not just name), route to clinical
            # But avoid routing simple name statements
            message_lower = message.lower().strip()
            # Check if it's likely just a name statement
            if any(phrase in message_lower for phrase in ["i am", "i'm", "my name is", "name is"]):
                return False
            # Otherwise, if message is longer or seems medical, route to clinical
            if len(message) > 20:
                return True
            return False
    
    
    def get_patient_context(self, patient_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve patient context for conversation.
        
        Args:
            patient_name: Patient name
            
        Returns:
            Patient context dictionary
        """
        from utils.helpers import get_patient_by_name
        return get_patient_by_name(patient_name)
    
    def set_clinical_agent(self, clinical_agent):
        """Set reference to clinical agent for routing."""
        self.clinical_agent = clinical_agent

