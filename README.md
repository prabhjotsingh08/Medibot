# Post Discharge Medical AI Assistant

A multi-agent AI system designed to assist patients with post-discharge medical care. The system features intelligent agent routing, RAG (Retrieval Augmented Generation) for medical knowledge retrieval, and comprehensive patient report management.

## 🏥 Overview

This application provides a conversational AI assistant that helps patients with:
- Retrieving and understanding their discharge reports
- Answering medication-related questions
- Explaining discharge instructions
- Providing medical information using RAG from clinical knowledge bases
- Web search for up-to-date medical information

## ✨ Features

### Multi-Agent Architecture
- **Receptionist Agent**: Handles initial patient queries, patient lookup, and intelligent routing
- **Clinical Agent**: Manages complex clinical questions, medication guidance, and medical knowledge retrieval

### Core Capabilities
- 🤖 **Intelligent Agent Routing**: Automatically routes queries to the appropriate agent based on context
- 📄 **Patient Report Retrieval**: Fetches and explains patient discharge reports
- 💊 **Medication Information**: Answers questions about medications and dosages
- 📚 **RAG (Retrieval Augmented Generation)**: Retrieves relevant medical information from knowledge base
- 🔍 **Web Search Integration**: Searches the web for up-to-date medical information (optional)
- 💬 **Streamlit UI**: User-friendly conversational interface
- 🔌 **FastAPI Backend**: RESTful API for programmatic access

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Main.py │
    └────┬────┘
         │
    ┌────▼──────────────────┐
    │ Receptionist Agent    │◄──┐
    └────┬──────────────────┘  │
         │ Routes to           │
    ┌────▼──────────────────┐  │
    │  Clinical Agent       │───┘
    └────┬──────────────────┘
         │
    ┌────▼──────────────────────────┐
    │          Tools                │
    ├───────────────────────────────┤
    │ • Patient Report Tool         │
    │ • RAG Retrieval Tool          │
    │ • Web Search Tool             │
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │      ChromaDB Vectorstore      │
    │  (Medical Knowledge Base)      │
    └────────────────────────────────┘
```

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key (for LLM functionality)
- (Optional) SerpAPI key (for web search functionality)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd datasmith_assignment
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   # Required
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Optional - Web Search
   SERPAPI_API_KEY=your_serpapi_key_here
   USE_WEB_SEARCH=true
   
   # Optional - Configuration
   LLM_MODEL=gemini-2.5-flash
   LLM_TEMPERATURE=0.0
   EMBEDDING_PROVIDER=huggingface
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   
   # Optional - FastAPI Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   LOG_LEVEL=INFO
   ```

## 🔧 Configuration

The application uses Pydantic settings for configuration. Key settings can be found in `config/settings.py`:

- **LLM Configuration**: Model selection, temperature settings
- **Embedding Configuration**: Embedding provider and model selection (defaults to free HuggingFace)
- **RAG Configuration**: Chunk size, overlap settings
- **ChromaDB Configuration**: Vector database persistence settings
- **API Configuration**: FastAPI host, port, CORS settings

## 📖 Usage

### Running the Streamlit Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

### Using the FastAPI Server

```bash
# Run the FastAPI server
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# Or using uvicorn directly
uvicorn api.server:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the FastAPI server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📡 API Endpoints

### POST `/api/v1/chat`
Send a message to the medical assistant.

**Request:**
```json
{
  "message": "What medications am I taking?",
  "patient_name": "John Doe",
  "agent_type": "receptionist",
  "conversation_id": "optional-id"
}
```

**Response:**
```json
{
  "response": "Based on your discharge report...",
  "agent_type": "clinical",
  "conversation_id": "optional-id",
  "metadata": {
    "citations": [...],
    "web_sources": [...]
  }
}
```

### POST `/api/v1/patient-report`
Retrieve a patient's discharge report.

**Request:**
```json
{
  "patient_name": "John Doe"
}
```

### GET `/api/v1/agents`
List available agents and their capabilities.

### GET `/api/v1/conversations/{conversation_id}`
Retrieve conversation history.

### GET `/health`
Health check endpoint.

## 📁 Project Structure

```
datasmith_assignment/
├── agents/                    # Multi-agent system
│   ├── clinical_agent.py     # Clinical specialist agent
│   └── receptionist_agent.py  # Receptionist/routing agent
├── api/                       # FastAPI backend
│   ├── models.py             # Pydantic models
│   ├── routes.py             # API route handlers
│   └── server.py             # FastAPI application
├── config/                    # Configuration
│   └── settings.py           # Application settings
├── data/                      # Data files
│   ├── nephrology/           # Medical PDF documents
│   └── patient_reports.json  # Patient discharge reports
├── rag/                      # RAG implementation
│   ├── chunking.py           # Document chunking
│   ├── embeddings.py         # Embedding model setup
│   ├── loader.py             # PDF/document loading
│   └── vectorstore.py        # ChromaDB vectorstore
├── tools/                     # Agent tools
│   ├── patient_report_tool.py # Patient report retrieval
│   ├── rag_retrieval_tool.py  # RAG knowledge retrieval
│   └── web_search_tool.py    # Web search integration
├── utils/                     # Utilities
│   ├── helpers.py            # Helper functions
│   └── logger_config.py      # Logging configuration
├── chroma_db/                 # ChromaDB persistent storage
├── logs/                      # Application logs
├── main.py                    # Streamlit application entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔑 Key Components

### Agents

#### ReceptionistAgent
- Handles initial patient identification
- Retrieves patient discharge reports
- Routes to Clinical Agent for complex queries
- Manages conversation flow

#### ClinicalAgent
- Answers clinical and medication questions
- Uses RAG to retrieve medical knowledge
- Provides web search capabilities
- Explains discharge instructions

### Tools

- **Patient Report Tool**: Fetches patient discharge reports from JSON database
- **RAG Retrieval Tool**: Queries ChromaDB vectorstore for relevant medical information
- **Web Search Tool**: Searches the web for up-to-date medical information (via SerpAPI)

### RAG System

The RAG system uses:
- **ChromaDB** for vector storage
- **HuggingFace embeddings** (all-MiniLM-L6-v2) by default
- PDF documents stored in `data/nephrology/` are automatically processed and indexed

## ⚠️ Medical Disclaimers

**Important**: This is an AI assistant for educational purposes only.

- Always consult healthcare professionals for medical advice
- This assistant provides information only, not medical diagnosis or treatment recommendations
- For emergencies, seek immediate medical attention or call emergency services (911/999)
- The system is not a replacement for professional medical care

## 🛠️ Development

### Logging

The application uses structured logging with `logfire`. Logs are written to:
- Console output
- Log files in `logs/` directory

### Vectorstore Initialization

On first run, the application will automatically:
1. Check for existing ChromaDB vectorstore
2. If not found, process PDF documents in `data/nephrology/`
3. Build and persist the vectorstore for future use

## 📝 Dependencies

Key dependencies include:
- `streamlit` - Web UI framework
- `langchain` - LLM framework and agent orchestration
- `langchain-google-genai` - Google Gemini integration
- `chromadb` - Vector database
- `sentence-transformers` - Embedding models
- `fastapi` - REST API framework
- `pydantic` - Data validation

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- Built for educational and demonstration purposes
- Uses Google Gemini for LLM capabilities
- HuggingFace for free embedding models
- ChromaDB for vector storage

---

**Note**: This application requires API keys for full functionality. Make sure to configure your `.env` file before running.
