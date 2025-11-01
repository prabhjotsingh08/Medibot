"""
Web Search Tool using SerpAPI

This tool provides web search capabilities for the medical assistant.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from langchain.tools import tool
from config.settings import settings
from utils.logger_config import get_logger
import logfire
from datetime import datetime
import httpx
import re
from html import unescape

logger = get_logger(__name__)


def _fetch_url_content(url: str, timeout: int = 10, max_length: int = 10000) -> str:
    """
    Fetch and extract text content from a URL.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        max_length: Maximum length of extracted content
        
    Returns:
        Extracted text content or empty string if failed
    """
    try:
        logger.debug(f"Fetching content from URL: {url}")
        
        # Create HTTP client with timeout and user agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            
            html_content = response.text
            
            # Try to use BeautifulSoup if available for better parsing
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text(separator=' ', strip=True)
            except ImportError:
                # Fallback: simple regex-based extraction
                # Remove script and style tags
                text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)
                
                # Extract text from HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)
                text = unescape(text)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Limit length
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            logger.debug(f"Extracted {len(text)} characters from {url}")
            return text
            
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching URL: {url}")
        return ""
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error {e.response.status_code} fetching URL: {url}")
        return ""
    except Exception as e:
        logger.warning(f"Error fetching content from {url}: {e}")
        return ""


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(description="Search query string")
    max_results: Optional[int] = Field(default=5, description="Maximum number of results to return")


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for medical information.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Search results as formatted string
    """
    if not settings.USE_WEB_SEARCH:
        return "Web search is disabled."
    
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] web_search called with query: {query[:50]}...")
        result = _serpapi_search(query, max_results)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] web_search completed")
        return result
            
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [ERROR] web_search error: {e}")
        return f"Error performing web search: {str(e)}"


def _serpapi_search(query: str, max_results: int = 5, return_structured: bool = False, fetch_content: bool = False) -> Any:
    """
    Search using SerpAPI (requires API key).
    
    Args:
        query: Search query
        max_results: Maximum number of results
        return_structured: If True, returns list of dicts with structured data
        fetch_content: If True, fetches full content from URLs (only when return_structured=True)
        
    Returns:
        Either formatted string or list of dicts depending on return_structured
    """
    if not settings.SERPAPI_API_KEY:
        error_msg = "SerpAPI key not configured"
        logger.error(error_msg)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [ERROR] {error_msg}")
        if return_structured:
            return []
        return "Error: SerpAPI key not configured. Please set SERPAPI_API_KEY in your environment."
    
    try:
        from serpapi import GoogleSearch
        
        params = {
            "q": query,
            "api_key": settings.SERPAPI_API_KEY,
            "num": max_results,
            "engine": "google"  # Explicitly set engine
        }
        
        logger.debug(f"Calling SerpAPI with query: {query}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [DEBUG] Calling SerpAPI with query: {query[:50]}...")
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Check for errors in response
        if "error" in results:
            error_msg = f"SerpAPI error: {results.get('error', 'Unknown error')}"
            logger.error(error_msg)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [ERROR] {error_msg}")
            if return_structured:
                return []
            return f"Error: {error_msg}"
        
        # Check for other error indicators
        if "search_metadata" in results:
            status = results["search_metadata"].get("status", "")
            if status and status != "Success":
                error_msg = f"SerpAPI search failed with status: {status}"
                logger.error(error_msg)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [ERROR] {error_msg}")
                if return_structured:
                    return []
                return f"Error: {error_msg}"
        
        # Debug: log response keys to understand structure
        logger.debug(f"SerpAPI response keys: {list(results.keys())}")
        
        # Try multiple possible result keys
        organic_results = results.get("organic_results", [])
        
        # If no organic_results, try other possible keys
        if not organic_results:
            logger.warning(f"No 'organic_results' in response. Available keys: {list(results.keys())}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [WARNING] No organic_results found. Response keys: {list(results.keys())[:10]}")
            
            # Check for answer box or knowledge graph
            answer_box = results.get("answer_box", {})
            knowledge_graph = results.get("knowledge_graph", {})
            
            # If there are alternative result structures, try to use them
            if answer_box and isinstance(answer_box, dict):
                logger.info("Found answer_box in response")
                if return_structured:
                    url = answer_box.get("link", "")
                    result = {
                        "title": answer_box.get("title", answer_box.get("answer", "")),
                        "url": url,
                        "snippet": answer_box.get("answer", answer_box.get("snippet", ""))
                    }
                    # Fetch content if requested and URL is available
                    if fetch_content and url:
                        content = _fetch_url_content(url)
                        if content:
                            result["content"] = content
                        else:
                            result["content"] = result.get("snippet", "")
                    return [result]
            
            if not organic_results:
                logger.warning(f"No results found for query: {query}")
                if return_structured:
                    return []
                return f"No results found for query: {query}"
        
        logger.info(f"Found {len(organic_results)} organic results")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [DEBUG] Found {len(organic_results)} organic results")
        
        # Structure results
        structured_results = []
        for result in organic_results[:max_results]:
            if isinstance(result, dict):
                url = result.get("link", result.get("url", ""))
                structured_result = {
                    "title": result.get("title", ""),
                    "url": url,
                    "snippet": result.get("snippet", result.get("description", ""))
                }
                
                # Fetch full content from URL if requested
                if fetch_content and url and return_structured:
                    logger.info(f"Fetching content from: {url}")
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [INFO] Fetching content from URL {len(structured_results) + 1}/{len(organic_results[:max_results])}: {url[:60]}...")
                    content = _fetch_url_content(url)
                    if content:
                        structured_result["content"] = content
                        logger.info(f"Fetched {len(content)} characters from {url}")
                    else:
                        logger.warning(f"Failed to fetch content from {url}")
                        # Use snippet as fallback
                        structured_result["content"] = structured_result.get("snippet", "")
                
                structured_results.append(structured_result)
            else:
                logger.warning(f"Unexpected result format: {type(result)}")
        
        logger.info(f"Structured {len(structured_results)} results")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [DEBUG] Structured {len(structured_results)} results")
        
        if return_structured:
            return structured_results
        
        # Format results for string return
        if not structured_results:
            return f"No results found for query: {query}"
        
        formatted_results = f"Web search results for '{query}':\n\n"
        for i, result in enumerate(structured_results, 1):
            formatted_results += f"{i}. {result['title']}\n"
            formatted_results += f"   URL: {result['url']}\n"
            formatted_results += f"   {result['snippet']}\n\n"
        
        return formatted_results.strip()
        
    except ImportError:
        error_msg = "serpapi not installed. Install with: pip install google-search-results"
        logger.error(error_msg)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [ERROR] {error_msg}")
        if return_structured:
            return []
        return "SerpAPI not available. Please install google-search-results package."
    except Exception as e:
        error_msg = f"SerpAPI search error: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [ERROR] {error_msg}")
        import traceback
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TOOL] [ERROR] Traceback: {traceback.format_exc()}")
        if return_structured:
            return []
        return f"Error searching with SerpAPI: {str(e)}"

