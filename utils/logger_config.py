"""
Logging Configuration using Logfire

This module sets up structured logging with Logfire for observability.
Also prints essential logs to terminal.
"""

import os
import sys
import logfire
from datetime import datetime
from pathlib import Path
from typing import Optional
from config.settings import settings


def setup_logging(log_level: Optional[str] = None):
    """
    Set up logging with Logfire and terminal output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Get log level
    level = log_level or settings.LOG_LEVEL
    log_level_map = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL"
    }
    log_level_str = log_level_map.get(level.upper(), "INFO")
    
    # Configure Logfire
    try:
        logfire.configure()
        
        # Print initialization message to terminal
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Logging initialized with Logfire (Level: {log_level_str})")
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] Failed to initialize Logfire: {e}. Using basic logging.")
        import logging
        logging.basicConfig(level=logging.INFO)
    
    return logfire


def get_logger(name: str = None):
    """
    Get a logger instance using Logfire.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    if name:
        # Use with_tags correctly - tags must be hashable (strings, not dicts)
        # Extract just the module name if it's a full path
        module_name = name.split('.')[-1] if '.' in name else name
        return logfire.with_tags(module_name)
    return logfire

