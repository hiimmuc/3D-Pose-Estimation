#!/usr/bin/env python
"""
Decoration utilities for console output and visualization.

This module centralizes all decoration-related constants and utilities,
including ANSI color codes and text formatting functions.
"""

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
BOLD = '\033[1m'
END = '\033[0m'

def color_text(text: str, color: str) -> str:
    """Wrap text with ANSI color code.
    
    Args:
        text: Text to color
        color: Color code constant from this module
        
    Returns:
        Colored text string
    """
    return f"{color}{text}{END}"

def success(text: str) -> str:
    """Format text as success message (green).
    
    Args:
        text: Message text
        
    Returns:
        Formatted success message
    """
    return f"{GREEN}✓{END} {text}"

def error(text: str) -> str:
    """Format text as error message (red).
    
    Args:
        text: Message text
        
    Returns:
        Formatted error message
    """
    return f"{RED}✗{END} {text}"

def warning(text: str) -> str:
    """Format text as warning message (yellow).
    
    Args:
        text: Message text
        
    Returns:
        Formatted warning message
    """
    return f"{YELLOW}⚠{END} {text}"

def info(text: str) -> str:
    """Format text as info message (blue).
    
    Args:
        text: Message text
        
    Returns:
        Formatted info message
    """
    return f"{BLUE}➤{END} {text}"
