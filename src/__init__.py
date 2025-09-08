"""
Apertus Swiss AI Transparency Library
Core module for transparent AI analysis and applications
"""

from .apertus_core import ApertusCore
from .transparency_analyzer import ApertusTransparencyAnalyzer
from .multilingual_assistant import SwissMultilingualAssistant
from .pharma_analyzer import PharmaDocumentAnalyzer

__version__ = "1.0.0"
__author__ = "Swiss AI Community"
__email__ = "community@swissai.dev"

__all__ = [
    "ApertusCore",
    "ApertusTransparencyAnalyzer", 
    "SwissMultilingualAssistant",
    "PharmaDocumentAnalyzer"
]
