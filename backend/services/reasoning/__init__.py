"""
Reasoning module for VOKG
Provides LLM-powered reasoning over knowledge graphs
"""

from .reasoning_engine import ReasoningEngine, ReasoningResult
from .context_builder import ContextBuilder
from .prompts import REASONING_SYSTEM_PROMPT, build_reasoning_prompt

__all__ = [
    'ReasoningEngine',
    'ReasoningResult',
    'ContextBuilder',
    'REASONING_SYSTEM_PROMPT',
    'build_reasoning_prompt'
]
