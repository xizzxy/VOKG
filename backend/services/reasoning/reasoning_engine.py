"""
LLM-powered reasoning engine
Answers complex questions about video content using knowledge graphs
"""

import os
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .context_builder import ContextBuilder
from .prompts import REASONING_SYSTEM_PROMPT, build_reasoning_prompt
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReasoningResult:
    """Result from reasoning engine"""
    question: str
    answer: str
    reasoning: str
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    evidence: List[str]
    context_used: str

    def to_dict(self) -> Dict:
        return {
            'question': self.question,
            'answer': self.answer,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'evidence': self.evidence
        }


class ReasoningEngine:
    """
    LLM-powered reasoning over knowledge graphs
    Supports OpenAI and Gemini
    """

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize reasoning engine

        Args:
            provider: LLM provider ("openai" or "gemini", from env if None)
            model: Model name (default based on provider)
            temperature: Sampling temperature (low for factual reasoning)
            max_tokens: Max tokens in response
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai")
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set default models
        if model:
            self.model = model
        elif self.provider == "openai":
            self.model = "gpt-4o"  # Best reasoning model
        elif self.provider == "gemini":
            self.model = "gemini-2.0-flash-exp"
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Initialize client
        self.client = self._init_client()
        self.context_builder = ContextBuilder(max_tokens=4000)

        logger.info(f"ReasoningEngine initialized with {self.provider}:{self.model}")

    def _init_client(self):
        """Initialize LLM client"""
        if self.provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                return openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY environment variable not set")
                genai.configure(api_key=api_key)
                return genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def reason(
        self,
        query: str,
        objects: List[Dict],
        interactions: List[Dict],
        query_embedding: Optional[Any] = None
    ) -> ReasoningResult:
        """
        Answer a complex question using LLM reasoning

        Args:
            query: Natural language question
            objects: List of detected objects with labels
            interactions: List of interactions
            query_embedding: Optional query embedding for context ranking

        Returns:
            ReasoningResult with answer and explanation
        """
        logger.info(f"Reasoning query: {query}")

        # Build context
        graph_context = self.context_builder.build_context(
            objects,
            interactions,
            query=query,
            query_embedding=query_embedding
        )

        # Build prompt
        prompt = build_reasoning_prompt(query, graph_context, max_tokens=4000)

        # Call LLM
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                response_text = response.choices[0].message.content

            else:  # gemini
                full_prompt = f"{REASONING_SYSTEM_PROMPT}\n\n{prompt}"
                response = self.client.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens
                    }
                )
                response_text = response.text

            # Parse response
            result = self._parse_response(query, response_text, graph_context)

            logger.info(f"Reasoning complete: {result.confidence} confidence")
            return result

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return ReasoningResult(
                question=query,
                answer=f"Error during reasoning: {str(e)}",
                reasoning="",
                confidence="LOW",
                evidence=[],
                context_used=graph_context
            )

    def _parse_response(
        self,
        query: str,
        response_text: str,
        context: str
    ) -> ReasoningResult:
        """Parse LLM response into structured result"""
        # Extract sections using regex
        reasoning_match = re.search(r'\*\*Reasoning:\*\*\s*\n(.*?)\n\n\*\*Answer:\*\*', response_text, re.DOTALL)
        answer_match = re.search(r'\*\*Answer:\*\*\s*\n(.*?)\n\n\*\*Confidence:\*\*', response_text, re.DOTALL)
        confidence_match = re.search(r'\*\*Confidence:\*\*\s*\n(.*?)\n\n\*\*Evidence:\*\*', response_text, re.DOTALL)
        evidence_match = re.search(r'\*\*Evidence:\*\*\s*\n(.*?)$', response_text, re.DOTALL)

        # Extract or use defaults
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text
        answer = answer_match.group(1).strip() if answer_match else response_text
        confidence = confidence_match.group(1).strip() if confidence_match else "MEDIUM"
        evidence_text = evidence_match.group(1).strip() if evidence_match else ""

        # Parse evidence into list
        evidence = []
        if evidence_text:
            evidence = [line.strip('- ').strip() for line in evidence_text.split('\n') if line.strip()]

        return ReasoningResult(
            question=query,
            answer=answer,
            reasoning=reasoning,
            confidence=confidence.upper() if confidence.upper() in ["HIGH", "MEDIUM", "LOW"] else "MEDIUM",
            evidence=evidence,
            context_used=context
        )

    def analyze_video(
        self,
        objects: List[Dict],
        interactions: List[Dict]
    ) -> ReasoningResult:
        """
        Provide general analysis of video content

        Args:
            objects: List of detected objects
            interactions: List of interactions

        Returns:
            ReasoningResult with video analysis
        """
        # Build summary statistics
        summary = self.context_builder.build_summary_statistics(objects, interactions)

        # Create analysis query
        query = "Analyze this video and describe what is happening."

        # Build context
        graph_context = self.context_builder.build_context(objects, interactions)

        # Build prompt
        from .prompts import build_analysis_prompt
        prompt = build_analysis_prompt(summary)

        # Call LLM
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt + "\n\n" + graph_context}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                response_text = response.choices[0].message.content

            else:  # gemini
                full_prompt = f"{REASONING_SYSTEM_PROMPT}\n\n{prompt}\n\n{graph_context}"
                response = self.client.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens
                    }
                )
                response_text = response.text

            return ReasoningResult(
                question=query,
                answer=response_text,
                reasoning="Video analysis based on knowledge graph",
                confidence="HIGH",
                evidence=[f"{summary['total_objects']} objects", f"{summary['total_interactions']} interactions"],
                context_used=graph_context
            )

        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return ReasoningResult(
                question=query,
                answer=f"Error during analysis: {str(e)}",
                reasoning="",
                confidence="LOW",
                evidence=[],
                context_used=graph_context
            )
