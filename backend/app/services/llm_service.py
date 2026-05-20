"""
LLM service using Groq API with Llama 3 via LangChain's ChatGroq.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMService:
    """
    Wrapper around Groq's ChatGroq for all LLM calls within the RAG pipeline.
    Provides streaming and batch invocation helpers.
    """

    _instance: "LLMService | None" = None
    _llm: ChatGroq | None = None

    def __new__(cls) -> "LLMService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self) -> None:
        """Instantiate the ChatGroq client."""
        if self._llm is not None:
            return

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get your key at https://console.groq.com/"
            )

        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant",)
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))

        logger.info("Initializing Groq LLM | model=%s | temperature=%s", model, temperature)

        self._llm = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=api_key,
            max_retries=3,
        )
        logger.info("Groq LLM initialized successfully.")

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            self.initialize()
        return self._llm  # type: ignore[return-value]

    # ──────────────────────────────────────────────────────────
    # Core invocation
    # ──────────────────────────────────────────────────────────

    def invoke(
        self,
        system_prompt: str,
        human_message: str,
        temperature_override: Optional[float] = None,
    ) -> str:
        """
        Send a system + human message pair and return the text response.

        Args:
            system_prompt: System-level instructions.
            human_message: User/human turn content.
            temperature_override: Optionally override temperature for this call.

        Returns:
            LLM response text.
        """
        try:
            llm = self.llm
            if temperature_override is not None:
                llm = self.llm.with_config(
                    configurable={"temperature": temperature_override}
                )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message),
            ]
            response = llm.invoke(messages)
            return response.content  # type: ignore[union-attr]

        except Exception as exc:
            logger.error("LLM invocation failed: %s", exc, exc_info=True)
            raise

    def invoke_with_history(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        current_message: str,
    ) -> str:
        """
        Invoke the LLM with full conversation history for memory-aware responses.

        Args:
            system_prompt: System instructions.
            conversation_history: List of {'role': 'user'|'assistant', 'content': '...'}.
            current_message: Latest user message.

        Returns:
            LLM response text.
        """
        from langchain_core.messages import AIMessage

        messages: List[Any] = [SystemMessage(content=system_prompt)]

        for turn in conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=current_message))

        try:
            response = self.llm.invoke(messages)
            return response.content  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("LLM history invocation failed: %s", exc, exc_info=True)
            raise

    def stream(
        self,
        system_prompt: str,
        human_message: str,
    ):
        """
        Stream tokens from the LLM.

        Args:
            system_prompt: System instructions.
            human_message: User message content.

        Yields:
            Text chunks from the streaming response.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ]
        try:
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as exc:
            logger.error("LLM streaming failed: %s", exc, exc_info=True)
            raise

    def is_available(self) -> bool:
        """Check whether the LLM client is initialized."""
        return self._llm is not None


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """Dependency-injection helper — returns the singleton LLMService."""
    service = LLMService()
    service.initialize()
    return service
