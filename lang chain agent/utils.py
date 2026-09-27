"""
utils.py
--------
Utility helpers for the LangGraph Agent Assignment.

get_chat_model() automatically selects between:
  - Google Gemini  -> when GOOGLE_API_KEY is set in the environment
  - Groq           -> when GROQ_API_KEY is set (fallback)

Priority: GOOGLE_API_KEY > GROQ_API_KEY
"""

import os
from langchain_core.language_models.chat_models import BaseChatModel


def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """
    Factory that returns a configured chat-model instance.

    Resolution order
    ----------------
    1. If GOOGLE_API_KEY is present -> ChatGoogleGenerativeAI (Gemini)
    2. If GROQ_API_KEY is present   -> ChatGroq
    3. Otherwise raise a RuntimeError with helpful instructions.

    Parameters
    ----------
    temperature : float
        Sampling temperature forwarded to the underlying model.
        Default is 0.0 for deterministic, factual answers.

    Returns
    -------
    BaseChatModel
        A LangChain-compatible chat model ready for tool-binding.
    """

    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")

    if google_api_key:
        # -- Google Gemini path -----------------------------------------------
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            ) from exc

        model_name: str = os.getenv("GOOGLE_MODEL", "gemini-3.8-flash")
        print(f"[utils] Using Google Gemini model: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=google_api_key,
        )

    if groq_api_key:
        # -- Groq path --------------------------------------------------------
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ImportError(
                "langchain-groq is not installed. "
                "Run: pip install langchain-groq"
            ) from exc

        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        print(f"[utils] Using Groq model: {model_name}")
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=groq_api_key,
        )

    raise RuntimeError(
        "\n\nNo LLM API key found!\n"
        "Set one of the following environment variables:\n"
        "  - GOOGLE_API_KEY  (for Google Gemini)\n"
        "  - GROQ_API_KEY    (for Groq: Llama, Mixtral, ...)\n\n"
        "Example (Windows PowerShell):\n"
        '  $env:GOOGLE_API_KEY = "your-key-here"\n'
        "Example (bash / zsh):\n"
        '  export GOOGLE_API_KEY="your-key-here"\n'
    )
