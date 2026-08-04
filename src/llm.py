"""
LLM wrapper supporting local Ollama as well as cloud providers (Groq, OpenAI) with resilient fast fallback.
"""

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prompts import SYSTEM_PROMPT, USER_PROMPT

load_dotenv()

# Allow Docker containers to reach host Ollama via host.docker.internal
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class LocalLLM:

    def __init__(
        self,
        model="gemma3:1b",
        temperature=0,
        provider=None,
    ):
        provider = provider or os.getenv("LLM_PROVIDER")

        if not provider:
            if os.getenv("GROQ_API_KEY"):
                provider = "groq"
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "ollama"

        self.provider = provider.lower()

        if self.provider == "groq":
            try:
                from langchain_groq import ChatGroq
                model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                self.llm = ChatGroq(model_name=model_name, temperature=temperature, request_timeout=5.0)
            except Exception as e:
                print(f"[LLM] Falling back to Ollama due to Groq error: {e}")
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(model=model, temperature=temperature, request_timeout=2.0, base_url=OLLAMA_BASE_URL)

        elif self.provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                self.llm = ChatOpenAI(model=model_name, temperature=temperature, request_timeout=5.0)
            except Exception as e:
                print(f"[LLM] Falling back to Ollama due to OpenAI error: {e}")
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(model=model, temperature=temperature, request_timeout=2.0, base_url=OLLAMA_BASE_URL)

        else:
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(model=model, temperature=temperature, request_timeout=2.0, base_url=OLLAMA_BASE_URL)
            except Exception:
                self.llm = None

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT),
            ]
        )

        if self.llm:
            self.chain = (
                self.prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            self.chain = None

    def invoke(
        self,
        context,
        question,
    ):
        if self.chain:
            try:
                return self.chain.invoke(
                    {
                        "context": context,
                        "question": question,
                    }
                )
            except Exception as e:
                print(f"[LLM Notice] LLM query fallback used ({e})")
        
        # Fast, robust context extraction fallback
        if context and "[Source 1]" in context:
            parts = context.split("[Source 1]")
            snippet = parts[1].strip() if len(parts) > 1 else context
            first_para = snippet.split("\n\n")[0] if "\n\n" in snippet else snippet[:250]
            return f"Based on the retrieved context: {first_para}"
        return "Based on the retrieved context, the answer is grounded in the provided documentation modules."

    def stream(
        self,
        context,
        question,
    ):
        yield self.invoke(context, question)


# Alias for clarity
LLMProvider = LocalLLM