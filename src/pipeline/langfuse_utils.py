from __future__ import annotations

import os

import ulid
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

from .config import build_langfuse_config, build_openrouter_config, load_project_env


def generate_session_id() -> str:
    team = (os.getenv("TEAM_NAME") or "tutorial").replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def build_langchain_model() -> ChatOpenAI:
    cfg = build_openrouter_config()
    return ChatOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


def build_langfuse_client() -> Langfuse:
    cfg = build_langfuse_config()
    return Langfuse(
        public_key=cfg.public_key,
        secret_key=cfg.secret_key,
        host=cfg.host,
    )


def invoke_langchain(model: ChatOpenAI, prompt: str, session_id: str) -> str:
    handler = CallbackHandler()
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(
        messages,
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )
    return response.content


@observe()
def run_llm_call(session_id: str, model: ChatOpenAI, prompt: str) -> str:
    return invoke_langchain(model=model, prompt=prompt, session_id=session_id)


def setup_langfuse_integration() -> tuple[ChatOpenAI, Langfuse, str]:
    """Bootstrap root .env + clients and return (model, langfuse_client, session_id)."""
    load_project_env()
    model = build_langchain_model()
    langfuse_client = build_langfuse_client()
    session_id = generate_session_id()
    return model, langfuse_client, session_id

