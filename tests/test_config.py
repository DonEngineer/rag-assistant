"""Config loads sane local defaults without any env vars set."""

import os
from unittest import mock

from rag_assistant.config import Settings


def test_defaults():
    with mock.patch.dict(os.environ, {}, clear=True):
        s = Settings()
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.llm_model == "llama3.1"
    assert s.top_k == 5


def test_env_override():
    with mock.patch.dict(os.environ, {"TOP_K": "10", "OLLAMA_LLM_MODEL": "qwen2.5"}):
        s = Settings()
    assert s.top_k == 10
    assert s.llm_model == "qwen2.5"
