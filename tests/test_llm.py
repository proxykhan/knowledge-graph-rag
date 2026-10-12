import json

import pytest
from pydantic import BaseModel

from kgrag import llm
from kgrag.config import settings


class Small(BaseModel):
    answer: str


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _patch_post(monkeypatch, response, seen=None):
    import requests

    def fake_post(url, json=None, timeout=None):
        if seen is not None:
            seen.append((url, json))
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(requests, "post", fake_post)


def test_ollama_returns_validated_schema_and_usage(monkeypatch):
    seen = []
    _patch_post(monkeypatch, FakeResponse(200, {"message": {"content": '{"answer": "TSMC"}'},
                                                "prompt_eval_count": 120, "eval_count": 8, "done_reason": "stop"}), seen)
    result, usage = llm.generate_structured("sys", "user", Small, provider="ollama")
    assert result.answer == "TSMC" and usage.input_tokens == 120 and usage.output_tokens == 8
    url, body = seen[0]
    assert url.endswith("/api/chat") and body["format"]["properties"]["answer"]["type"] == "string"
    assert body["model"] == settings.ollama_model and body["stream"] is False


def test_ollama_failures_are_retryable_errors(monkeypatch):
    import requests

    _patch_post(monkeypatch, requests.ConnectionError("refused"))
    with pytest.raises(llm.LLMError, match="not reachable"):
        llm.generate_structured("s", "u", Small, provider="ollama")
    _patch_post(monkeypatch, FakeResponse(404, {"error": "model not found"}))
    with pytest.raises(llm.LLMError, match="ollama pull"):
        llm.generate_structured("s", "u", Small, provider="ollama")
    _patch_post(monkeypatch, FakeResponse(200, {"message": {"content": "not json"}, "done_reason": "stop"}))
    with pytest.raises(llm.LLMError, match="schema"):
        llm.generate_structured("s", "u", Small, provider="ollama")


def test_extraction_and_answering_models_are_separate(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "extraction_provider", "gemini")
    assert llm.model_name() == settings.ollama_model
    assert llm.model_name(settings.extraction_provider) == settings.gemini_model
