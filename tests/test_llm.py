"""Tests for the LLM provider abstraction in services/llm.py.

No real network: we patch httpx.AsyncClient so each test controls the
request shape seen and the response sent back. The goal is to pin down
the contract each adapter has with its vendor API — request body, auth
headers, response parsing — because vendor shapes drift silently.
"""
import json

import httpx
import pytest

from app.services import llm


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)
        self.reason_phrase = "OK" if status_code == 200 else "ERR"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("POST", "https://example.com")
            raise httpx.HTTPStatusError("error", request=req, response=self)  # type: ignore


class FakeClient:
    """Stand-in for httpx.AsyncClient — records last POST, returns fixed payload."""
    calls: list[dict] = []

    def __init__(self, payload, status_code=200, *args, **kwargs):
        self._payload = payload
        self._status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, json=None, **kwargs):
        FakeClient.calls.append({"url": url, "headers": headers or {}, "json": json})
        return FakeResponse(self._payload, self._status_code)


@pytest.fixture(autouse=True)
def reset_calls():
    FakeClient.calls = []
    yield
    FakeClient.calls = []


def _patch_client(monkeypatch, payload, status_code=200):
    def factory(*args, **kwargs):
        return FakeClient(payload, status_code, *args, **kwargs)
    monkeypatch.setattr(llm.httpx, "AsyncClient", factory)


class TestProviderRegistry:
    def test_all_expected_providers_present(self):
        assert set(llm.PROVIDERS) >= {"mistral", "openai", "anthropic", "google"}

    def test_each_provider_has_at_least_one_model(self):
        for pid, cfg in llm.PROVIDERS.items():
            assert cfg["models"], f"{pid} has no models"
            for m in cfg["models"]:
                assert m["id"] and m["label"]

    def test_default_model_for_known_provider(self):
        assert llm.default_model_for("mistral") == llm.PROVIDERS["mistral"]["models"][0]["id"]

    def test_default_model_for_unknown_provider(self):
        assert llm.default_model_for("nonsense") == ""


class TestOpenAIStyle:
    @pytest.mark.asyncio
    async def test_mistral_uses_openai_shape(self, monkeypatch):
        _patch_client(monkeypatch, {
            "choices": [{"message": {"content": "result text"}}]
        })
        out = await llm.complete(
            "mistral", "sk-test", "mistral-medium-latest",
            system="sys", user="usr",
        )
        assert out == "result text"
        call = FakeClient.calls[0]
        assert call["url"] == "https://api.mistral.ai/v1/chat/completions"
        assert call["headers"]["Authorization"] == "Bearer sk-test"
        body = call["json"]
        assert body["model"] == "mistral-medium-latest"
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        assert body["messages"][1] == {"role": "user", "content": "usr"}
        assert body["temperature"] == 0

    @pytest.mark.asyncio
    async def test_openai_provider_hits_openai_host(self, monkeypatch):
        _patch_client(monkeypatch, {"choices": [{"message": {"content": "ok"}}]})
        await llm.complete("openai", "sk-1", "gpt-4o-mini", system="s", user="u")
        assert FakeClient.calls[0]["url"] == "https://api.openai.com/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_string(self, monkeypatch):
        _patch_client(monkeypatch, {"choices": [{"message": {"content": None}}]})
        out = await llm.complete("mistral", "k", "m", system="s", user="u")
        assert out == ""


class TestAnthropic:
    @pytest.mark.asyncio
    async def test_sends_system_as_top_level_field(self, monkeypatch):
        _patch_client(monkeypatch, {
            "content": [{"type": "text", "text": "claude answer"}]
        })
        out = await llm.complete(
            "anthropic", "sk-ant", "claude-haiku-4-5",
            system="you are a bot", user="hi",
        )
        assert out == "claude answer"
        call = FakeClient.calls[0]
        assert call["url"] == "https://api.anthropic.com/v1/messages"
        assert call["headers"]["x-api-key"] == "sk-ant"
        assert call["headers"]["anthropic-version"] == "2023-06-01"
        body = call["json"]
        # Critical: system is a top-level param, NOT a message with role=system
        assert body["system"] == "you are a bot"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert all(m["role"] != "system" for m in body["messages"])

    @pytest.mark.asyncio
    async def test_concatenates_multiple_text_blocks(self, monkeypatch):
        _patch_client(monkeypatch, {
            "content": [
                {"type": "text", "text": "part A "},
                {"type": "text", "text": "part B"},
                {"type": "tool_use", "name": "ignored"},
            ]
        })
        out = await llm.complete("anthropic", "k", "claude-haiku-4-5", system="s", user="u")
        assert out == "part A part B"


class TestGoogle:
    @pytest.mark.asyncio
    async def test_api_key_in_query_string_not_header(self, monkeypatch):
        _patch_client(monkeypatch, {
            "candidates": [{"content": {"parts": [{"text": "gemini says hi"}]}}]
        })
        out = await llm.complete(
            "google", "AIza-test", "gemini-2.5-flash",
            system="sys", user="usr",
        )
        assert out == "gemini says hi"
        call = FakeClient.calls[0]
        assert "key=AIza-test" in call["url"]
        assert "gemini-2.5-flash:generateContent" in call["url"]
        # Must NOT put the key in the Authorization header — only in URL
        assert "Authorization" not in call["headers"]
        body = call["json"]
        assert body["systemInstruction"]["parts"][0]["text"] == "sys"
        assert body["contents"][0]["parts"][0]["text"] == "usr"

    @pytest.mark.asyncio
    async def test_no_candidates_returns_empty(self, monkeypatch):
        _patch_client(monkeypatch, {"candidates": []})
        out = await llm.complete("google", "k", "gemini-2.5-flash", system="s", user="u")
        assert out == ""


class TestDispatch:
    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            await llm.complete("unknown", "k", "m", system="s", user="u")


class TestCheckKey:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        _patch_client(monkeypatch, {"choices": [{"message": {"content": "hi"}}]})
        ok, err = await llm.check_key("mistral", "k", "mistral-small-latest")
        assert ok is True
        assert err == ""

    @pytest.mark.asyncio
    async def test_http_error_extracts_message(self, monkeypatch):
        _patch_client(monkeypatch, {
            "error": {"message": "Invalid API key", "type": "invalid_request_error"}
        }, status_code=401)
        ok, err = await llm.check_key("mistral", "bad", "mistral-small-latest")
        assert ok is False
        assert "401" in err
        assert "Invalid API key" in err
