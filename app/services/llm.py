"""LLM provider abstraction.

Each provider (Mistral, OpenAI, Anthropic, Google Gemini) has its own auth
header, request shape and response shape. This module hides those
differences behind a single `complete()` coroutine so the rest of the app
never has to care which vendor is serving the request.

No vendor SDKs are imported — all four providers are hit via raw httpx.
That keeps `requirements.txt` short and avoids the install-time weight of
four separate client libraries just to send a chat completion.
"""
from __future__ import annotations

import logging
from typing import TypedDict

import httpx

logger = logging.getLogger(__name__)


# --- Provider registry ------------------------------------------------------

class ModelOption(TypedDict):
    id: str
    label: str


class ProviderConfig(TypedDict):
    label: str
    models: list[ModelOption]
    get_key_url: str
    note: str  # shown under the API key field; empty string if none


PROVIDERS: dict[str, ProviderConfig] = {
    "mistral": {
        "label": "Mistral",
        "models": [
            {"id": "mistral-medium-latest", "label": "Medium (лучшая)"},
            {"id": "mistral-small-latest",  "label": "Small (точная)"},
            {"id": "open-mistral-nemo",     "label": "Nemo (быстрая)"},
        ],
        "get_key_url": "https://console.mistral.ai",
        "note": "Из России может потребоваться VPN",
    },
    "openai": {
        "label": "OpenAI",
        "models": [
            {"id": "gpt-4o-mini", "label": "GPT-4o mini (дешёвая)"},
            {"id": "gpt-4o",      "label": "GPT-4o (лучшая)"},
            {"id": "gpt-4.1-mini","label": "GPT-4.1 mini"},
        ],
        "get_key_url": "https://platform.openai.com/api-keys",
        "note": "Из России может потребоваться VPN",
    },
    "anthropic": {
        "label": "Anthropic",
        "models": [
            {"id": "claude-haiku-4-5",  "label": "Haiku 4.5 (быстрая)"},
            {"id": "claude-sonnet-4-6", "label": "Sonnet 4.6 (лучшая)"},
        ],
        "get_key_url": "https://console.anthropic.com/settings/keys",
        "note": "Из России может потребоваться VPN",
    },
    "google": {
        "label": "Google Gemini",
        "models": [
            {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash (быстрая)"},
            {"id": "gemini-2.5-pro",   "label": "Gemini 2.5 Pro (лучшая)"},
        ],
        "get_key_url": "https://aistudio.google.com/apikey",
        "note": "",
    },
}


DEFAULT_PROVIDER = "mistral"


def default_model_for(provider: str) -> str:
    """Return the first (usually best default) model for a provider."""
    cfg = PROVIDERS.get(provider)
    if not cfg or not cfg["models"]:
        return ""
    return cfg["models"][0]["id"]


# --- Dispatch ---------------------------------------------------------------

async def complete(
    provider: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 8000,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> str:
    """Send a chat completion and return the model's text output.

    Raises `httpx.HTTPStatusError` / `httpx.TimeoutException` on failure
    so the caller can log them or retry.
    """
    if provider == "mistral":
        return await _openai_style(
            "https://api.mistral.ai/v1/chat/completions",
            api_key, model, system, user, max_tokens, temperature, timeout,
        )
    if provider == "openai":
        return await _openai_style(
            "https://api.openai.com/v1/chat/completions",
            api_key, model, system, user, max_tokens, temperature, timeout,
        )
    if provider == "anthropic":
        return await _anthropic(
            api_key, model, system, user, max_tokens, temperature, timeout,
        )
    if provider == "google":
        return await _google(
            api_key, model, system, user, max_tokens, temperature, timeout,
        )
    raise ValueError(f"Unknown provider: {provider!r}")


async def check_key(provider: str, api_key: str, model: str) -> tuple[bool, str]:
    """Lightweight key validation — sends a minimal ping."""
    try:
        # max_tokens=16 because some providers (Gemini) reject max_tokens<2
        # and we want to see "hi" come back rather than an empty response.
        await complete(
            provider, api_key, model,
            system="Ты — ассистент. Отвечай очень коротко.",
            user="ping",
            max_tokens=16,
            timeout=15,
        )
        return True, ""
    except httpx.HTTPStatusError as e:
        return False, f"{e.response.status_code}: {_extract_error(e.response)}"
    except httpx.TimeoutException:
        return False, "Timeout — провайдер не отвечает"
    except Exception as e:
        return False, str(e)


# --- Provider implementations -----------------------------------------------

async def _openai_style(
    url: str, api_key: str, model: str,
    system: str, user: str,
    max_tokens: int, temperature: float, timeout: float,
) -> str:
    """OpenAI chat-completions format — used by OpenAI AND Mistral.

    Both vendors accept the same schema, so we share the implementation.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"].get("content") or "").strip()


async def _anthropic(
    api_key: str, model: str,
    system: str, user: str,
    max_tokens: int, temperature: float, timeout: float,
) -> str:
    """Anthropic Messages API — `system` is a top-level parameter, not a role."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        blocks = data.get("content") or []
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()


async def _google(
    api_key: str, model: str,
    system: str, user: str,
    max_tokens: int, temperature: float, timeout: float,
) -> str:
    """Google Gemini generateContent — API key goes in the query string."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent?key={api_key}"
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()


def _extract_error(resp: httpx.Response) -> str:
    """Pull a human-readable message out of a provider error response."""
    try:
        data = resp.json()
    except Exception:
        return resp.text[:200] if resp.text else resp.reason_phrase

    if isinstance(data, dict):
        # OpenAI / Mistral: {"error": {"message": ..., "type": ...}}
        err = data.get("error")
        if isinstance(err, dict):
            return err.get("message") or err.get("type") or str(err)
        if isinstance(err, str):
            return err
        # Anthropic: {"type": "error", "error": {"type": ..., "message": ...}}
        if data.get("type") == "error" and isinstance(data.get("message"), str):
            return data["message"]
        # Google: {"error": {"message": ..., "code": ...}} (already handled above)
        if "message" in data:
            return str(data["message"])
    return str(data)[:200]
