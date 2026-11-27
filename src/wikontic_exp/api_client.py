"""Один явный запрос к OpenAI-совместимому endpoint."""
import os

import httpx

import re
from pathlib import Path

DEFAULT_CREDS_FILE = Path("~/.config/wikontic/creds")


def _read_credentials(path: str | Path) -> tuple[str, str]:
    path = Path(path).expanduser()
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(
            r"\s*(LLM_URL|LLM_KEY)\s*[:=]\s*(.*?)\s*", line, re.IGNORECASE
        )
        if match:
            name, value = match.groups()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[name.upper()] = value

    if not values.get("LLM_URL") or not values.get("LLM_KEY"):
        raise ValueError(f"В {path} должны быть заданы LLM_URL и LLM_KEY")
    return values["LLM_URL"], values["LLM_KEY"]

def chat_response(
    model: str,
    messages: list[dict[str, str]],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | httpx.Timeout = 120,
    **parameters,
) -> httpx.Response:
    """Вернуть полный HTTP-ответ для записи статуса, тела и usage."""
    url = base_url if base_url is not None else os.environ["LLM_URL"]
    key = api_key if api_key is not None else os.environ["LLM_KEY"]
    return httpx.post(
        f"{url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": messages, **parameters},
        timeout=timeout,
    )


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | httpx.Timeout = 120,
    **parameters,
) -> str:
    """Вернуть текст ответа; при ошибке HTTP или отсутствии текста — исключение."""
    response = chat_response(
        model, messages, base_url=base_url, api_key=api_key, timeout=timeout, **parameters
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Модель не вернула текст ответа")
    return text
