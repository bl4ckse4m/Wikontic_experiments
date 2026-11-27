import httpx
import pytest

from wikontic_exp.api_client import chat


def test_chat_sends_one_request_and_preserves_text(monkeypatch):
    calls = []
    messages = [{"role": "user", "content": "Привет"}]

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, request=httpx.Request("POST", url),
                              json={"choices": [{"message": {"content": " ответ\n"}}]})

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("LLM_URL", "https://example.invalid/v1/")
    monkeypatch.setenv("LLM_KEY", "test")
    assert chat("model-a", messages, max_tokens=16) == " ответ\n"
    assert calls == [("https://example.invalid/v1/chat/completions", {
        "headers": {"Authorization": "Bearer test"},
        "json": {"model": "model-a", "messages": messages, "max_tokens": 16},
        "timeout": 120,
    })]


@pytest.mark.parametrize(("status", "error"), [(429, httpx.HTTPStatusError), (200, ValueError)])
def test_failed_call_is_not_an_answer(monkeypatch, status, error):
    def post(url, **kwargs):
        return httpx.Response(status, request=httpx.Request("POST", url),
                              json={"choices": [{"message": {"content": None}}]})

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(error):
        chat("model-b", [], base_url="https://example.invalid/v1", api_key="test")
