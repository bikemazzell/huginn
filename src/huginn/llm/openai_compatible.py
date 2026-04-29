import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from huginn.schemas import ModelEndpointConfig


@dataclass
class OpenAICompatibleEmbedder:
    config: ModelEndpointConfig

    def embed_text(self, text: str, *, kind: str = "document") -> list[float]:
        payload = {"model": self.config.model, "input": text}
        data = _post_json(self.config, "/embeddings", payload, operation="embeddings")
        try:
            return list(data["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Invalid embeddings response payload") from exc

    def embed_texts(
        self, texts: list[str], *, kind: str = "document"
    ) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.config.model, "input": texts}
        data = _post_json(self.config, "/embeddings", payload, operation="embeddings")
        try:
            items = data["data"]
            if len(items) != len(texts):
                raise RuntimeError(
                    f"Embeddings response had {len(items)} entries for {len(texts)} inputs"
                )
            return [list(item["embedding"]) for item in items]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Invalid embeddings response payload") from exc


@dataclass
class OpenAICompatibleChatModel:
    config: ModelEndpointConfig

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = _post_json(self.config, "/chat/completions", payload, operation="chat completions")
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Invalid chat completions response payload") from exc


def _post_json(
    config: ModelEndpointConfig,
    path: str,
    payload: dict[str, object],
    *,
    operation: str,
) -> dict:
    url = config.base_url.rstrip("/") + path
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenAI-compatible {operation} request failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI-compatible {operation} request failed: {exc.reason}") from exc
