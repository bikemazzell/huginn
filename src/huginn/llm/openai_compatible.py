import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from huginn.schemas import ModelEndpointConfig


_EMBED_BATCH_SIZE = 256


@dataclass
class OpenAICompatibleEmbedder:
    config: ModelEndpointConfig

    def embed_text(self, text: str, *, kind: str = "document") -> list[float]:
        return self._embed_single_text(text)

    def embed_texts(
        self, texts: list[str], *, kind: str = "document"
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.config.model, "input": texts}
        try:
            data = _post_json(self.config, "/embeddings", payload, operation="embeddings")
        except RuntimeError as exc:
            if len(texts) == 1 or "HTTP 500" not in str(exc):
                return [self._embed_single_text(texts[0])]
            midpoint = len(texts) // 2
            return self._embed_batch(texts[:midpoint]) + self._embed_batch(texts[midpoint:])
        try:
            items = data["data"]
            if len(items) != len(texts):
                raise RuntimeError(
                    f"Embeddings response had {len(items)} entries for {len(texts)} inputs"
                )
            return [list(item["embedding"]) for item in items]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Invalid embeddings response payload") from exc

    def _embed_single_text(self, text: str) -> list[float]:
        payload = {"model": self.config.model, "input": text}
        try:
            data = _post_json(self.config, "/embeddings", payload, operation="embeddings")
        except RuntimeError as exc:
            words = text.split()
            if len(words) <= 1 or "HTTP 500" not in str(exc):
                raise
            return self._embed_single_text(" ".join(words[: len(words) // 2]))
        try:
            return list(data["data"][0]["embedding"])
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
            "reasoning": "off",
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
