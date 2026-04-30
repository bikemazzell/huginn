from dataclasses import dataclass

from huginn.llm.openai_compatible import OpenAICompatibleChatModel, OpenAICompatibleEmbedder
from huginn.retrieve.basic import lexical_features
from huginn.schemas import ModelsConfig


@dataclass
class RuntimeClients:
    embedder: object
    chat: object | None


class LocalLexicalEmbedder:
    def embed_text(self, text: str, *, kind: str = "document") -> dict[str, float]:
        return lexical_features(text)

    def embed_texts(
        self, texts: list[str], *, kind: str = "document"
    ) -> list[dict[str, float]]:
        return [lexical_features(text) for text in texts]


class NomicPrefixEmbedder:
    def __init__(self, inner: object) -> None:
        self.inner = inner

    def embed_text(self, text: str, *, kind: str = "document") -> list[float] | dict[str, float]:
        return self.inner.embed_text(_nomic_prefix(kind) + text)

    def embed_texts(
        self, texts: list[str], *, kind: str = "document"
    ) -> list[list[float]] | list[dict[str, float]]:
        prefix = _nomic_prefix(kind)
        prefixed = [prefix + text for text in texts]
        if hasattr(self.inner, "embed_texts"):
            return self.inner.embed_texts(prefixed)
        return [self.inner.embed_text(text) for text in prefixed]


def _nomic_prefix(kind: str) -> str:
    return "search_query: " if kind == "query" else "search_document: "


def describe_models(models: ModelsConfig) -> dict[str, str]:
    return {
        "chat_model": models.chat.model,
        "embedding_model": models.embedding.model,
    }


def build_runtime_clients(models: ModelsConfig) -> RuntimeClients:
    if models.embedding.model == "local-lexical":
        embedder = LocalLexicalEmbedder()
    else:
        raw_embedder = OpenAICompatibleEmbedder(models.embedding)
        if models.embedding.model.startswith("nomic-embed-text-"):
            embedder = NomicPrefixEmbedder(raw_embedder)
        else:
            embedder = raw_embedder

    chat_model: object | None
    if models.chat.model == "disabled":
        chat_model = None
    else:
        chat_model = OpenAICompatibleChatModel(models.chat)

    return RuntimeClients(
        embedder=embedder,
        chat=chat_model,
    )
