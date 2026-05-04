"""Optional Ollama embeddings for Celebro v2."""

from __future__ import annotations


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434") -> None:
        import ollama

        self.model_name = model
        self._client = ollama.Client(host=host)

    def embed(self, text: str) -> list[float]:
        response = self._client.embed(model=self.model_name, input=text)
        return list(response.embeddings[0])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(model=self.model_name, input=texts)
        return [list(v) for v in response.embeddings]
