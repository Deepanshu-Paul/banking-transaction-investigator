from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True)
class RagChunk:
    chunk_index: int
    content: str
    token_count: int


def chunk_text(
    text: str,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[RagChunk]:
    if not text.strip():
        return []

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError(
            "overlap_tokens must be >= 0 and < max_tokens"
        )

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks: list[RagChunk] = []
    step = max_tokens - overlap_tokens

    for chunk_index, start in enumerate(
        range(0, len(tokens), step)
    ):
        chunk_tokens = tokens[start : start + max_tokens]

        if not chunk_tokens:
            break

        content = encoding.decode(chunk_tokens)

        chunks.append(
            RagChunk(
                chunk_index=chunk_index,
                content=content,
                token_count=len(chunk_tokens),
            )
        )

        if start + max_tokens >= len(tokens):
            break

    return chunks