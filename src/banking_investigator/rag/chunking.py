from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=lambda value: len(encoding.encode(value)),
    )

    split_contents = splitter.split_text(text)

    return [
        RagChunk(
            chunk_index=index,
            content=content,
            token_count=len(encoding.encode(content)),
        )
        for index, content in enumerate(split_contents)
    ]