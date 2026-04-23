import json
import os
import time
from typing import Any

import nltk
from sentence_transformers import SentenceTransformer

import chromadb

from ..latency import write_add_latency_summary

from .config import (
    DATASET_PATH,
    EMBEDDER_DEVICE,
    EMBEDDER_MODEL,
    RAG_CHROMA_COLLECTION_NAME,
    RAG_DB_PATH,
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def _tokenize_text(text: str) -> list[str]:
    return nltk.word_tokenize(text)


def _flatten_conversation(conversation: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in conversation.items():
        if key in ("speaker_a", "speaker_b") or "date" in key or "timestamp" in key:
            continue
        timestamp = conversation.get(f"{key}_date_time", "")
        section_label = f"[{key}]"
        if timestamp:
            lines.append(f"{section_label} {timestamp}")
        else:
            lines.append(section_label)
        for chat in value:
            speaker = chat.get("speaker", "")
            text = chat.get("text", "")
            lines.append(f"{speaker}: {text}")
        lines.append("")
    return "\n".join(lines).strip()


def _chunk_text(text: str, chunk_size: int) -> list[dict[str, Any]]:
    tokens = _tokenize_text(text)
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(tokens), chunk_size):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append({"start": start, "end": end, "text": chunk_text})
    return chunks


class RAGIndexer:
    def __init__(self, dataset_path: str = DATASET_PATH, chunk_size: int = 512):
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.embedder = SentenceTransformer(EMBEDDER_MODEL, device=EMBEDDER_DEVICE)
        os.makedirs(RAG_DB_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(path=RAG_DB_PATH)
        self.collection = self._prepare_collection()

    def _prepare_collection(self):
        try:
            self.client.delete_collection(name=RAG_CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        return self.client.create_collection(name=RAG_CHROMA_COLLECTION_NAME)

    def load_dataset(self) -> list[dict[str, Any]]:
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def index_dataset(self) -> None:
        data = self.load_dataset()

        per_conversation_seconds: list[float] = []

        for conversation_idx, item in enumerate(data):
            t_conv0 = time.perf_counter()
            conversation = item.get("conversation", {})
            speaker_a = conversation.get("speaker_a", "")
            speaker_b = conversation.get("speaker_b", "")
            session_text = _flatten_conversation(conversation)
            chunks = _chunk_text(session_text, self.chunk_size)

            chunk_texts: list[str] = []
            chunk_ids: list[str] = []
            metadatas: list[dict[str, Any]] = []

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{conversation_idx}-{chunk_idx}-{self.chunk_size}"
                chunk_texts.append(chunk["text"])
                chunk_ids.append(chunk_id)
                metadatas.append(
                    {
                        "conversation_idx": conversation_idx,
                        "chunk_idx": chunk_idx,
                        "chunk_size": self.chunk_size,
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                        "token_start": chunk["start"],
                        "token_end": chunk["end"],
                    }
                )

            embeddings = self.embedder.encode(chunk_texts, convert_to_numpy=True)
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=metadatas,
                embeddings=embeddings.tolist(),
            )
            per_conversation_seconds.append(time.perf_counter() - t_conv0)

        baseline_dir = os.path.dirname(os.path.abspath(__file__))
        add_summary_path = os.path.join(baseline_dir, "results", "add_latency_summary.json")
        path = write_add_latency_summary(
            add_summary_path,
            baseline="rag",
            per_batch_seconds=per_conversation_seconds,
            primary_key="per_conversation_index_sec",
            sample_unit="conversation",
        )
        print(f"[RAG-INDEX] Wrote index latency summary to {path}")

    def build(self) -> None:
        os.makedirs(RAG_DB_PATH, exist_ok=True)
        self.index_dataset()


def main() -> None:
    indexer = RAGIndexer(chunk_size=512)
    indexer.build()


if __name__ == "__main__":
    main()
