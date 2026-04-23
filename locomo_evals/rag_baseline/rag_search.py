import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import (
    DATASET_PATH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LLM_CONCURRENCY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_K,
    EMBEDDER_DEVICE,
    EMBEDDER_MODEL,
    RAG_CHROMA_COLLECTION_NAME,
    RAG_DB_PATH,
    VLLM_MODEL,
    get_vllm_client,
)
from ..latency import write_search_latency_summary

from .prompts import RAG_ANSWER_PROMPT


class RAGSearch:
    def __init__(
        self,
        output_path: str = "results.json",
        top_k: int = DEFAULT_TOP_K,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_workers: int = DEFAULT_LLM_CONCURRENCY,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.output_path = output_path
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.max_workers = max(1, int(max_workers))
        self.max_tokens = int(max_tokens)
        self.client = get_vllm_client()
        self.db_client = chromadb.PersistentClient(path=RAG_DB_PATH)
        self.collection = self.db_client.get_collection(name=RAG_CHROMA_COLLECTION_NAME)
        self.embedder = SentenceTransformer(EMBEDDER_MODEL, device=EMBEDDER_DEVICE)
        self.results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # Chroma's PersistentClient is not safe for concurrent writes; the
        # embedder has internal state too. Guard both behind a lock so the
        # thread pool only truly parallelizes the HTTP LLM calls.
        self._retrieval_lock = threading.Lock()
        self._save_lock = threading.Lock()

    def search_chunks(self, query: str) -> list[dict[str, Any]]:
        with self._retrieval_lock:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True).tolist()
            query_result = self.collection.query(
                query_embeddings=query_embedding,
                n_results=self.top_k,
                include=["documents", "metadatas", "distances"],
            )

        ids = query_result.get("ids", [[]])[0]
        docs = query_result.get("documents", [[]])[0]
        metas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        results = []
        for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances):
            results.append(
                {
                    "chunk_id": chunk_id,
                    "chunk": doc,
                    "metadata": meta,
                    "distance": float(distance),
                }
            )
        return results

    def generate_answer(self, context: str, question: str) -> tuple[str, float]:
        prompt = RAG_ANSWER_PROMPT.format(context=context, question=question)
        start = time.time()
        response = self.client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        elapsed = time.time() - start
        content = ""
        try:
            content = response.choices[0].message.content
        except Exception:
            content = response.choices[0]["message"]["content"] if response.choices else ""
        return content.strip(), elapsed

    def process_question(self, item: dict[str, Any], conversation_idx: int) -> dict[str, Any]:
        question = str(item.get("question", ""))
        answer = str(item.get("answer", ""))
        category = item.get("category", -1)
        evidence = item.get("evidence", [])

        t_retrieve0 = time.perf_counter()
        top_chunks = self.search_chunks(question)
        retrieval_latency = time.perf_counter() - t_retrieve0
        context_lines = []
        for chunk in top_chunks:
            meta = chunk["metadata"]
            context_lines.append(
                f"Chunk {chunk['chunk_id']} (conv {meta.get('conversation_idx')}): {chunk['chunk']}"
            )
        context = "\n\n".join(context_lines)

        response, generation_time = self.generate_answer(context, question)
        total_latency = retrieval_latency + generation_time
        return {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "retrieved_chunks": top_chunks,
            "num_chunks": len(top_chunks),
            "generation_time": generation_time,
            "search_latency_sec": retrieval_latency,
            "generation_latency_sec": generation_time,
            "total_latency_sec": total_latency,
        }

    def process_data_file(self, file_path: str = DATASET_PATH) -> None:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Flatten everything into (conv_idx, qa_position, qa_item) so we can
        # fan the whole workload out across the thread pool at once.
        tasks: list[tuple[int, int, dict[str, Any]]] = []
        for conv_idx, item in enumerate(data):
            qa_list = item.get("qa", [])
            for qa_pos, qa_item in enumerate(qa_list):
                tasks.append((conv_idx, qa_pos, qa_item))

        # Pre-size per-conversation result lists so we can drop results into
        # their original positions regardless of completion order.
        per_conv_len: dict[int, int] = defaultdict(int)
        for conv_idx, qa_pos, _ in tasks:
            per_conv_len[conv_idx] = max(per_conv_len[conv_idx], qa_pos + 1)
        slots: dict[str, list[Any]] = {
            str(conv_idx): [None] * n for conv_idx, n in per_conv_len.items()
        }

        completed = 0
        save_every = max(1, len(tasks) // 20)  # ~20 incremental saves total

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_meta = {
                pool.submit(self.process_question, qa_item, conv_idx): (conv_idx, qa_pos)
                for conv_idx, qa_pos, qa_item in tasks
            }
            for future in tqdm(
                as_completed(future_to_meta),
                total=len(future_to_meta),
                desc="RAG search",
            ):
                conv_idx, qa_pos = future_to_meta[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                slots[str(conv_idx)][qa_pos] = result
                completed += 1
                if completed % save_every == 0:
                    self._flush(slots)

        self._flush(slots)
        summary_path = write_search_latency_summary(self.output_path, self.results, baseline="rag")
        print(f"  [latency] Wrote search latency summary to {summary_path}")

    def _flush(self, slots: dict[str, list[Any]]) -> None:
        # Copy into self.results (dropping any still-empty slots) and persist.
        self.results = defaultdict(list)
        for conv_key, items in slots.items():
            self.results[conv_key] = [it for it in items if it is not None]
        self._save()

    def _save(self) -> None:
        with self._save_lock:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)


def main() -> None:
    searcher = RAGSearch(output_path="results/rag_search_results.json")
    searcher.process_data_file()


if __name__ == "__main__":
    main()
