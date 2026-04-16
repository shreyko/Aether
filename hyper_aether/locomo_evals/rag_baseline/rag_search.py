import json
import os
import time
from collections import defaultdict
from typing import Any

import chromadb
from chromadb.config import Settings

from .config import (
    DATASET_PATH,
    RAG_CHROMA_COLLECTION_NAME,
    RAG_DB_PATH,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_SIZE,
    VLLM_MODEL,
    get_vllm_client,
)
from .prompts import RAG_ANSWER_PROMPT


class RAGSearch:
    def __init__(self, output_path: str = "results.json", top_k: int = DEFAULT_TOP_K, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.output_path = output_path
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.client = get_vllm_client()
        self.db_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=RAG_DB_PATH))
        self.collection = self.db_client.get_collection(name=RAG_CHROMA_COLLECTION_NAME)
        self.results: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def search_chunks(self, query: str) -> list[dict[str, Any]]:
        query_result = self.collection.query(
            query_texts=[query],
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

        top_chunks = self.search_chunks(question)
        context_lines = []
        for chunk in top_chunks:
            meta = chunk["metadata"]
            context_lines.append(
                f"Chunk {chunk['chunk_id']} (conv {meta.get('conversation_idx')}): {chunk['chunk']}"
            )
        context = "\n\n".join(context_lines)

        response, generation_time = self.generate_answer(context, question)
        return {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "retrieved_chunks": top_chunks,
            "num_chunks": len(top_chunks),
            "generation_time": generation_time,
        }

    def process_data_file(self, file_path: str = DATASET_PATH) -> None:
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            qa_list = item.get("qa", [])
            for qa_item in qa_list:
                result = self.process_question(qa_item, idx)
                self.results[str(idx)].append(result)
            self._save()
        self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)


def main() -> None:
    searcher = RAGSearch(output_path="results/rag_search_results.json")
    searcher.process_data_file()


if __name__ == "__main__":
    main()
