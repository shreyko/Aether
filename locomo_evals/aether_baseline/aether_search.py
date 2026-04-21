"""SEARCH phase for the Aether LOCOMO baseline.

For each (conversation, QA) pair we:

1. Load the two per-speaker hypergraph kernels produced by the ADD phase.
2. Retrieve a ``top_k`` envelope of memories from each kernel for the
   question.
3. Format them into the shared ANSWER_PROMPT template and call vLLM to
   generate a short answer.

The public interface (``MemorySearch.process_data_file``) mirrors the mem0
baseline so the runner / eval / scores stages are interchangeable.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from jinja2 import Template
from tqdm import tqdm

from .config import (
    AETHER_DB_PATH,
    AETHER_SEARCH_WORKERS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_K,
    VLLM_MODEL,
    get_vllm_client,
)
from .memory_kernel import HypergraphMemoryOS
from .prompts import ANSWER_PROMPT


def _kernel_path(user_id: str) -> str:
    return os.path.join(AETHER_DB_PATH, f"{user_id}.pkl")


class MemorySearch:
    def __init__(
        self,
        output_path: str = "results.json",
        top_k: int = DEFAULT_TOP_K,
        max_workers: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.output_path = output_path
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.client = get_vllm_client()
        self.max_workers = max_workers if max_workers is not None else AETHER_SEARCH_WORKERS

        self.results: dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

        # Cache loaded kernels across QAs on the same conversation. Kernels
        # are read-only during SEARCH so sharing is safe.
        self._kernels: dict[str, HypergraphMemoryOS] = {}
        self._kernel_lock = threading.Lock()

    def _get_kernel(self, user_id: str) -> HypergraphMemoryOS:
        with self._kernel_lock:
            cached = self._kernels.get(user_id)
            if cached is not None:
                return cached
        path = _kernel_path(user_id)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No saved aether kernel at {path}. Run the ADD phase first."
            )
        kernel = HypergraphMemoryOS.load(path)
        with self._kernel_lock:
            self._kernels.setdefault(user_id, kernel)
            return self._kernels[user_id]

    def search_memory(self, user_id: str, query: str, max_retries: int = 2):
        start = time.time()
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                kernel = self._get_kernel(user_id)
                envelope = kernel.retrieve_envelope(query, top_k=self.top_k)
                break
            except Exception as e:
                last_err = e
                if attempt >= max_retries:
                    raise
                time.sleep(0.5)
        else:  # pragma: no cover
            raise last_err  # type: ignore[misc]

        elapsed = time.time() - start
        formatted = []
        for m in envelope:
            ts = m.get("metadata", {}).get("timestamp", "")
            formatted.append(
                {
                    "memory": f"{m['abstraction']} (Details: {m['value']})",
                    "timestamp": ts,
                    "score": round(float(m.get("score", 0.0)), 2),
                    "node_id": m.get("node_id", ""),
                }
            )
        return formatted, elapsed

    def answer_question(self, speaker_a_uid: str, speaker_b_uid: str, question: str):
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(self.search_memory, speaker_a_uid, question)
            fut_b = pool.submit(self.search_memory, speaker_b_uid, question)
            sp1_memories, sp1_time = fut_a.result()
            sp2_memories, sp2_time = fut_b.result()

        search_1 = [f"{m['timestamp']}: {m['memory']}" for m in sp1_memories]
        search_2 = [f"{m['timestamp']}: {m['memory']}" for m in sp2_memories]

        template = Template(ANSWER_PROMPT)
        prompt = template.render(
            speaker_1_user_id=speaker_a_uid.split("_")[0],
            speaker_2_user_id=speaker_b_uid.split("_")[0],
            speaker_1_memories=json.dumps(search_1, indent=4),
            speaker_2_memories=json.dumps(search_2, indent=4),
            question=question,
        )

        t1 = time.time()
        response = self.client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        response_time = time.time() - t1

        return (
            response.choices[0].message.content,
            sp1_memories,
            sp2_memories,
            sp1_time,
            sp2_time,
            response_time,
        )

    def process_question(self, qa_item: dict, speaker_a_uid: str, speaker_b_uid: str):
        question = qa_item.get("question", "")
        answer = qa_item.get("answer", "")
        category = qa_item.get("category", -1)
        evidence = qa_item.get("evidence", [])

        response, sp1_mems, sp2_mems, sp1_time, sp2_time, resp_time = self.answer_question(
            speaker_a_uid, speaker_b_uid, question
        )

        return {
            "question": question,
            "answer": str(answer),
            "category": category,
            "evidence": evidence,
            "response": response,
            "speaker_1_memories": sp1_mems,
            "speaker_2_memories": sp2_mems,
            "num_speaker_1_memories": len(sp1_mems),
            "num_speaker_2_memories": len(sp2_mems),
            "speaker_1_memory_time": sp1_time,
            "speaker_2_memory_time": sp2_time,
            "response_time": resp_time,
        }

    def process_data_file(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

        slots: dict[int, list[dict | None]] = defaultdict(list)
        for idx, item in enumerate(data):
            slots[idx] = [None] * len(item["qa"])

        resume_enabled = os.getenv("AETHER_SEARCH_RESUME", "1") != "0"
        resumed_count = 0
        if resume_enabled and os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r") as f:
                    prior = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  [resume] Ignoring unreadable checkpoint {self.output_path}: {exc}")
                prior = {}
            for key, rows in prior.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                if idx not in slots:
                    continue
                for qa_idx, row in enumerate(rows):
                    if qa_idx >= len(slots[idx]):
                        break
                    if row is None:
                        continue
                    if not isinstance(row, dict) or "response" not in row:
                        continue
                    slots[idx][qa_idx] = row
                    resumed_count += 1
            if resumed_count:
                print(f"  [resume] Loaded {resumed_count} pre-answered QAs from {self.output_path}")

        jobs: list[tuple[int, int, str, str, dict]] = []
        for idx, item in enumerate(data):
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            speaker_a_uid = f"{speaker_a}_{idx}"
            speaker_b_uid = f"{speaker_b}_{idx}"
            for qa_idx, qa_item in enumerate(item["qa"]):
                if slots[idx][qa_idx] is not None:
                    continue
                jobs.append((idx, qa_idx, speaker_a_uid, speaker_b_uid, qa_item))

        if not jobs:
            print("  [resume] All QAs already answered; writing final output only.")
            with self._lock:
                self._flush_slots(slots)
            return

        def _worker(job):
            idx, qa_idx, sa_uid, sb_uid, qa_item = job
            result = self.process_question(qa_item, sa_uid, sb_uid)
            return idx, qa_idx, result

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(_worker, j) for j in jobs]
            completed = 0
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Answering QAs"):
                idx, qa_idx, result = fut.result()
                with self._lock:
                    slots[idx][qa_idx] = result
                    completed += 1
                    if completed % 200 == 0:
                        self._flush_slots(slots)

        with self._lock:
            self._flush_slots(slots)

    def _flush_slots(self, slots: dict[int, list[dict | None]]):
        self.results = defaultdict(list)
        for idx in sorted(slots.keys()):
            for row in slots[idx]:
                if row is not None:
                    self.results[str(idx)].append(row)
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
