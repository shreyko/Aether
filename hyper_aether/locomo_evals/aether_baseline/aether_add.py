"""ADD phase for the Aether LOCOMO baseline.

For each LOCOMO conversation we build *two* hypergraph memory kernels —
one per speaker — mirroring the (speaker_a_N, speaker_b_N) user-id scheme
that the mem0 baseline uses. Each speaker's transcript is chunked into
small turn batches, passed through the vLLM-backed extractor to produce
memory entries, and ingested into that speaker's hypergraph. The kernels
are pickled to ``aether_db/`` so the SEARCH phase (a separate process)
can reload them.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .config import (
    AETHER_ADD_BATCH_SIZE,
    AETHER_ADD_WORKERS,
    AETHER_DB_PATH,
    DEFAULT_EXTRACT_MAX_TOKENS,
    VLLM_MODEL,
    get_vllm_client,
)
from .extractor import extract_hypergraph_nodes
from .memory_kernel import HypergraphMemoryOS


def _kernel_path(user_id: str) -> str:
    return os.path.join(AETHER_DB_PATH, f"{user_id}.pkl")


def _format_chunk_text(messages: list[dict], timestamp: str) -> str:
    header = f"[{timestamp}]" if timestamp else ""
    lines = [header] if header else []
    for m in messages:
        lines.append(m["content"])
    return "\n".join(lines).strip()


class AetherADD:
    def __init__(
        self,
        data_path: str,
        batch_size: int | None = None,
        max_workers: int | None = None,
    ):
        self.data_path = data_path
        self.batch_size = batch_size if batch_size is not None else AETHER_ADD_BATCH_SIZE
        self.max_workers = max_workers if max_workers is not None else AETHER_ADD_WORKERS
        self.client = get_vllm_client()
        self.data: list[dict] | None = None
        if data_path:
            self.load_data()

        os.makedirs(AETHER_DB_PATH, exist_ok=True)

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def _build_speaker_jobs(self, item: dict, idx: int) -> list[tuple[str, list[tuple[list[dict], str]]]]:
        """Produce ``[(user_id, [(messages, timestamp), ...]), ...]`` for one conversation.

        Same formatting as the mem0 baseline so the two systems see the same
        per-speaker transcript view.
        """
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_uid = f"{speaker_a}_{idx}"
        speaker_b_uid = f"{speaker_b}_{idx}"

        batches_a: list[tuple[list[dict], str]] = []
        batches_b: list[tuple[list[dict], str]] = []

        for key in conversation:
            if key in ("speaker_a", "speaker_b") or "date" in key or "timestamp" in key:
                continue
            timestamp = conversation.get(f"{key}_date_time", "")
            chats = conversation[key]

            msgs_a: list[dict] = []
            msgs_b: list[dict] = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    msgs_a.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    msgs_b.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    msgs_a.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    msgs_b.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})

            for i in range(0, len(msgs_a), self.batch_size):
                batches_a.append((msgs_a[i : i + self.batch_size], timestamp))
            for i in range(0, len(msgs_b), self.batch_size):
                batches_b.append((msgs_b[i : i + self.batch_size], timestamp))

        return [(speaker_a_uid, batches_a), (speaker_b_uid, batches_b)]

    def _ingest_speaker_sequential(
        self,
        user_id: str,
        batches: list[tuple[list[dict], str]],
        retries: int = 2,
    ) -> tuple[str, int]:
        """Extract + ingest every batch for ``user_id`` in order.

        Ingestion is sequential per-user (kernel mutation is stateful), but
        multiple user_ids run concurrently via the outer thread pool, which
        keeps the vLLM server saturated.
        """
        kernel = HypergraphMemoryOS()
        ingested = 0
        for messages, timestamp in batches:
            chunk_text = _format_chunk_text(messages, timestamp)
            for attempt in range(retries + 1):
                memories = extract_hypergraph_nodes(
                    chunk_text,
                    client=self.client,
                    model_name=VLLM_MODEL,
                    max_tokens=DEFAULT_EXTRACT_MAX_TOKENS,
                )
                if memories or attempt >= retries:
                    break
                time.sleep(0.5)

            for mem in memories:
                kernel.ingest_memory(
                    node_id=mem.node_id,
                    abstraction=mem.abstraction,
                    value=mem.value,
                    contexts=mem.contexts,
                    metadata={"timestamp": timestamp, "user_id": user_id},
                )
                ingested += 1

        kernel.save(_kernel_path(user_id))
        return user_id, ingested

    def process_all_conversations(self):
        if not self.data:
            raise ValueError("No data loaded.")

        if os.getenv("AETHER_SKIP_DELETE_ALL", "0") == "1":
            print("AETHER_SKIP_DELETE_ALL=1; reusing existing aether_db/ contents.")
        else:
            # Wipe stale kernels so a rerun starts from a clean slate.
            if os.path.isdir(AETHER_DB_PATH):
                for name in os.listdir(AETHER_DB_PATH):
                    path = os.path.join(AETHER_DB_PATH, name)
                    try:
                        if os.path.isfile(path) or os.path.islink(path):
                            os.unlink(path)
                        else:
                            shutil.rmtree(path, ignore_errors=True)
                    except OSError:
                        pass
            os.makedirs(AETHER_DB_PATH, exist_ok=True)

        all_jobs: list[tuple[str, list[tuple[list[dict], str]]]] = []
        for idx, item in enumerate(self.data):
            all_jobs.extend(self._build_speaker_jobs(item, idx))

        total_batches = sum(len(b) for _, b in all_jobs)
        print(
            f"[Aether-ADD] Ingesting {total_batches} chunks across {len(all_jobs)} user_ids "
            f"with {self.max_workers} workers (batch_size={self.batch_size})...",
            flush=True,
        )

        _lock = threading.Lock()
        summary: list[tuple[str, int]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._ingest_speaker_sequential, uid, batches): (uid, len(batches))
                for uid, batches in all_jobs
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Ingesting user_ids"):
                uid, _n = futures[fut]
                try:
                    result = fut.result()
                    with _lock:
                        summary.append(result)
                except Exception as e:
                    print(f"[Aether-ADD] user_id={uid} failed: {e}", flush=True)
                    raise

        total_ingested = sum(n for _, n in summary)
        print(
            f"[Aether-ADD] Done: {total_ingested} memories ingested across {len(all_jobs)} user_ids.",
            flush=True,
        )
