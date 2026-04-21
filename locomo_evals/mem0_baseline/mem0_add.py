import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from mem0 import Memory
from tqdm import tqdm

from .config import CUSTOM_INSTRUCTIONS, MEM0_CONFIG


class MemoryADD:
    def __init__(
        self,
        data_path: str,
        batch_size: int | None = None,
        max_workers: int | None = None,
    ):
        self.mem = Memory.from_config(MEM0_CONFIG)
        # Bigger batch_size => fewer mem0.add() round-trips (each of which
        # fires ~2 LLM calls internally). Override via $MEM0_ADD_BATCH_SIZE.
        if batch_size is None:
            batch_size = int(os.getenv("MEM0_ADD_BATCH_SIZE", "8"))
        self.batch_size = batch_size
        # Parallelism across independent user_ids. Each conversation yields
        # 2 user_ids (speaker_a_N / speaker_b_N), all fully independent in
        # mem0's chroma + vLLM backend.
        if max_workers is None:
            max_workers = int(os.getenv("MEM0_ADD_WORKERS", "10"))
        self.max_workers = max_workers
        self.data_path = data_path
        self.data = None
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id: str, messages: list[dict], metadata: dict, retries: int = 3):
        for attempt in range(retries):
            try:
                # mem0 OSS SDK has varied parameter names across versions.
                # Try custom_instructions first, then instructions, then plain add.
                try:
                    self.mem.add(
                        messages,
                        user_id=user_id,
                        metadata=metadata,
                        custom_instructions=CUSTOM_INSTRUCTIONS,
                    )
                except TypeError as te:
                    if "custom_instructions" in str(te) and "unexpected keyword argument" in str(te):
                        try:
                            self.mem.add(
                                messages,
                                user_id=user_id,
                                metadata=metadata,
                                instructions=CUSTOM_INSTRUCTIONS,
                            )
                        except TypeError as te2:
                            if "instructions" in str(te2) and "unexpected keyword argument" in str(te2):
                                self.mem.add(
                                    messages,
                                    user_id=user_id,
                                    metadata=metadata,
                                )
                            else:
                                raise
                    else:
                        raise
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise e

    def _ingest_speaker_sequential(self, user_id: str, batches: list[tuple[list[dict], str]]):
        """Ingest all (batch, timestamp) pairs for a single user_id in order.

        We keep this sequential per user_id because mem0's add flow is
        stateful (extract facts -> look up similar -> decide add/update),
        and parallel writes against the same user_id can race on the
        "similar memory" lookup. Parallelism across user_ids is safe.
        """
        for messages, timestamp in batches:
            self.add_memory(user_id, messages, metadata={"timestamp": timestamp})

    def _build_speaker_jobs(self, item: dict, idx: int) -> list[tuple[str, list[tuple[list[dict], str]]]]:
        """Produce [(user_id, [(batch, timestamp), ...]), ...] for one conversation."""
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

    def process_all_conversations(self):
        if not self.data:
            raise ValueError("No data loaded.")

        # Clear existing memories for every user_id serially first so we
        # don't race with in-flight writes from the worker pool. This is a
        # no-op on a fresh mem0_db, but can be slow (minutes) if the
        # underlying chroma DB has thousands of stale vectors from a
        # previous run. Set MEM0_SKIP_DELETE_ALL=1 if you've already wiped
        # mem0_db on disk and want to skip this entirely.
        if os.getenv("MEM0_SKIP_DELETE_ALL", "0") == "1":
            print("MEM0_SKIP_DELETE_ALL=1; skipping pre-wipe of existing memories.")
        else:
            print(
                f"Clearing existing memories for {len(self.data)} conversations "
                "(20 serial delete_all calls)...",
                flush=True,
            )
            for idx, item in enumerate(self.data):
                conv = item["conversation"]
                self.mem.delete_all(user_id=f"{conv['speaker_a']}_{idx}")
                self.mem.delete_all(user_id=f"{conv['speaker_b']}_{idx}")
                print(f"  cleared conv {idx + 1}/{len(self.data)}", flush=True)

        # Build the full (user_id, batches) job list across all conversations.
        all_jobs: list[tuple[str, list[tuple[list[dict], str]]]] = []
        for idx, item in enumerate(self.data):
            all_jobs.extend(self._build_speaker_jobs(item, idx))

        total_batches = sum(len(b) for _, b in all_jobs)
        print(
            f"Ingesting {total_batches} batches across {len(all_jobs)} user_ids "
            f"with {self.max_workers} workers (batch_size={self.batch_size})..."
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._ingest_speaker_sequential, uid, batches): (uid, len(batches))
                for uid, batches in all_jobs
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Ingesting user_ids"):
                uid, n = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[ADD] user_id={uid} failed after {n} batches: {e}")
                    raise

        print(f"[ADD] Done: {total_batches} batches across {len(all_jobs)} user_ids.")
