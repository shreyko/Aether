import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from jinja2 import Template
from mem0 import Memory
from tqdm import tqdm

from .config import MEM0_CONFIG, VLLM_MODEL, get_vllm_client
from .prompts import ANSWER_PROMPT


class MemorySearch:
    def __init__(
        self,
        output_path: str = "results.json",
        top_k: int = 30,
        max_workers: int | None = None,
    ):
        self.mem = Memory.from_config(MEM0_CONFIG)
        self.top_k = top_k
        self.client = get_vllm_client()
        self.results: dict[str, list] = defaultdict(list)
        self.output_path = output_path
        self.max_workers = max_workers if max_workers is not None else int(os.getenv("MEM0_SEARCH_WORKERS", "32"))
        # Guard _save() and the shared results dict when futures complete out
        # of order across worker threads.
        self._lock = threading.Lock()

    def search_memory(self, user_id: str, query: str, max_retries: int = 3):
        start = time.time()
        memories = None
        for attempt in range(max_retries):
            try:
                memories = self.mem.search(
                    query, filters={"user_id": user_id}, top_k=self.top_k
                )
                break
            except Exception:
                if attempt >= max_retries - 1:
                    raise
                print(f"  Retrying search for {user_id}...")
                time.sleep(1)

        elapsed = time.time() - start
        formatted = []
        iterable = memories.get("results", memories) if isinstance(memories, dict) else memories
        for m in iterable or []:
            mem_obj = m if isinstance(m, dict) else m
            memory_text = mem_obj.get("memory", "")
            ts = mem_obj.get("metadata", {}).get("timestamp", "")
            score = round(mem_obj.get("score", 0.0), 2)
            formatted.append({"memory": memory_text, "timestamp": ts, "score": score})

        return formatted, elapsed

    def answer_question(self, speaker_a_uid: str, speaker_b_uid: str, question: str):
        # Fire the two per-speaker memory searches concurrently; they're
        # independent and together dominate the per-question latency when
        # the embedder is on GPU.
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
        # Cap output length. Without this, vLLM defaults to
        # max_model_len - prompt_tokens (~29k for our setup). At
        # temperature=0, Llama-3.2-3B occasionally falls into repetition
        # loops and burns the full budget on a single answer, pinning a
        # worker thread and its KV-cache slab for 5-25 minutes. LOCOMO
        # ground-truth answers are 1-3 sentences, so 512 tokens is
        # generous but bounded.
        response = self.client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
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

        # Pre-allocate per-conversation result slots so we can write results
        # back to their original positions regardless of completion order.
        slots: dict[int, list[dict | None]] = defaultdict(list)
        for idx, item in enumerate(data):
            slots[idx] = [None] * len(item["qa"])

        # Resume support: if an earlier run of the SEARCH phase wrote a
        # partial checkpoint to ``output_path`` (flushed every 200
        # completions), pre-fill the slots from it and skip those jobs.
        # Set MEM0_SEARCH_RESUME=0 to force a clean rerun.
        resume_enabled = os.getenv("MEM0_SEARCH_RESUME", "1") != "0"
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
                    # Sanity check: keys we always write.
                    if not isinstance(row, dict) or "response" not in row:
                        continue
                    slots[idx][qa_idx] = row
                    resumed_count += 1
            if resumed_count:
                print(f"  [resume] Loaded {resumed_count} pre-answered QAs from {self.output_path}")

        # Flatten remaining QAs across every conversation into one pool so we
        # can saturate vLLM regardless of per-conversation size. Skip any
        # (idx, qa_idx) already populated via the resume path above.
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
                    # Periodic checkpoint so a long run is recoverable without
                    # holding the full list in a single save at the end.
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
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
