import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from jinja2 import Template
from mem0 import Memory
from tqdm import tqdm

from ..latency import write_search_latency_summary
from ..mem0_baseline.prompts import ANSWER_PROMPT

from .config import MEM0G_CONFIG, VLLM_MODEL, get_vllm_client


class MemorySearch:
    """SEARCH phase for mem0 **graph** (Kuzu) — mirrors the vector mem0 baseline output schema."""

    def __init__(
        self,
        output_path: str = "results.json",
        top_k: int = 30,
        max_workers: int | None = None,
    ):
        self.mem = Memory.from_config(MEM0G_CONFIG)
        self.top_k = top_k
        self.client = get_vllm_client()
        self.results: dict[str, list] = defaultdict(list)
        self.output_path = output_path
        self.max_workers = max_workers if max_workers is not None else int(
            os.getenv("MEM0G_SEARCH_WORKERS", os.getenv("MEM0_SEARCH_WORKERS", "32"))
        )
        self._lock = threading.Lock()

    def search_memory(self, user_id: str, query: str, max_retries: int = 3):
        start = time.time()
        memories = None

        def _invoke_search():
            """mem0ai 1.x ``Memory.search`` expects ``user_id=...`` for OSS; bare
            ``filters={\"user_id\": ...}`` can raise ValidationError ('user_id required')."""
            try:
                return self.mem.search(
                    query,
                    user_id=user_id,
                    top_k=self.top_k,
                )
            except TypeError:
                pass
            try:
                return self.mem.search(query, user_id=user_id, limit=self.top_k)
            except TypeError:
                pass
            try:
                return self.mem.search(
                    query,
                    filters={"user_id": user_id},
                    top_k=self.top_k,
                )
            except TypeError:
                return self.mem.search(
                    query,
                    filters={"user_id": user_id},
                    limit=self.top_k,
                )

        for attempt in range(max_retries):
            try:
                memories = _invoke_search()
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
            max_tokens=512,
        )
        response_time = time.time() - t1

        content = ""
        try:
            content = response.choices[0].message.content or ""
        except Exception:
            content = response.choices[0]["message"]["content"] if response.choices else ""

        return (
            content,
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

        search_latency = max(sp1_time, sp2_time)
        total_latency = search_latency + resp_time

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
            "search_latency_sec": search_latency,
            "generation_latency_sec": resp_time,
            "total_latency_sec": total_latency,
        }

    def process_data_file(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

        slots: dict[int, list[dict | None]] = defaultdict(list)
        for idx, item in enumerate(data):
            slots[idx] = [None] * len(item["qa"])

        resume_enabled = os.getenv("MEM0G_SEARCH_RESUME", os.getenv("MEM0_SEARCH_RESUME", "1")) != "0"
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
                summary_path = write_search_latency_summary(self.output_path, self.results, baseline="mem0g")
                print(f"  [latency] Wrote search latency summary to {summary_path}")
            return

        def _worker(job):
            idx, qa_idx, sa_uid, sb_uid, qa_item = job
            result = self.process_question(qa_item, sa_uid, sb_uid)
            return idx, qa_idx, result

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(_worker, j) for j in jobs]
            completed = 0
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Answering QAs (mem0g)"):
                idx, qa_idx, result = fut.result()
                with self._lock:
                    slots[idx][qa_idx] = result
                    completed += 1
                    if completed % 200 == 0:
                        self._flush_slots(slots)

        with self._lock:
            self._flush_slots(slots)
            summary_path = write_search_latency_summary(self.output_path, self.results, baseline="mem0g")
            print(f"  [latency] Wrote search latency summary to {summary_path}")

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
