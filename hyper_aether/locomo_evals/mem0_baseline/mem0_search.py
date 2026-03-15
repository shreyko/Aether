import json
import time
from collections import defaultdict

from jinja2 import Template
from mem0 import Memory
from tqdm import tqdm

from .config import MEM0_CONFIG, VLLM_MODEL, get_vllm_client
from .prompts import ANSWER_PROMPT


class MemorySearch:
    def __init__(self, output_path: str = "results.json", top_k: int = 30):
        self.mem = Memory.from_config(MEM0_CONFIG)
        self.top_k = top_k
        self.client = get_vllm_client()
        self.results: dict[str, list] = defaultdict(list)
        self.output_path = output_path

    def search_memory(self, user_id: str, query: str, max_retries: int = 3):
        start = time.time()
        for attempt in range(max_retries):
            try:
                memories = self.mem.search(query, user_id=user_id, limit=self.top_k)
                break
            except Exception as e:
                if attempt >= max_retries - 1:
                    raise
                print(f"  Retrying search for {user_id}...")
                time.sleep(1)

        elapsed = time.time() - start
        formatted = []
        for m in memories.get("results", memories) if isinstance(memories, dict) else memories:
            mem_obj = m if isinstance(m, dict) else m
            memory_text = mem_obj.get("memory", "")
            ts = mem_obj.get("metadata", {}).get("timestamp", "")
            score = round(mem_obj.get("score", 0.0), 2)
            formatted.append({"memory": memory_text, "timestamp": ts, "score": score})

        return formatted, elapsed

    def answer_question(self, speaker_a_uid: str, speaker_b_uid: str, question: str):
        sp1_memories, sp1_time = self.search_memory(speaker_a_uid, question)
        sp2_memories, sp2_time = self.search_memory(speaker_b_uid, question)

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

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_uid = f"{speaker_a}_{idx}"
            speaker_b_uid = f"{speaker_b}_{idx}"

            qa = item["qa"]
            for qa_item in tqdm(qa, total=len(qa), desc=f"  Questions (conv {idx})", leave=False):
                result = self.process_question(qa_item, speaker_a_uid, speaker_b_uid)
                self.results[str(idx)].append(result)

            self._save()

        self._save()

    def _save(self):
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
