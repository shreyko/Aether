import json
import time

from mem0 import Memory
from tqdm import tqdm

from .config import CUSTOM_INSTRUCTIONS, MEM0_CONFIG


class MemoryADD:
    def __init__(self, data_path: str, batch_size: int = 2):
        self.mem = Memory.from_config(MEM0_CONFIG)
        self.batch_size = batch_size
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
                # Prefer `custom_instructions` (current API) and fall back to `instructions`.
                try:
                    self.mem.add(
                        messages,
                        user_id=user_id,
                        metadata=metadata,
                        custom_instructions=CUSTOM_INSTRUCTIONS,
                    )
                except TypeError as te:
                    if "custom_instructions" in str(te) and "unexpected keyword argument" in str(te):
                        self.mem.add(
                            messages,
                            user_id=user_id,
                            metadata=metadata,
                            instructions=CUSTOM_INSTRUCTIONS,
                        )
                    else:
                        raise
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise e

    def add_memories_for_speaker(self, speaker: str, messages: list[dict], timestamp: str, desc: str):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch, metadata={"timestamp": timestamp})

    def process_conversation(self, item: dict, idx: int):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_uid = f"{speaker_a}_{idx}"
        speaker_b_uid = f"{speaker_b}_{idx}"

        self.mem.delete_all(user_id=speaker_a_uid)
        self.mem.delete_all(user_id=speaker_b_uid)

        for key in conversation:
            if key in ("speaker_a", "speaker_b") or "date" in key or "timestamp" in key:
                continue

            date_key = f"{key}_date_time"
            timestamp = conversation.get(date_key, "")
            chats = conversation[key]

            messages_a: list[dict] = []
            messages_b: list[dict] = []

            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages_a.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_b.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages_a.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_b.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})

            self.add_memories_for_speaker(
                speaker_a_uid, messages_a, timestamp, f"[Conv {idx}] Adding memories for {speaker_a}"
            )
            self.add_memories_for_speaker(
                speaker_b_uid, messages_b, timestamp, f"[Conv {idx}] Adding memories for {speaker_b}"
            )

        print(f"[Conv {idx}] Memories added for {speaker_a_uid} and {speaker_b_uid}")

    def process_all_conversations(self):
        if not self.data:
            raise ValueError("No data loaded.")
        for idx, item in enumerate(self.data):
            print(f"\n{'=' * 60}")
            print(f"PROCESSING CONVERSATION {idx + 1}/{len(self.data)}")
            print(f"{'=' * 60}")
            self.process_conversation(item, idx)
