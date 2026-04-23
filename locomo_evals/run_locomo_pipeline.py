import sys
import os
import json
import ollama
from tqdm import tqdm

# Point Python to your parent directory to import your kernel
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from memory_kernel import HypergraphMemoryOS
from extractor import extract_hypergraph_nodes

_DEFAULT_OLLAMA = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")


def generate_agent_answer(question: str, memory_context: str, model_name=None):
    if model_name is None:
        model_name = _DEFAULT_OLLAMA
    """
    The Search & Generate Phase: 
    Forces the agent to answer using only the retrieved Hypergraph envelope.
    """
    prompt = f"""
    You are an AI assistant taking a memory test about a user's conversation history. 
    Use ONLY the provided Memory Context to answer the Question.
    If the answer is not in the context, say "I don't know".
    
    Memory Context:
    {memory_context}
    
    Question: {question}
    Answer:
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0} 
        )
        return response.message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', 'locomo10.json')
    output_path = os.path.join(os.path.dirname(__file__), 'results','locomo_results.json')
    
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    all_results = []
    
    # Iterate through the dataset exactly as mem0 does
    for idx, item in enumerate(data):
        print(f"\n{'='*60}")
        print(f"PROCESSING CONVERSATION {idx + 1}/{len(data)}")
        print(f"{'='*60}")
        
        # Initialize a fresh Memory OS for this specific conversation
        memos = HypergraphMemoryOS()
        
        conversation = item.get("conversation", {})
        qa_list = item.get("qa", [])
        
        speaker_a = conversation.get("speaker_a", "User")
        speaker_b = conversation.get("speaker_b", "Assistant")
        
        # ---------------------------------------------------------
        # PHASE 1: THE "ADD" PHASE (Ingestion)
        # ---------------------------------------------------------
        print(f"\n[Phase 1: ADD] Ingesting history for {speaker_a} & {speaker_b}...")
        
        # Loop through the session keys (ignoring metadata keys)
        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
                
            chats = conversation[key]
            
            # Combine the chats into a single transcript block to pass to the extractor
            transcript_chunk = ""
            for chat in chats:
                transcript_chunk += f"{chat['speaker']}: {chat['text']}\n"
                
            # Ollama model (default Qwen3.5 4B-class tag); use vLLM baselines for HF IDs
            if transcript_chunk.strip():
                print(f"  -> Extracting facts from session: {key}...")
                nodes = extract_hypergraph_nodes(transcript_chunk)
                for mem in nodes:
                    memos.ingest_memory(mem.node_id, mem.abstraction, mem.value, mem.contexts)
                    
        # ---------------------------------------------------------
        # PHASE 2: THE "SEARCH" PHASE (Testing)
        # ---------------------------------------------------------
        print(f"\n[Phase 2: SEARCH] Answering {len(qa_list)} questions...")
        
        for qa_item in tqdm(qa_list, desc="Testing"):
            question = qa_item.get("question", "")
            gold_answer = qa_item.get("answer", "")
            category = qa_item.get("category", -1)
            
            # 1. Search the Hypergraph (The Dual-Graph Jump)
            retrieved_envelope = memos.retrieve_envelope(question)
            
            # 2. Generate the Agent's Response
            agent_response = generate_agent_answer(question, retrieved_envelope)
            
            # 3. Store for the LLM Judge
            all_results.append({
                "conversation_idx": idx,
                "question": question,
                "ground_truth": gold_answer,
                "generated_answer": agent_response,
                "type": str(category)  # The Judge script expects this key
            })
            
        # Save progress after every conversation
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=4)
            
    print(f"\nEvaluation complete! Responses saved to {output_path}")
    print("Next step: Run 'uv run locomo_evals/judge_locomo.py' to score the accuracy.")

if __name__ == "__main__":
    main()