import sys
import os
import json
import time
import pickle
from tqdm import tqdm
import uuid
from dotenv import load_dotenv
from google import genai
from google.genai import types 
from google.genai.errors import APIError

# Point Python to your parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from mem_kernel2_0 import (
    HypergraphMemoryOSV2,
    DateMemoryBlock,
    EntityMemoryBlock,
    FactMemoryBlock,
    PreferenceMemoryBlock,
    RelationshipMemoryBlock,
    GoalMemoryBlock,
    LocationMemoryBlock,     
    StateChangeMemoryBlock,  
    BaseMemoryBlock
)

load_dotenv()

# Initialize Gemini Client
client = genai.Client()
MODEL_ID = "gemini-2.5-flash"

def extract_memory_blocks(text: str, retries=3):
    """Processes a single conversational turn and extracts memory blocks."""
    prompt = f"""
    Extract key long-term conversational memories from the following text. 
    If there are no meaningful long-term details, return an empty array [].
    
    Output strictly as a JSON array of objects. Keys required:
    - "abstraction": Brief summary.
    - "value": Specific detail.
    - "contexts": 1-3 overarching themes.
    - "block_type": EXACTLY one of: "Temporal/Date", "Entity/Person/Pet", "Static Fact", "Preference/Trait", "Relationship", "Goal/Intention", "Spatial/Location", "State Change/Update".
    
    Optional keys if relevant to block_type:
    - "Temporal/Date": "raw_date"
    - "Entity/Person/Pet": "entity_name", "entity_type"
    - "Static Fact": "category"
    - "Relationship": "source_entity", "target_entity"
    - "Goal/Intention": "status"
    - "State Change/Update": "previous_state"
    
    Text: "{text}"
    """
    
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json" # Forces valid JSON; no parsing errors!
                )
            )
            
            data = json.loads(response.text.strip())
            
            blocks = []
            for item in data:
                b_type = item.get("block_type", "Generic")
                node_id = f"node_{uuid.uuid4().hex}"  
                
                # Deduplicate and normalize contexts
                raw_contexts = item.get('contexts', [])
                clean_contexts = list({str(c).strip().title() for c in raw_contexts})
                
                if b_type == "Temporal/Date": blocks.append(DateMemoryBlock(node_id, item.get('abstraction', ''), item.get('value', ''), clean_contexts, item.get('raw_date')))
                elif b_type == "Entity/Person/Pet": blocks.append(EntityMemoryBlock(node_id, item.get('abstraction', ''), item.get('value', ''), clean_contexts, item.get('entity_name'), item.get('entity_type')))
                elif b_type == "Static Fact": blocks.append(FactMemoryBlock(node_id, item.get('abstraction', ''), item.get('value', ''), clean_contexts, item.get('category')))
                elif b_type == "Preference/Trait": blocks.append(PreferenceMemoryBlock(node_id, item.get('abstraction', ''), item.get('value', ''), clean_contexts))
                elif b_type == "Relationship": blocks.append(RelationshipMemoryBlock(node_id, item.get('abstraction', ''), item.get('value', ''), clean_contexts, item.get('source_entity'), item.get('target_entity')))
                elif b_type == "Goal/Intention": blocks.append(GoalMemoryBlock(node_id, item.get('abstraction', ''), item.get('value', ''), clean_contexts, item.get('status', 'Active')))
                elif b_type == "Spatial/Location": blocks.append(LocationMemoryBlock(node_id, item.get('abstraction', ''), item.get('value', ''), clean_contexts))
                elif b_type == "State Change/Update": blocks.append(StateChangeMemoryBlock(node_id, item.get('abstraction', ''), item.get('value', ''), clean_contexts, item.get('previous_state')))
                else: blocks.append(BaseMemoryBlock(node_id, item.get('abstraction', ''), item.get('value', ''), clean_contexts))
            return blocks
            
        except APIError as e:
            if e.code == 429: 
                time.sleep((attempt + 1) * 5)
            else: break
        except Exception as e:
            print(f"Parse Error: {e}")
            break
            
    return []

def generate_agent_answer_gemini(question: str, memory_context: str):
    prompt = f"""
    Answer the question using ONLY the Memory Context. 
    Be as brief and direct as possible.
    If the answer is not in the context, say "I don't know".
    
    Context: {memory_context}
    Question: {question}
    Answer:
    """
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', 'locomo10.json') 
    output_path = os.path.join(os.path.dirname(__file__), 'results','locomo_results_gemini.json')
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_graphs')
    os.makedirs(save_dir, exist_ok=True)
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    all_results = []
    START_INDEX = 6
    
    # Resume from existing results
    if os.path.exists(output_path):
        print(f"Loading existing results from {output_path} to resume...")
        with open(output_path, 'r') as f:
            try:
                existing_results = json.load(f)
                # Keep only results generated before START_INDEX to avoid duplicates
                all_results = [res for res in existing_results if res.get("conversation_idx", 0) < START_INDEX]
            except json.JSONDecodeError:
                pass
    
    for idx, item in enumerate(data):
        # Skip conversations before the START_INDEX
        if idx < START_INDEX:
            continue
            
        print(f"\n{'='*60}")
        print(f"PROCESSING CONVERSATION {idx + 1}/{len(data)}")
        
        graph_path = os.path.join(save_dir, f'graph_conv_{idx}.pkl')
        
        if os.path.exists(graph_path):
            print(f"\n[Phase 1: SKIP] Loading previously saved hypergraph for conversation {idx}...")
            with open(graph_path, 'rb') as f:
                memos = pickle.load(f)
        else:
            memos = HypergraphMemoryOSV2()
            
            conversation = item.get("conversation", {})
            
            # --- PHASE 1: TURN-BY-TURN INGESTION ---
            print("\n[Phase 1: ADD] Ingesting conversation turn-by-turn...")
            
            for key, chats in conversation.items():
                if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key: continue
                
                print(f"  -> Processing {key}...")
                
                # Get the actual date for this session
                session_date_key = f"{key}_date_time"
                current_date = conversation.get(session_date_key, "Unknown Date")
                
                for chat in chats:
                    # Prepend the absolute date to the text
                    text_to_process = f"[Date: {current_date}] {chat.get('speaker', 'U')}: {chat.get('text', '')}"
                    
                    if len(text_to_process.strip()) < 15: 
                        continue 
                    
                    blocks = extract_memory_blocks(text_to_process)
                    for block in blocks:
                        memos.ingest_block(block)
                    
                    # Small sleep to manage rate limits while iterating line by line
                    time.sleep(1)
            
            # Save the populated hypergraph to disk
            with open(graph_path, 'wb') as f:
                pickle.dump(memos, f)
            print("  -> Hypergraph saved to disk!")
                
        qa_list = item.get("qa", [])
        
        # --- PHASE 2: SEARCH & GENERATE ---
        print(f"\n[Phase 2: SEARCH] Answering {len(qa_list)} questions...")
        
        for qa_item in tqdm(qa_list, desc="Testing"):
            question = qa_item.get("question", "")
            gold_answer = qa_item.get("answer", "")
            category = qa_item.get("category", -1)
            
            # Retrieve from V2 Hypergraph
            retrieved_envelope = memos.retrieve_envelope(question, top_k=3)
            
            # Print to see exactly what the Graph pulls out
            print(f"\nQ: {question}")
            print(f"Found Context: {retrieved_envelope}")
            
            # Answer via Gemini
            agent_response = generate_agent_answer_gemini(question, retrieved_envelope)
            
            all_results.append({
                "conversation_idx": idx,
                "question": question,
                "ground_truth": gold_answer,
                "generated_answer": agent_response,
                "type": str(category) 
            })
            
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=4)
            
    print(f"\nEvaluation complete! Responses saved to {output_path}")

if __name__ == "__main__":
    main()