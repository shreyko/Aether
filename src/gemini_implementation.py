import os
import json
from google import genai
from dotenv import load_dotenv

# Import the NEW kernel and blocks
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

class GeminiAPI:
    def __init__(self, api_key=None):
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        self.client = genai.Client()

    def process_text(self, text):
        prompt = f"""
        Extract key conversational memories from the following text. 
        Format the output strictly as a JSON array where each object has the following keys:
        - "node_id": A short, unique snake_case identifier.
        - "abstraction": A high-level summary of the memory.
        - "value": The specific details or values.
        - "contexts": An array of 1 to 3 overarching themes (e.g., "vacation", "finance").
        - "block_type": MUST be one of: "Temporal/Date", "Entity/Person/Pet", "Static Fact", "Preference/Trait", "Relationship", "Goal/Intention", "Spatial/Location", "State Change/Update".
        
        Depending on block_type, include these optional keys if relevant:
        - "Temporal/Date": "raw_date"
        - "Entity/Person/Pet": "entity_name", "entity_type"
        - "Static Fact": "category"
        - "Relationship": "source_entity", "target_entity"
        - "Goal/Intention": "status"
        - "State Change/Update": "previous_state"
        
        Text to process: "{text}"
        
        Return ONLY valid JSON without markdown wrapping.
        """
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash", # Updated to a stable model
            contents=prompt,
        )
        
        response_text = response.text.strip()
        
        if response_text.startswith("```json"): response_text = response_text[7:]
        elif response_text.startswith("```"): response_text = response_text[3:]
        if response_text.endswith("```"): response_text = response_text[:-3]
            
        try:
            data = json.loads(response_text.strip())
            blocks = []
            
            # Map JSON output to our new specific Memory Block classes
            for item in data:
                b_type = item.get("block_type", "Generic")
                if b_type == "Temporal/Date":
                    blocks.append(DateMemoryBlock(item['node_id'], item['abstraction'], item['value'], item['contexts'], item.get('raw_date')))
                elif b_type == "Entity/Person/Pet":
                    blocks.append(EntityMemoryBlock(item['node_id'], item['abstraction'], item['value'], item['contexts'], item.get('entity_name'), item.get('entity_type')))
                elif b_type == "Static Fact":
                    blocks.append(FactMemoryBlock(item['node_id'], item['abstraction'], item['value'], item['contexts'], item.get('category')))
                elif b_type == "Preference/Trait":
                    blocks.append(PreferenceMemoryBlock(item['node_id'], item['abstraction'], item['value'], item['contexts']))
                elif b_type == "Relationship":
                    blocks.append(RelationshipMemoryBlock(item['node_id'], item['abstraction'], item['value'], item['contexts'], item.get('source_entity'), item.get('target_entity')))
                elif b_type == "Goal/Intention":
                    blocks.append(GoalMemoryBlock(item['node_id'], item['abstraction'], item['value'], item['contexts'], item.get('status', 'Active')))
                elif b_type == "Spatial/Location":
                    blocks.append(LocationMemoryBlock(item['node_id'], item['abstraction'], item['value'], item['contexts']))
                elif b_type == "State Change/Update":
                    blocks.append(StateChangeMemoryBlock(item['node_id'], item['abstraction'], item['value'], item['contexts'], item.get('previous_state')))
                else:
                    blocks.append(BaseMemoryBlock(item['node_id'], item['abstraction'], item['value'], item['contexts']))
            
            return {"blocks": blocks}
            
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse Gemini JSON output: {e}\nRaw output: {response.text}")

def run_simulation():
    print("=== INITIALIZING MEMORY OS V2 ===")
    memos = HypergraphMemoryOSV2() # Using the new V2 Kernel
    
    api_key = os.getenv("GEMINI_API_KEY")  
    gemini_api = GeminiAPI(api_key)
    
    print("\n=== STARTING INGESTION (GEMINI API PROCESSING) ===")
    
    # Session 1: Travel plans
    chunk_1 = "I really want to take a vacation to Japan in April for the cherry blossoms. My absolute maximum budget is $3000, which includes the flights."
    print(f"\nProcessing Chunk 1: {chunk_1}")
    response_1 = gemini_api.process_text(chunk_1)
    for block in response_1['blocks']:
        memos.ingest_block(block) # Using the new ingest_block method
        
    # Session 2: Pet discussion
    chunk_2 = "By the way, since I'm leaving in April, I desperately need to find someone to watch my golden retriever, Max. He needs daily walks."
    print(f"\nProcessing Chunk 2: {chunk_2}")
    response_2 = gemini_api.process_text(chunk_2)
    for block in response_2['blocks']:
        memos.ingest_block(block)

    print("\n=== STARTING RETRIEVAL (DUAL HYPERGRAPH ACTIVATION) ===")
    query = "When is the user planning to go on vacation and what are their pet care needs?"
    print(f"User Query: '{query}'")
    
    agent_context = memos.retrieve_envelope(query)
    
    print("\n--- WHAT THE AGENT SEES IN ITS WORKING SET ---")
    print(agent_context)
    print("----------------------------------------------")
    print("Active Hyperedges:", memos.theme_to_edge_id)

if __name__ == "__main__":
    run_simulation()