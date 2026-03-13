from memory_kernel import HypergraphMemoryOS
from extractor import extract_hypergraph_nodes

def run_simulation():
    # Make sure Ollama is running and has this model pulled!
    LOCAL_MODEL = "llama3.2" 
    
    print("=== INITIALIZING MEMORY OS ===")
    memos = HypergraphMemoryOS()
    
    print("\n=== STARTING INGESTION (LOCAL LLM PROCESSING) ===")
    # Session 1: Travel plans
    chunk_1 = "I really want to take a vacation to Japan in April for the cherry blossoms. My absolute maximum budget is $3000, which includes the flights."
    print(f"\nProcessing Chunk 1: {chunk_1}")
    nodes_1 = extract_hypergraph_nodes(chunk_1, model_name=LOCAL_MODEL)
    for mem in nodes_1:
        memos.ingest_memory(mem.node_id, mem.abstraction, mem.value, mem.contexts)
        
    # Session 2: Pet discussion
    chunk_2 = "By the way, since I'm leaving in April, I desperately need to find someone to watch my golden retriever, Max. He needs daily walks."
    print(f"\nProcessing Chunk 2: {chunk_2}")
    nodes_2 = extract_hypergraph_nodes(chunk_2, model_name=LOCAL_MODEL)
    for mem in nodes_2:
        memos.ingest_memory(mem.node_id, mem.abstraction, mem.value, mem.contexts)

    print("\n=== STARTING RETRIEVAL (DUAL HYPERGRAPH ACTIVATION) ===")
    # The agent asks a question that seems only about the dog.
    query = "What kind of dog does the user have?"
    print(f"User Query: '{query}'")
    
    # The Memory OS retrieves the full situational envelope
    agent_context = memos.retrieve_envelope(query)
    
    print("\n--- WHAT THE AGENT SEES IN ITS WORKING SET ---")
    print(agent_context)
    print("----------------------------------------------")
    print(memos.theme_to_edge_id)
   
if __name__ == "__main__":
    run_simulation()