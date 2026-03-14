from memory_kernel import HypergraphMemoryOS
from extractor import extract_hypergraph_nodes, Backend


def run_simulation():
    BACKEND = Backend.VLLM
    MODEL = "Qwen/Qwen3.5-9B"

    llm = None
    if BACKEND == Backend.VLLM:
        from vllm import LLM
        print(f"=== LOADING vLLM MODEL: {MODEL} ===")
        llm = LLM(model=MODEL)

    print("=== INITIALIZING MEMORY OS ===")
    memos = HypergraphMemoryOS()

    print("\n=== STARTING INGESTION (LLM PROCESSING) ===")

    chunk_1 = "I really want to take a vacation to Japan in April for the cherry blossoms. My absolute maximum budget is $3000, which includes the flights."
    print(f"\nProcessing Chunk 1: {chunk_1}")
    nodes_1 = extract_hypergraph_nodes(
        chunk_1, model_name=MODEL, backend=BACKEND, llm=llm,
    )
    for mem in nodes_1:
        memos.ingest_memory(mem.node_id, mem.abstraction, mem.value, mem.contexts)

    chunk_2 = "By the way, since I'm leaving in April, I desperately need to find someone to watch my golden retriever, Max. He needs daily walks."
    print(f"\nProcessing Chunk 2: {chunk_2}")
    nodes_2 = extract_hypergraph_nodes(
        chunk_2, model_name=MODEL, backend=BACKEND, llm=llm,
    )
    for mem in nodes_2:
        memos.ingest_memory(mem.node_id, mem.abstraction, mem.value, mem.contexts)

    print("\n=== STARTING RETRIEVAL (DUAL HYPERGRAPH ACTIVATION) ===")
    query = "What is the name of the user"
    print(f"User Query: '{query}'")

    agent_context = memos.retrieve_envelope(query)

    print("\n--- WHAT THE AGENT SEES IN ITS WORKING SET ---")
    print(agent_context)
    print("----------------------------------------------")
    print(memos.theme_to_edge_id)


if __name__ == "__main__":
    run_simulation()
