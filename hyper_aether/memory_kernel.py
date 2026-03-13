import xgi
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class HypergraphMemoryOS:
    def __init__(self):
        # The core Hypergraph from XGI
        self.H = xgi.Hypergraph()
        
        # Storage for the Abstraction + Value content
        self.memory_store = {}
        # A strict map to force strings to match exact edge IDs
        self.theme_to_edge_id = {}
        
        # Local, lightweight embedding model for the initial "Seed Search"
        print("[Kernel] Loading local embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.node_embeddings = {}
        self.node_ids = []

    def ingest_memory(self, node_id: str, abstraction: str, value: str, contexts: list[str]):
        """Phase 1: Writing to the Hypergraph"""
        self.memory_store[node_id] = {
            "abstraction": abstraction,
            "value": value
        }
        
        self.H.add_node(node_id)
        
        for context in contexts:
            if context in self.theme_to_edge_id:
                # Force the node into the exact same integer edge ID
                edge_id = self.theme_to_edge_id[context]
                self.H.add_node_to_edge(edge_id, node_id)
            else:
                # Create a new edge, let XGI assign an int, and save it
                self.H.add_edge([node_id])
                new_edge_id = list(self.H.edges)[-1] # Grab the newly created ID
                self.theme_to_edge_id[context] = new_edge_id
      
        embedding = self.embedder.encode(abstraction)
        self.node_embeddings[node_id] = embedding
        if node_id not in self.node_ids:
            self.node_ids.append(node_id)
        
        print(f"[Kernel] Ingested '{node_id}' into hyperedges: {contexts}")

    def _find_semantic_seed(self, query: str):
        """Finds the single closest memory vertex using Cosine Similarity."""
        if not self.node_ids:
            return None
            
        query_embed = self.embedder.encode(query)
        embeddings_matrix = np.array([self.node_embeddings[nid] for nid in self.node_ids])
        
        similarities = cosine_similarity([query_embed], embeddings_matrix)[0]
        best_idx = np.argmax(similarities)
        return self.node_ids[best_idx]

    def retrieve_envelope(self, query: str):
        """Phase 2: Dual-Space Retrieval (The Context Switch)"""
        seed_node = self._find_semantic_seed(query)
        if not seed_node:
            return "Memory is empty."
            
        print(f"\n[Retrieval] 1. Seed Node triggered: {seed_node} -> '{self.memory_store[seed_node]['abstraction']}'")
        
        # Dual Graph Jump: Find active Hyperedges for this node
        active_contexts = list(self.H.nodes.memberships(seed_node))
        print(f"[Retrieval] 2. Activating Contexts (Hyperedges): {active_contexts}")
        
        # Activate the full envelope
        activated_nodes = set()
        for context in active_contexts:
            members = self.H.edges.members(context)
            activated_nodes.update(members)
            
        print(f"[Retrieval] 3. Envelope extracted: {len(activated_nodes)} interconnected nodes loaded.")
        
        # Package the Abstractions and Values for the Agent
        working_set = []
        for n_id in activated_nodes:
            data = self.memory_store[n_id]
            formatted = f"[{n_id}] {data['abstraction']} (Details: {data['value']})"
            working_set.append(formatted)
            
        return "\n".join(working_set)