import xgi
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class HypergraphMemoryOS:
    def __init__(self):
        self.H = xgi.Hypergraph()
        self.memory_store = {}
        
        print("[Kernel] Loading local embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Node trackers
        self.node_embeddings = {}
        self.node_ids = []
        
        # --- NEW: Edge trackers for Semantic Consolidation ---
        self.theme_to_edge_id = {}    # Maps standard theme string -> XGI Edge ID
        self.edge_embeddings = {}     # Maps standard theme string -> Vector Embedding
        self.similarity_threshold = 0.65 # Threshold to merge edges (tuneable parameter)

    def _find_similar_edge(self, proposed_context: str):
        """Checks if a semantically similar hyperedge already exists."""
        if not self.edge_embeddings:
            return None
            
        query_embed = self.embedder.encode(proposed_context)
        themes = list(self.edge_embeddings.keys())
        embeddings_matrix = np.array(list(self.edge_embeddings.values()))
        
        similarities = cosine_similarity([query_embed], embeddings_matrix)[0]
        best_idx = np.argmax(similarities)
        
        # If the similarity is above our threshold, return the existing theme name
        if similarities[best_idx] >= self.similarity_threshold:
            return themes[best_idx]
            
        return None

    def ingest_memory(self, node_id: str, abstraction: str, value: str, contexts: list[str]):
        """Phase 1: Writing to the Hypergraph with Edge Consolidation"""
        self.memory_store[node_id] = {
            "abstraction": abstraction,
            "value": value
        }
        
        self.H.add_node(node_id)
        
        # --- Track the actual edge names we end up using ---
        resolved_contexts = [] 
        
        # Process the Hyperedges
        for proposed_context in contexts:
            # 1. Check if a similar edge already exists
            matched_theme = self._find_similar_edge(proposed_context)
            
            if matched_theme:
                # Merge into the existing edge
                edge_id = self.theme_to_edge_id[matched_theme]
                self.H.add_node_to_edge(edge_id, node_id)
                print(f"    [Merge] '{proposed_context}' consolidated into -> '{matched_theme}'")
                
                resolved_contexts.append(matched_theme) # Track the merged name
            else:
                # Create a brand new edge
                self.H.add_edge([node_id])
                new_edge_id = list(self.H.edges)[-1]
                
                # Save the new theme and its embedding
                self.theme_to_edge_id[proposed_context] = new_edge_id
                self.edge_embeddings[proposed_context] = self.embedder.encode(proposed_context)
                
                resolved_contexts.append(proposed_context) # Track the new name
                
        # Update Node embeddings for standard retrieval
        embedding = self.embedder.encode(abstraction)
        self.node_embeddings[node_id] = embedding
        if node_id not in self.node_ids:
            self.node_ids.append(node_id)
            
        # Print the resolved list instead of the raw input
        print(f"[Kernel] Ingested '{node_id}' into hyperedges: {resolved_contexts}")

    def _find_semantic_seed(self, query: str):
        """Finds the closest memory vertex to start the jump."""
        if not self.node_ids:
            return None
            
        query_embed = self.embedder.encode(query)
        embeddings_matrix = np.array([self.node_embeddings[nid] for nid in self.node_ids])
        
        similarities = cosine_similarity([query_embed], embeddings_matrix)[0]
        best_idx = np.argmax(similarities)
        return self.node_ids[best_idx]

    def retrieve_envelope(self, query: str):
        """Phase 2: Dual-Space Retrieval"""
        seed_node = self._find_semantic_seed(query)
        if not seed_node:
            return "Memory is empty."
            
        active_contexts = list(self.H.nodes.memberships(seed_node))
        
        activated_nodes = set()
        for context in active_contexts:
            members = self.H.edges.members(context)
            activated_nodes.update(members)
            
        working_set = []
        for n_id in activated_nodes:
            data = self.memory_store[n_id]
            formatted = f"[{n_id}] {data['abstraction']} (Details: {data['value']})"
            working_set.append(formatted)
            
        return "\n".join(working_set)