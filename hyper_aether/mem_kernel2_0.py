import xgi
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- MEMORY BLOCK CLASSES ---

class BaseMemoryBlock:
    def __init__(self, node_id: str, abstraction: str, value: str, contexts: list[str]):
        self.node_id = node_id
        self.abstraction = abstraction
        self.value = value
        self.contexts = contexts
        self.block_type = "Generic"

    def get_embedding_string(self) -> str:
        """Returns the string to be vectorized. Overridden by subclasses to boost semantics."""
        return f"[{self.block_type}] {self.abstraction}"

class DateMemoryBlock(BaseMemoryBlock):
    def __init__(self, node_id: str, abstraction: str, value: str, contexts: list[str], raw_date: str = None):
        super().__init__(node_id, abstraction, value, contexts)
        self.block_type = "Temporal/Date"
        self.raw_date = raw_date # Could be parsed into a datetime object layer

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] Date info: {self.value}. Context: {self.abstraction}"

class EntityMemoryBlock(BaseMemoryBlock):
    """Tracks specific people, pets, or named entities in the user's life."""
    def __init__(self, node_id: str, abstraction: str, value: str, contexts: list[str], entity_name: str = "Unknown", entity_type: str = "Person/Entity"):
        super().__init__(node_id, abstraction, value, contexts)
        self.block_type = "Entity/Person/Pet"
        self.entity_name = entity_name
        self.entity_type = entity_type

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] Entity {self.entity_name} ({self.entity_type}): {self.abstraction}. Details: {self.value}"

class FactMemoryBlock(BaseMemoryBlock):
    """Tracks static, objective facts about the user (e.g., profession, age, allergies)."""
    def __init__(self, node_id: str, abstraction: str, value: str, contexts: list[str], category: str = "General"):
        super().__init__(node_id, abstraction, value, contexts)
        self.block_type = "Static Fact"
        self.category = category

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] {self.category} Fact: {self.abstraction}. Details: {self.value}"

class PreferenceMemoryBlock(BaseMemoryBlock):
    """Tracks user likes, dislikes, and personal tastes."""
    def __init__(self, node_id: str, abstraction: str, value: str, contexts: list[str]):
        super().__init__(node_id, abstraction, value, contexts)
        self.block_type = "Preference/Trait"

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] User likes/dislikes: {self.value}. Context: {self.abstraction}"

class RelationshipMemoryBlock(BaseMemoryBlock):
    """Tracks connections between entities (e.g., Family, Friends, Coworkers)."""
    def __init__(self, node_id: str, abstraction: str, value: str, contexts: list[str], source_entity: str = "User", target_entity: str = "Unknown"):
        super().__init__(node_id, abstraction, value, contexts)
        self.block_type = "Relationship"
        self.source_entity = source_entity
        self.target_entity = target_entity

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] Connection between {self.source_entity} and {self.target_entity}. Relation: {self.abstraction}. Details: {self.value}"

class GoalMemoryBlock(BaseMemoryBlock):
    """Tracks user intentions, future plans, and active goals."""
    def __init__(self, node_id: str, abstraction: str, value: str, contexts: list[str], status: str = "Active"):
        super().__init__(node_id, abstraction, value, contexts)
        self.block_type = f"Goal/Intention ({status})"
        self.status = status

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] User wants to: {self.abstraction}. Plan: {self.value}"

class LocationMemoryBlock(BaseMemoryBlock):
    """Tracks spatial memory, where things are or where events take place."""
    def __init__(self, node_id: str, abstraction: str, value: str, contexts: list[str]):
        super().__init__(node_id, abstraction, value, contexts)
        self.block_type = "Spatial/Location"

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] Location info: {self.value}. Context: {self.abstraction}"

class StateChangeMemoryBlock(BaseMemoryBlock):
    """Tracks updates to previously established facts (e.g., moved to a new city, changed jobs)."""
    def __init__(self, node_id: str, abstraction: str, value: str, contexts: list[str], previous_state: str = None):
        super().__init__(node_id, abstraction, value, contexts)
        self.block_type = "State Change/Update"
        self.previous_state = previous_state

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] Update: {self.abstraction}. Was: {self.previous_state}, Now: {self.value}"


# --- HYPERGRAPH KERNEL ---

class HypergraphMemoryOSV2:
    def __init__(self):
        self.H = xgi.Hypergraph()
        self.memory_store = {}
        
        print("[Kernel v2.0] Loading local embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.node_embeddings = {}
        self.node_ids = []
        
        self.theme_to_edge_id = {}    
        self.edge_embeddings = {}     
        self.similarity_threshold = 0.65 

    def _find_similar_edge(self, proposed_context: str):
        if not self.edge_embeddings:
            return None
            
        query_embed = self.embedder.encode(proposed_context)
        themes = list(self.edge_embeddings.keys())
        embeddings_matrix = np.array(list(self.edge_embeddings.values()))
        
        similarities = cosine_similarity([query_embed], embeddings_matrix)[0]
        best_idx = np.argmax(similarities)
        
        if similarities[best_idx] >= self.similarity_threshold:
            return themes[best_idx]
            
        return None

    def ingest_block(self, block: BaseMemoryBlock):
        """Phase 1: Writing typed memory blocks to the Hypergraph"""
        self.memory_store[block.node_id] = {
            "type": block.block_type,
            "abstraction": block.abstraction,
            "value": block.value
        }
        
        self.H.add_node(block.node_id)
        resolved_contexts = [] 
        
        for proposed_context in block.contexts:
            matched_theme = self._find_similar_edge(proposed_context)
            
            if matched_theme:
                edge_id = self.theme_to_edge_id[matched_theme]
                self.H.add_node_to_edge(edge_id, block.node_id)
                resolved_contexts.append(matched_theme)
            else:
                self.H.add_edge([block.node_id])
                new_edge_id = list(self.H.edges)[-1]
                
                self.theme_to_edge_id[proposed_context] = new_edge_id
                self.edge_embeddings[proposed_context] = self.embedder.encode(proposed_context)
                resolved_contexts.append(proposed_context) 
                
        # Generate embedding using the type-aware string
        embedding = self.embedder.encode(block.get_embedding_string())
        self.node_embeddings[block.node_id] = embedding
        if block.node_id not in self.node_ids:
            self.node_ids.append(block.node_id)
            
        print(f"[Kernel v2.0] Ingested '{block.block_type}' block '{block.node_id}' into hyperedges: {resolved_contexts}")

    def _find_semantic_seeds(self, query: str, top_k: int = 2):
        """Returns the top K semantic seed nodes to handle multi-part queries."""
        if not self.node_ids:
            return []
            
        query_embed = self.embedder.encode(query)
        embeddings_matrix = np.array([self.node_embeddings[nid] for nid in self.node_ids])
        
        similarities = cosine_similarity([query_embed], embeddings_matrix)[0]
        
        # Get indices of the top_k highest similarities
        best_indices = np.argsort(similarities)[-top_k:]
        return [self.node_ids[i] for i in best_indices]

    def retrieve_envelope(self, query: str, top_k: int = 3):
        seed_nodes = self._find_semantic_seeds(query, top_k)
        if not seed_nodes:
            return "Memory is empty."
            
        activated_nodes = set()
        
        for seed_node in seed_nodes:
            active_contexts = list(self.H.nodes.memberships(seed_node))
            for context in active_contexts:
                members = self.H.edges.members(context)
                activated_nodes.update(members)
            
        working_set = []
        is_time_query = any(word in query.lower() for word in ["when", "how long", "time", "date", "year", "month"])
        
        for n_id in activated_nodes:
            data = self.memory_store[n_id]
            formatted = f"[{data['type']}] {n_id}: {data['abstraction']} (Details: {data['value']})"
            
            # If the user asks about time, dramatically prioritize Date/State Change blocks
            if is_time_query and data['type'] in ["Temporal/Date", "State Change/Update"]:
                working_set.insert(0, f"!!! CRITICAL TIMELINE INFO: {formatted}")
            else:
                working_set.append(formatted)
            
        return "\n".join(working_set)