"""Hypergraph memory kernel (V2) for the Aether LOCOMO baseline.

This is a port of ``hyper_aether/mem_kernel2_0.py`` (V2) adapted for the
LOCOMO pipeline. Compared to V1, V2 introduces:

* **Typed memory blocks** -- each memory is one of 8 subclasses
  (Date, Entity, Fact, Preference, Relationship, Goal, Location,
  StateChange) plus a Generic base. Each subclass emits a type-biased
  embedding string (``get_embedding_string``) so the retriever can
  better discriminate by kind.
* **Multi-seed retrieval** -- the dual-graph jump now starts from the
  top-N semantically similar nodes (default 2) instead of only the
  single nearest node.
* **Time-query boost** -- when the question contains time keywords
  (``when``, ``how long``, ``time``, ``date``, ``year``, ``month``),
  ``Temporal/Date`` and ``State Change/Update`` memories are surfaced
  to the front of the returned envelope.

The production hardening from the V1 baseline port is preserved:

* the ``SentenceTransformer`` is loaded lazily via a module-level
  singleton (CPU by default so it does not fight vLLM for the GPU);
* all mutations are guarded by a ``threading.Lock`` so the ADD/SEARCH
  thread pools can safely share a kernel instance;
* ``save()`` / ``load()`` let the ADD and SEARCH phases (separate
  processes) share state via pickled kernels in ``aether_db/``;
* ``retrieve_envelope`` returns a structured ``list[dict]`` (not a
  formatted string) so the answer prompt can format memories the same
  way the mem0 baseline does, and it accepts an ``top_k`` cap on the
  number of returned memories (default 30 -- matches the old behavior).
"""

from __future__ import annotations

import os
import pickle
import re
import threading
from typing import Any

import numpy as np
import xgi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .config import AETHER_SIMILARITY_THRESHOLD, EMBEDDER_DEVICE, EMBEDDER_MODEL


# ---------------------------------------------------------------------------
# Typed memory blocks (V2)
# ---------------------------------------------------------------------------


class BaseMemoryBlock:
    block_type: str = "Generic"

    def __init__(
        self,
        node_id: str,
        abstraction: str,
        value: str,
        contexts: list[str],
    ) -> None:
        self.node_id = node_id
        self.abstraction = abstraction
        self.value = value
        self.contexts = list(contexts)

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] {self.abstraction}"

    def extra_fields(self) -> dict[str, Any]:
        """Per-subclass extra fields to persist alongside the block."""
        return {}


class DateMemoryBlock(BaseMemoryBlock):
    block_type = "Temporal/Date"

    def __init__(
        self,
        node_id: str,
        abstraction: str,
        value: str,
        contexts: list[str],
        raw_date: str | None = None,
    ) -> None:
        super().__init__(node_id, abstraction, value, contexts)
        self.raw_date = raw_date

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] Date info: {self.value}. Context: {self.abstraction}"

    def extra_fields(self) -> dict[str, Any]:
        return {"raw_date": self.raw_date}


class EntityMemoryBlock(BaseMemoryBlock):
    block_type = "Entity/Person/Pet"

    def __init__(
        self,
        node_id: str,
        abstraction: str,
        value: str,
        contexts: list[str],
        entity_name: str = "Unknown",
        entity_type: str = "Person/Entity",
    ) -> None:
        super().__init__(node_id, abstraction, value, contexts)
        self.entity_name = entity_name or "Unknown"
        self.entity_type = entity_type or "Person/Entity"

    def get_embedding_string(self) -> str:
        return (
            f"[{self.block_type}] Entity {self.entity_name} ({self.entity_type}): "
            f"{self.abstraction}. Details: {self.value}"
        )

    def extra_fields(self) -> dict[str, Any]:
        return {"entity_name": self.entity_name, "entity_type": self.entity_type}


class FactMemoryBlock(BaseMemoryBlock):
    block_type = "Static Fact"

    def __init__(
        self,
        node_id: str,
        abstraction: str,
        value: str,
        contexts: list[str],
        category: str = "General",
    ) -> None:
        super().__init__(node_id, abstraction, value, contexts)
        self.category = category or "General"

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] {self.category} Fact: {self.abstraction}. Details: {self.value}"

    def extra_fields(self) -> dict[str, Any]:
        return {"category": self.category}


class PreferenceMemoryBlock(BaseMemoryBlock):
    block_type = "Preference/Trait"

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] User likes/dislikes: {self.value}. Context: {self.abstraction}"


class RelationshipMemoryBlock(BaseMemoryBlock):
    block_type = "Relationship"

    def __init__(
        self,
        node_id: str,
        abstraction: str,
        value: str,
        contexts: list[str],
        source_entity: str = "User",
        target_entity: str = "Unknown",
    ) -> None:
        super().__init__(node_id, abstraction, value, contexts)
        self.source_entity = source_entity or "User"
        self.target_entity = target_entity or "Unknown"

    def get_embedding_string(self) -> str:
        return (
            f"[{self.block_type}] Connection between {self.source_entity} and "
            f"{self.target_entity}. Relation: {self.abstraction}. Details: {self.value}"
        )

    def extra_fields(self) -> dict[str, Any]:
        return {"source_entity": self.source_entity, "target_entity": self.target_entity}


class GoalMemoryBlock(BaseMemoryBlock):
    block_type = "Goal/Intention"

    def __init__(
        self,
        node_id: str,
        abstraction: str,
        value: str,
        contexts: list[str],
        status: str = "Active",
    ) -> None:
        super().__init__(node_id, abstraction, value, contexts)
        self.status = status or "Active"
        # Mirror the V2 reference: the displayed block_type includes status.
        self.block_type = f"Goal/Intention ({self.status})"

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] User wants to: {self.abstraction}. Plan: {self.value}"

    def extra_fields(self) -> dict[str, Any]:
        return {"status": self.status}


class LocationMemoryBlock(BaseMemoryBlock):
    block_type = "Spatial/Location"

    def get_embedding_string(self) -> str:
        return f"[{self.block_type}] Location info: {self.value}. Context: {self.abstraction}"


class StateChangeMemoryBlock(BaseMemoryBlock):
    block_type = "State Change/Update"

    def __init__(
        self,
        node_id: str,
        abstraction: str,
        value: str,
        contexts: list[str],
        previous_state: str | None = None,
    ) -> None:
        super().__init__(node_id, abstraction, value, contexts)
        self.previous_state = previous_state

    def get_embedding_string(self) -> str:
        return (
            f"[{self.block_type}] Update: {self.abstraction}. "
            f"Was: {self.previous_state}, Now: {self.value}"
        )

    def extra_fields(self) -> dict[str, Any]:
        return {"previous_state": self.previous_state}


# Canonical block_type -> class mapping. Used by downstream factories to
# rehydrate the right subclass from an extractor payload. Keys are
# case-insensitive-matched by ``block_class_for``.
BLOCK_CLASSES: dict[str, type[BaseMemoryBlock]] = {
    "Temporal/Date": DateMemoryBlock,
    "Entity/Person/Pet": EntityMemoryBlock,
    "Static Fact": FactMemoryBlock,
    "Preference/Trait": PreferenceMemoryBlock,
    "Relationship": RelationshipMemoryBlock,
    "Goal/Intention": GoalMemoryBlock,
    "Spatial/Location": LocationMemoryBlock,
    "State Change/Update": StateChangeMemoryBlock,
}


def block_class_for(block_type: str | None) -> type[BaseMemoryBlock]:
    """Resolve a (possibly noisy) extractor ``block_type`` to a block class.

    Falls back to ``BaseMemoryBlock`` for unknown values so the ADD phase
    never crashes on an unexpected label from the LLM.
    """
    if not block_type:
        return BaseMemoryBlock
    key = block_type.strip()
    if key in BLOCK_CLASSES:
        return BLOCK_CLASSES[key]
    # Case-insensitive match.
    lower = key.lower()
    for canonical, cls in BLOCK_CLASSES.items():
        if canonical.lower() == lower:
            return cls
    return BaseMemoryBlock


# ---------------------------------------------------------------------------
# Embedder singleton
# ---------------------------------------------------------------------------


_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            _EMBEDDER = SentenceTransformer(EMBEDDER_MODEL, device=EMBEDDER_DEVICE)
        return _EMBEDDER


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


_TIME_QUERY_RE = re.compile(
    r"\b(when|how long|time|date|year|month|day|week|ago|yesterday|tomorrow)\b",
    re.IGNORECASE,
)
_TIME_BOOST_TYPES = {"Temporal/Date", "State Change/Update"}


def _is_time_query(query: str) -> bool:
    return bool(_TIME_QUERY_RE.search(query or ""))


# Default number of seed nodes for multi-seed retrieval (V2). Overridable via env.
_DEFAULT_NUM_SEEDS = int(os.getenv("AETHER_NUM_SEEDS", "2"))


class HypergraphMemoryOS:
    """V2 hypergraph memory kernel used by the Aether LOCOMO baseline.

    The public surface (``ingest_block``, ``retrieve_envelope``, ``save``,
    ``load``) is what ``aether_add.py`` and ``aether_search.py`` call. A
    thin ``ingest_memory`` shim is kept for callers that still operate in
    V1 terms (plain ``abstraction``/``value``/``contexts``).
    """

    def __init__(
        self,
        similarity_threshold: float = AETHER_SIMILARITY_THRESHOLD,
        verbose: bool = False,
    ) -> None:
        self.H = xgi.Hypergraph()
        self.memory_store: dict[str, dict[str, Any]] = {}

        self.embedder = _get_embedder()

        self.node_embeddings: dict[str, np.ndarray] = {}
        self.node_ids: list[str] = []

        self.theme_to_edge_id: dict[str, Any] = {}
        self.edge_embeddings: dict[str, np.ndarray] = {}
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose

        self._lock = threading.Lock()

    # -------------------------- ingestion -------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _find_similar_edge(self, proposed_context: str) -> str | None:
        if not self.edge_embeddings:
            return None

        query_embed = self.embedder.encode(proposed_context)
        themes = list(self.edge_embeddings.keys())
        embeddings_matrix = np.array(list(self.edge_embeddings.values()))

        similarities = cosine_similarity([query_embed], embeddings_matrix)[0]
        best_idx = int(np.argmax(similarities))

        if similarities[best_idx] >= self.similarity_threshold:
            return themes[best_idx]
        return None

    def _ensure_unique_node_id(self, node_id: str) -> str:
        """Disambiguate duplicate node ids produced across chunks.

        Without this the second ingest silently overwrites the first (the
        V1 port already had this behavior; we keep it under V2).
        """
        if node_id not in self.memory_store:
            return node_id
        suffix = 2
        while f"{node_id}__{suffix}" in self.memory_store:
            suffix += 1
        return f"{node_id}__{suffix}"

    def ingest_block(
        self,
        block: BaseMemoryBlock,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Phase 1 (V2): write a typed memory block to the hypergraph.

        Returns the resolved (possibly disambiguated) ``node_id`` the
        block was stored under.
        """
        with self._lock:
            node_id = self._ensure_unique_node_id(block.node_id)
            block.node_id = node_id

            self.memory_store[node_id] = {
                "type": block.block_type,
                "abstraction": block.abstraction,
                "value": block.value,
                "extra": block.extra_fields(),
                "metadata": metadata or {},
            }

            self.H.add_node(node_id)
            resolved_contexts: list[str] = []

            for proposed_context in block.contexts:
                matched_theme = self._find_similar_edge(proposed_context)
                if matched_theme:
                    edge_id = self.theme_to_edge_id[matched_theme]
                    try:
                        self.H.add_node_to_edge(edge_id, node_id)
                    except Exception:
                        # xgi raises if the node is already in the edge; safe to ignore.
                        pass
                    resolved_contexts.append(matched_theme)
                    self._log(f"    [Merge] '{proposed_context}' -> '{matched_theme}'")
                else:
                    self.H.add_edge([node_id])
                    new_edge_id = list(self.H.edges)[-1]
                    self.theme_to_edge_id[proposed_context] = new_edge_id
                    self.edge_embeddings[proposed_context] = self.embedder.encode(
                        proposed_context
                    )
                    resolved_contexts.append(proposed_context)

            # V2: embed the *type-biased* string so retrieval can
            # discriminate between kinds (e.g. a date vs a preference).
            embedding = self.embedder.encode(block.get_embedding_string())
            self.node_embeddings[node_id] = embedding
            if node_id not in self.node_ids:
                self.node_ids.append(node_id)

            self._log(
                f"[Kernel v2] Ingested '{block.block_type}' block '{node_id}' "
                f"into hyperedges: {resolved_contexts}"
            )
            return node_id

    def ingest_memory(
        self,
        node_id: str,
        abstraction: str,
        value: str,
        contexts: list[str],
        metadata: dict[str, Any] | None = None,
        block_type: str | None = None,
        **extra: Any,
    ) -> str:
        """Back-compat shim that wraps a plain memory into a typed block.

        If ``block_type`` is given we build the matching subclass; otherwise
        we default to a generic ``BaseMemoryBlock`` (same behavior as V1).
        ``extra`` is forwarded as subclass-specific kwargs.
        """
        cls = block_class_for(block_type)
        try:
            block = cls(node_id=node_id, abstraction=abstraction, value=value, contexts=contexts, **extra)
        except TypeError:
            # Subclass didn't accept one of the extras; fall back to generic.
            block = BaseMemoryBlock(
                node_id=node_id, abstraction=abstraction, value=value, contexts=contexts
            )
        return self.ingest_block(block, metadata=metadata)

    # -------------------------- retrieval -------------------------------

    def _find_semantic_seeds(self, query: str, k: int = 2) -> list[str]:
        if not self.node_ids:
            return []
        query_embed = self.embedder.encode(query)
        embeddings_matrix = np.array([self.node_embeddings[nid] for nid in self.node_ids])
        similarities = cosine_similarity([query_embed], embeddings_matrix)[0]
        k = max(1, min(k, len(self.node_ids)))
        top_idx = np.argsort(-similarities)[:k]
        return [self.node_ids[i] for i in top_idx]

    def retrieve_envelope(
        self,
        query: str,
        top_k: int = 30,
        num_seeds: int | None = None,
    ) -> list[dict[str, Any]]:
        """Phase 2 (V2): dual-space retrieval with multi-seed + time boost.

        Args:
            query: The user question.
            top_k: Max number of memory dicts to return.
            num_seeds: Number of semantic seed nodes. Defaults to
                ``AETHER_NUM_SEEDS`` (env, default 2). V1 used 1.

        Returns a list of dicts, ranked primarily by cosine similarity to
        the query (using each node's type-biased embedding). If the query
        looks time-related, Temporal/Date and State Change/Update blocks
        are moved to the front of the list (keeping their relative order).
        """
        if num_seeds is None:
            num_seeds = _DEFAULT_NUM_SEEDS

        with self._lock:
            seeds = self._find_semantic_seeds(query, k=num_seeds)
            if not seeds:
                return []

            activated: set[str] = set()
            for seed in seeds:
                active_edges = list(self.H.nodes.memberships(seed))
                for edge_id in active_edges:
                    members = self.H.edges.members(edge_id)
                    activated.update(members)

            if not activated:
                return []

            activated_list = list(activated)
            query_embed = self.embedder.encode(query)
            embeddings_matrix = np.array(
                [self.node_embeddings[nid] for nid in activated_list]
            )
            sims = cosine_similarity([query_embed], embeddings_matrix)[0]
            order = np.argsort(-sims)[:top_k]

            out: list[dict[str, Any]] = []
            for idx in order:
                nid = activated_list[int(idx)]
                data = self.memory_store[nid]
                out.append(
                    {
                        "node_id": nid,
                        "type": data.get("type", "Generic"),
                        "abstraction": data["abstraction"],
                        "value": data["value"],
                        "extra": data.get("extra", {}),
                        "metadata": data.get("metadata", {}),
                        "score": float(sims[int(idx)]),
                    }
                )

            if _is_time_query(query):
                # Stable partition: time-relevant types first, rest after.
                time_hits = [m for m in out if m["type"] in _TIME_BOOST_TYPES]
                others = [m for m in out if m["type"] not in _TIME_BOOST_TYPES]
                out = time_hits + others

            return out

    # -------------------------- persistence -----------------------------

    def _dump_state(self) -> dict[str, Any]:
        # Rebuildable state (don't pickle the SentenceTransformer).
        edges_state: dict[Any, list[str]] = {}
        for edge_id in self.H.edges:
            edges_state[edge_id] = list(self.H.edges.members(edge_id))

        return {
            "version": 2,
            "memory_store": self.memory_store,
            "node_embeddings": self.node_embeddings,
            "node_ids": self.node_ids,
            "theme_to_edge_id": self.theme_to_edge_id,
            "edge_embeddings": self.edge_embeddings,
            "edges_state": edges_state,
            "similarity_threshold": self.similarity_threshold,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._dump_state(), f)

    @classmethod
    def load(cls, path: str) -> "HypergraphMemoryOS":
        with open(path, "rb") as f:
            state = pickle.load(f)

        obj = cls(
            similarity_threshold=state.get(
                "similarity_threshold", AETHER_SIMILARITY_THRESHOLD
            )
        )

        # V1 pickles stored {abstraction, value, metadata}; V2 adds 'type'
        # and 'extra'. Normalize on load so retrieve_envelope is uniform.
        raw_store = state.get("memory_store", {})
        normalized: dict[str, dict[str, Any]] = {}
        for nid, data in raw_store.items():
            normalized[nid] = {
                "type": data.get("type", "Generic"),
                "abstraction": data.get("abstraction", ""),
                "value": data.get("value", ""),
                "extra": data.get("extra", {}),
                "metadata": data.get("metadata", {}),
            }
        obj.memory_store = normalized
        obj.node_embeddings = state["node_embeddings"]
        obj.node_ids = state["node_ids"]
        obj.theme_to_edge_id = state["theme_to_edge_id"]
        obj.edge_embeddings = state["edge_embeddings"]

        for nid in obj.node_ids:
            obj.H.add_node(nid)
        for _edge_id, members in state["edges_state"].items():
            if members:
                obj.H.add_edge(list(members))

        # Rebuild theme_to_edge_id in terms of the freshly-assigned xgi edge
        # IDs. After add_edge above, edges are indexed 0..N-1 in insertion
        # order, matching the original ingestion order.
        theme_order = list(state["theme_to_edge_id"].keys())
        new_ids = list(obj.H.edges)
        if len(theme_order) == len(new_ids):
            obj.theme_to_edge_id = dict(zip(theme_order, new_ids))

        return obj
