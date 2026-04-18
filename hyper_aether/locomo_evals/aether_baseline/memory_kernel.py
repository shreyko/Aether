"""Hypergraph memory kernel used by the Aether LOCOMO baseline.

This is a port of ``hyper_aether/memory_kernel.py`` adapted for the LOCOMO
pipeline:

* The ``SentenceTransformer`` is loaded lazily (and, by default, on CPU) so
  it doesn't fight the vLLM server for the GPU.
* The kernel now exposes ``save()`` / ``load()`` / ``retrieve_envelope()`` so
  that the separate ADD and SEARCH phases (different processes) can share
  state via disk, mirroring how mem0 persists its ChromaDB between phases.
* ``retrieve_envelope`` accepts an optional ``top_k`` and returns a
  structured list of dicts (not just a formatted string) so the answer
  prompt can format memories the same way the mem0 baseline does.
"""

from __future__ import annotations

import os
import pickle
import threading
from typing import Any

import numpy as np
import xgi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .config import EMBEDDER_DEVICE, EMBEDDER_MODEL, AETHER_SIMILARITY_THRESHOLD


# Module-level singleton so multiple HypergraphMemoryOS instances on the
# same process don't each load their own 90 MB MiniLM checkpoint.
_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            _EMBEDDER = SentenceTransformer(EMBEDDER_MODEL, device=EMBEDDER_DEVICE)
        return _EMBEDDER


class HypergraphMemoryOS:
    def __init__(self, similarity_threshold: float = AETHER_SIMILARITY_THRESHOLD, verbose: bool = False):
        self.H = xgi.Hypergraph()
        self.memory_store: dict[str, dict[str, str]] = {}

        self.embedder = _get_embedder()

        self.node_embeddings: dict[str, np.ndarray] = {}
        self.node_ids: list[str] = []

        self.theme_to_edge_id: dict[str, Any] = {}
        self.edge_embeddings: dict[str, np.ndarray] = {}
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose

        # Guard all mutations; add/search phases each run their own
        # ThreadPoolExecutor and both touch these dicts.
        self._lock = threading.Lock()

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

    def ingest_memory(
        self,
        node_id: str,
        abstraction: str,
        value: str,
        contexts: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Phase 1: Writing to the Hypergraph with Edge Consolidation."""
        with self._lock:
            if node_id in self.memory_store:
                # Disambiguate duplicate IDs produced by the extractor across
                # different chunks (e.g. ``cherry_blossoms``). Without this
                # the second ingest silently overwrites the first.
                suffix = 2
                while f"{node_id}__{suffix}" in self.memory_store:
                    suffix += 1
                node_id = f"{node_id}__{suffix}"

            self.memory_store[node_id] = {
                "abstraction": abstraction,
                "value": value,
                "metadata": metadata or {},
            }

            self.H.add_node(node_id)
            resolved_contexts: list[str] = []

            for proposed_context in contexts:
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
                    self.edge_embeddings[proposed_context] = self.embedder.encode(proposed_context)
                    resolved_contexts.append(proposed_context)

            embedding = self.embedder.encode(abstraction)
            self.node_embeddings[node_id] = embedding
            if node_id not in self.node_ids:
                self.node_ids.append(node_id)

            self._log(f"[Kernel] Ingested '{node_id}' into hyperedges: {resolved_contexts}")

    def _find_semantic_seeds(self, query: str, k: int = 1) -> list[str]:
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
        num_seeds: int = 1,
    ) -> list[dict[str, Any]]:
        """Phase 2: Dual-Space Retrieval.

        Returns up to ``top_k`` memory dicts, ranked by abstraction-space
        cosine similarity to the query, drawn from the union of hyperedges
        activated by ``num_seeds`` semantic seeds.
        """
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
            embeddings_matrix = np.array([self.node_embeddings[nid] for nid in activated_list])
            sims = cosine_similarity([query_embed], embeddings_matrix)[0]
            order = np.argsort(-sims)[:top_k]

            out: list[dict[str, Any]] = []
            for idx in order:
                nid = activated_list[int(idx)]
                data = self.memory_store[nid]
                out.append(
                    {
                        "node_id": nid,
                        "abstraction": data["abstraction"],
                        "value": data["value"],
                        "metadata": data.get("metadata", {}),
                        "score": float(sims[int(idx)]),
                    }
                )
            return out

    # ---------- persistence ----------

    def _dump_state(self) -> dict[str, Any]:
        # Rebuildable state (don't pickle the SentenceTransformer).
        edges_state: dict[Any, list[str]] = {}
        for edge_id in self.H.edges:
            edges_state[edge_id] = list(self.H.edges.members(edge_id))

        return {
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

        obj = cls(similarity_threshold=state.get("similarity_threshold", AETHER_SIMILARITY_THRESHOLD))
        obj.memory_store = state["memory_store"]
        obj.node_embeddings = state["node_embeddings"]
        obj.node_ids = state["node_ids"]
        obj.theme_to_edge_id = state["theme_to_edge_id"]
        obj.edge_embeddings = state["edge_embeddings"]

        for nid in obj.node_ids:
            obj.H.add_node(nid)
        for _edge_id, members in state["edges_state"].items():
            if members:
                obj.H.add_edge(list(members))

        # Rebuild theme_to_edge_id in terms of the freshly-assigned xgi edge IDs.
        # After ``add_edge`` above, edges are indexed 0..N-1 in insertion order.
        theme_order = list(state["theme_to_edge_id"].keys())
        new_ids = list(obj.H.edges)
        if len(theme_order) == len(new_ids):
            obj.theme_to_edge_id = dict(zip(theme_order, new_ids))

        return obj
