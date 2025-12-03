import numpy as np
import os
import json
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_distances

# --- CONFIGURATION ---
GENERATION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Semantic Jump Threshold (Still a heuristic, but necessary for V2)
SEMANTIC_BOUNDARY_THRESHOLD = 0.45

# How many previous LTM episodes to retrieve
RETRIEVAL_TOP_K = 3


class SemanticCore:
    def __init__(self):
        try:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.has_api = True
        except:
            self.has_api = False
            print("⚠️ No API Key found. Running in simulation mode.")

        print("Loading Embedding Model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text)

    def complete(self, messages, json_mode=False):
        if not self.has_api:
            # Mock responses for testing without cost
            if json_mode:
                # Simulating entity extraction or summarization
                if "Identify the key Subject Entities" in messages[0]["content"]:
                    return '{"entities": ["Lentils", "Cooking", "Python"]}'
                return '{"summary": "Simulated summary.", "facts": [{"subject": "User", "predicate": "likes", "object": "coding"}]}'
            return "Simulated AI response."

        kwargs = {
            "model": GENERATION_MODEL,
            "messages": messages,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


@dataclass
class EpisodicMemory:
    """Represents a compressed, 'closed' memory file."""

    id: int
    summary: str
    vector: np.ndarray
    timestamp: float
    facts_extracted: List[dict]


@dataclass
class Turn:
    """A single raw interaction in the Short Term Memory."""

    role: str
    content: str
    vector: Optional[np.ndarray] = None


class JanusV2Engine:
    def __init__(self):
        self.brain = SemanticCore()

        # 1. Short Term Memory (The Active Buffer)
        self.stm: List[Turn] = []

        # 2. Long Term Memory (The Archive)
        self.ltm: List[EpisodicMemory] = []

        # 3. World Model (Symbolic Graph)
        # Structure: {Subject_Key: {Predicate: Object}}
        # Note: Keys are normalized to lowercase for this demo
        self.world_model: Dict[str, Dict[str, str]] = {}

        # State tracking
        self.last_user_vector: Optional[np.ndarray] = None
        self.episode_counter = 0

    def update_world_model(self, facts: List[dict]):
        """
        Updates the symbolic knowledge graph.
        """
        for f in facts:
            sub = f.get("subject", "Unknown").lower()  # Normalize keys
            pred = f.get("predicate", "related_to")
            obj = f.get("object", "")

            if sub not in self.world_model:
                self.world_model[sub] = {}

            self.world_model[sub][pred] = obj
            # In production, you would log confidence scores here
            print(f"   [Wait-State] Knowledge Updated: {sub} -> {pred} -> {obj}")

    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Uses the LLM to identify graph keys (Entities) from the user query.
        This is the 'Index Lookup' step.
        """
        system_instruction = (
            "You are a Knowledge Graph query engine. "
            "Identify the key Subject Entities in the user's text that we might have data on. "
            "Return ONLY a JSON object with key 'entities' containing a list of strings. "
            "Normalize entities to singular form."
        )

        raw_json = self.brain.complete(
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": query},
            ],
            json_mode=True,
        )

        try:
            # Sanitation in case LLM adds markdown wrappers
            clean_json = raw_json.replace("```json", "").replace("```", "")
            data = json.loads(clean_json)
            return data.get("entities", [])
        except json.JSONDecodeError:
            print("   ⚠️ Entity Extraction Failed (JSON Error)")
            return []

    def retrieve_subgraph(self, query_text: str) -> str:
        """
        Scalable Retrieval: Only fetches facts related to entities in the query.
        Complexity: O(k) where k is entities found, rather than O(N) total facts.
        """
        search_terms = self._extract_query_entities(query_text)
        if not search_terms:
            return ""

        print(f"   🔍 Semantic Search Terms: {search_terms}")

        found_facts = []

        # In a real system, this is a Vector Search on Keys.
        # Here we do fuzzy string matching against our dictionary keys.
        for term in search_terms:
            term_key = term.lower()

            # 1. Direct Lookup
            if term_key in self.world_model:
                preds = self.world_model[term_key]
                for p, o in preds.items():
                    found_facts.append(f"{term} {p} {o}")

            # 2. Simple Partial Match Fallback (e.g. 'lentil' finding 'red lentils')
            # (Warning: This is O(N) on keys, but keys < facts. Vector search fixes this.)
            else:
                for existing_key in self.world_model.keys():
                    if term_key in existing_key:
                        preds = self.world_model[existing_key]
                        for p, o in preds.items():
                            found_facts.append(f"{existing_key} {p} {o}")

        if not found_facts:
            return ""

        return "--- RELEVANT KNOWLEDGE GRAPH ---\n" + "\n".join(found_facts)

    def retrieve_context(self, query_vec: np.ndarray, query_text: str) -> str:
        """
        Retrieves relevant LTM summaries (Vector Search) and World Model facts (Graph Search).
        """
        context_parts = []

        # A. Symbolic Context (The "Hard" Facts) - Subgraph Only
        graph_context = self.retrieve_subgraph(query_text)
        if graph_context:
            context_parts.append(graph_context)

        # B. Episodic Context (The "Soft" Narrative) - Vector Search
        if self.ltm:
            ltm_vectors = np.array([m.vector for m in self.ltm])
            # cosine distance: 0 is identical, 1 is opposite
            dists = cosine_distances([query_vec], ltm_vectors)[0]

            # Get indices of top k closest
            top_indices = np.argsort(dists)[:RETRIEVAL_TOP_K]

            found_episodes = []
            for idx in top_indices:
                memory = self.ltm[idx]
                # Filter noise
                if dists[idx] < 0.6:
                    found_episodes.append(f"Episode {memory.id}: {memory.summary}")

            if found_episodes:
                context_parts.append("\n--- RELEVANT PAST EPISODES ---")
                context_parts.extend(found_episodes)

        return "\n".join(context_parts)

    def consolidate_memory(self):
        """
        The 'Sleep' Phase: Compresses STM into an LTM Episode.
        """
        if not self.stm:
            return

        text_block = "\n".join([f"{t.role}: {t.content}" for t in self.stm])

        system_instruction = (
            "Analyze this conversation segment. Return JSON with:\n"
            "1. 'summary': concise narrative summary.\n"
            "2. 'facts': list of objects with keys: 'subject', 'predicate', 'object'. "
            "Focus on permanent facts (names, preferences, tech stacks)."
        )

        print(f"\n   ⏳ Consolidating Episode {self.episode_counter}...")
        raw_json = self.brain.complete(
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": text_block},
            ],
            json_mode=True,
        )

        try:
            data = json.loads(raw_json)
            summary = data.get("summary", "No summary")
            facts = data.get("facts", [])

            # Calculate centroid vector
            user_vectors = [
                t.vector for t in self.stm if t.role == "user" and t.vector is not None
            ]
            if user_vectors:
                centroid = np.mean(user_vectors, axis=0)
            else:
                centroid = self.brain.embed(summary)

            episode = EpisodicMemory(
                id=self.episode_counter,
                summary=summary,
                vector=centroid,
                timestamp=time.time(),
                facts_extracted=facts,
            )
            self.ltm.append(episode)
            self.episode_counter += 1

            self.update_world_model(facts)
            print(f"   ✅ Archived: '{summary[:50]}...'")

        except json.JSONDecodeError:
            print("   ❌ Consolidation Failed: Invalid JSON")

        self.stm = []

    def chat(self, user_input: str):
        # 1. Embed Input
        u_vec = self.brain.embed(user_input)

        # 2. Check Event Boundary
        if self.last_user_vector is not None:
            dist = cosine_distances([u_vec], [self.last_user_vector])[0][0]
            print(f"   [Metric] Semantic Vel: {dist:.3f}")

            if dist > SEMANTIC_BOUNDARY_THRESHOLD:
                print("   ⚡ Topic Shift Detected. Triggering Consolidation.")
                self.consolidate_memory()

        self.last_user_vector = u_vec

        # 3. Retrieve Context (Passing both Vector and Text now)
        context_block = self.retrieve_context(u_vec, user_input)

        # 4. Build STM Context
        short_term_context = "\n".join([f"{t.role}: {t.content}" for t in self.stm])

        # 5. Generate Response
        system_prompt = (
            f"You are Janus. Use the context to answer.\n\n"
            f"{context_block}\n\n"
            f"--- CURRENT CONVERSATION ---\n"
            f"{short_term_context}"
        )

        response_text = self.brain.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        )

        self.stm.append(Turn(role="user", content=user_input, vector=u_vec))
        self.stm.append(Turn(role="janus", content=response_text))

        print(f"Janus: {response_text}\n")


# --- EXECUTION ---
if __name__ == "__main__":
    engine = JanusV2Engine()

    conversation_script = [
        # Topic 1
        "I'm planning a dinner party. My wife is vegetarian.",
        "What are some good high-protein vegetarian dishes?",
        "How do I cook lentils correctly?",
        # Topic 2 (Shift)
        "Okay enough food. I have a Python bug.",
        "It's an IndexError in my list comprehension.",
        # Topic 3 (Callback)
        "Wait, back to the dinner. What did you say about lentils?",
        # Topic 4 (Fact Check)
        "And does that work for my wife's diet?",
    ]

    print("--- Starting Janus V2 (Graph-RAG Hybrid) ---")
    for line in conversation_script:
        print(f"User: {line}")
        engine.chat(line)
        time.sleep(1)
