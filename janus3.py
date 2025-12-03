import numpy as np
import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_distances

# --- CONFIGURATION ---
GENERATION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# A semantic jump > 0.45 usually implies a topic change in this embedding space
# Lower = More sensitive (more fragments), Higher = coarser memory
SEMANTIC_BOUNDARY_THRESHOLD = 0.45

# How many previous LTM episodes to retrieve into context
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
            # Mock response for testing without cost
            if json_mode:
                return '{"summary": "Simulated summary of conversation.", "facts": [{"subject": "User", "predicate": "interest", "object": "testing"}]}'
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

    role: str  # 'user' or 'janus'
    content: str
    vector: Optional[np.ndarray] = None


class JanusV2Engine:
    def __init__(self):
        self.brain = SemanticCore()

        # 1. Short Term Memory (The Active Buffer)
        self.stm: List[Turn] = []

        # 2. Long Term Memory (The Archive)
        self.ltm: List[EpisodicMemory] = []

        # 3. World Model (The Symbolic Graph - simplified to Dict for this demo)
        # Structure: {Subject: {Predicate: Object}}
        self.world_model: Dict[str, Dict[str, str]] = {}

        # State tracking
        self.last_user_vector: Optional[np.ndarray] = None
        self.episode_counter = 0

    def update_world_model(self, facts: List[dict]):
        """
        Updates the symbolic knowledge graph.
        Overwrite logic handles 'updating' beliefs rather than just appending strings.
        """
        for f in facts:
            sub = f.get("subject", "Unknown")
            pred = f.get("predicate", "related_to")
            obj = f.get("object", "")

            if sub not in self.world_model:
                self.world_model[sub] = {}

            # This logic overwrites previous facts (e.g. Current City: Paris -> London)
            # A real graph DB would handle history/edges better.
            self.world_model[sub][pred] = obj
            print(f"   [Wait-State] Knowledge Updated: {sub} -> {pred} -> {obj}")

    def consolidate_memory(self):
        """
        The 'Sleep' Phase: Compresses STM into an LTM Episode.
        """
        if not self.stm:
            return

        text_block = "\n".join([f"{t.role}: {t.content}" for t in self.stm])

        # 1. Neuro-Symbolic Extraction Prompt
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

            # Calculate centroid vector of the user inputs in this episode
            user_vectors = [
                t.vector for t in self.stm if t.role == "user" and t.vector is not None
            ]
            if user_vectors:
                centroid = np.mean(user_vectors, axis=0)
            else:
                centroid = self.brain.embed(summary)

            # Store in LTM
            episode = EpisodicMemory(
                id=self.episode_counter,
                summary=summary,
                vector=centroid,
                timestamp=time.time(),
                facts_extracted=facts,
            )
            self.ltm.append(episode)
            self.episode_counter += 1

            # Update Symbolic State
            self.update_world_model(facts)

            print(f"   ✅ Archived: '{summary[:50]}...'")

        except json.JSONDecodeError:
            print("   ❌ Consolidation Failed: Invalid JSON")

        # Clear STM (Or keep the very last turn for continuity - here we clear)
        self.stm = []

    def retrieve_context(self, query_vec: np.ndarray) -> str:
        """
        Retrieves relevant LTM summaries and World Model facts.
        """
        context_parts = []

        # A. Symbolic Context (The "Hard" Facts)
        # In a real app, you'd fuzzy match entities in the query to keys in self.world_model
        if self.world_model:
            context_parts.append("--- KNOWN FACTS ---")
            for sub, preds in self.world_model.items():
                for pred, obj in preds.items():
                    context_parts.append(f"{sub} {pred} {obj}")

        # B. Episodic Context (The "Soft" Narrative)
        if self.ltm:
            # Cosine similarity search
            ltm_vectors = np.array([m.vector for m in self.ltm])
            # cosine distance is 1 - similarity. We want smallest distance.
            dists = cosine_distances([query_vec], ltm_vectors)[0]

            # Get indices of top k closest
            top_indices = np.argsort(dists)[:RETRIEVAL_TOP_K]

            context_parts.append("\n--- RELEVANT PAST EPISODES ---")
            for idx in top_indices:
                memory = self.ltm[idx]
                # Filter out irrelevant stuff (distance > 0.6 means not very relevant)
                if dists[idx] < 0.6:
                    context_parts.append(f"Episode {memory.id}: {memory.summary}")

        return "\n".join(context_parts)

    def chat(self, user_input: str):
        # 1. Embed Input
        u_vec = self.brain.embed(user_input)

        # 2. Check Event Boundary (The "Clustering" Replacement)
        # We only check distance against the LAST USER INPUT, not the Agent's output
        if self.last_user_vector is not None:
            dist = cosine_distances([u_vec], [self.last_user_vector])[0][0]
            print(f"   [Metric] Semantic Vel: {dist:.3f}")

            if dist > SEMANTIC_BOUNDARY_THRESHOLD:
                print("   ⚡ Topic Shift Detected. Triggering Consolidation.")
                # Save the CURRENT buffer (which represents the PREVIOUS topic)
                self.consolidate_memory()

        self.last_user_vector = u_vec

        # 3. Retrieve Context (LTM + Facts)
        long_term_context = self.retrieve_context(u_vec)

        # 4. Build STM Context (Recent conversation)
        short_term_context = "\n".join([f"{t.role}: {t.content}" for t in self.stm])

        # 5. Generate Response
        system_prompt = (
            f"You are Janus. Use the context to answer.\n\n"
            f"{long_term_context}\n\n"
            f"--- CURRENT CONVERSATION ---\n"
            f"{short_term_context}"
        )

        response_text = self.brain.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        )

        # 6. Update STM (Active Buffer)
        # We append the NEW input and the NEW response to the STM
        self.stm.append(Turn(role="user", content=user_input, vector=u_vec))
        self.stm.append(Turn(role="janus", content=response_text))

        print(f"Janus: {response_text}\n")


# --- EXECUTION ---
if __name__ == "__main__":
    engine = JanusV2Engine()

    # Simulating a flow that forces segmentation
    conversation_script = [
        # Topic 1: Cooking
        "I'm planning a dinner party. My wife is vegetarian.",
        "What are some good high-protein vegetarian dishes?",
        "How do I cook lentils correctly?",
        # Topic 2: Coding (High Semantic Velocity -> Should trigger Save of Topic 1)
        "Okay enough food. I have a Python bug.",
        "It's an IndexError in my list comprehension.",
        # Topic 3: Callback (Testing Retrieval)
        "Wait, back to the dinner. What did you say about lentils?",
        # Topic 4: Callback to Fact (Testing Symbol Grounding)
        "And does that work for my wife's diet?",
    ]

    print("--- Starting Janus V2 (Event Boundary Architecture) ---")
    for line in conversation_script:
        print(f"User: {line}")
        engine.chat(line)
        time.sleep(1)  # simulate think time
