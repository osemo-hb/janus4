"""
LLM Prompts for Janus Core.

Centralized prompt definitions for consistency and easy modification.
"""

# Unified extraction (entities, facts, summary)
EXTRACTION_SYSTEM_PROMPT = """You are an information extraction assistant. Analyze the conversation and extract:

1. ENTITIES: Named entities with their types
   Types: person, organization, location, concept, product, event

2. FACTS: Subject-predicate-object triples representing key information
   Use CANONICAL predicates from this list when possible:
   - worksFor, hasRole (employment)
   - residesIn, locatedIn, birthPlace (location)
   - knows, colleagueOf, spouse, parentOf, childOf (relationships)
   - hasSkill, uses (abilities)
   - hasPreference, hasNegativePreference (likes/dislikes)
   - owns, alumniOf, hasCredential (ownership/education)
   - intends, hasInterest (plans)
   - memberOf, partOf (membership)

3. COMPRESSED STATE: A 20-40 word abstraction capturing the key state of this conversation segment.
   This should be a highly compressed summary that captures the essential information.

4. TOPIC LABEL: A 2-3 word label for the topic being discussed.

5. SUMMARY: A concise 1-2 sentence summary.

Return JSON in this exact format:
{
    "entities": [
        {"name": "John Smith", "type": "person"},
        {"name": "Acme Corp", "type": "organization"}
    ],
    "facts": [
        {
            "subject": "John Smith",
            "predicate": "worksFor",
            "object": "Acme Corp",
            "source_span": "I work at Acme Corp"
        }
    ],
    "episode_state": {
        "compressed": "John discussed his role at Acme Corp as a senior developer. He prefers Python and works remotely from Seattle.",
        "participants": ["John Smith", "Acme Corp"],
        "topic_label": "work discussion"
    },
    "summary": "User discussed their work at Acme Corp and development preferences."
}

Guidelines:
- Use canonical predicate names from the list above
- Extract source_span: the original text that supports each fact
- Compressed state should be 20-40 words capturing key information
- participants should list canonical entity names involved
- topic_label should be 2-3 words describing the conversation topic"""


# Topic boundary detection
TOPIC_DETECTION_PROMPT = """Analyze if the new message starts a NEW TOPIC or continues the current conversation topic.

Recent conversation:
{context}

New message: "{new_message}"

A NEW TOPIC is indicated when:
- The subject matter changes significantly
- A completely new question or request is introduced
- The conversation shifts to an unrelated area
- There's a clear break in the flow of discussion

CONTINUATION is indicated when:
- The message follows up on the current discussion
- It asks for clarification about recent topics
- It provides additional information about what was discussed
- It responds to or builds upon recent messages

Return JSON:
{{
    "is_new_topic": true/false,
    "confidence": 0.0-1.0,
    "topic_label": "2-3 word label for the current/new topic",
    "reasoning": "Brief explanation"
}}"""


# Memory reranking
RERANK_PROMPT = """Rate each memory's relevance to the query on a scale of 0-10.

Query: "{query}"

Memories to evaluate:
{memories}

For each memory, assess:
- How directly relevant is it to answering the query?
- Does it provide useful context or information?
- Is it recent and applicable?

Return a JSON array with scores for each memory:
[
    {{"index": 0, "score": 8, "reason": "Directly addresses the question about..."}},
    {{"index": 1, "score": 3, "reason": "Tangentially related but not specific..."}}
]

Be strict: only give high scores (7+) to highly relevant memories.
Give low scores (0-3) to irrelevant or outdated information."""


# Episode distillation
EPISODE_DISTILLATION_PROMPT = """Synthesize these related conversation episodes into a single coherent summary.

Episodes (chronologically ordered):
{episodes}

Requirements:
- Preserve key facts, relationships, and temporal order
- Maximum {max_words} words
- Maintain important details while removing redundancy
- Focus on information useful for future conversations

Return JSON:
{{
    "synthesis": "Your synthesized summary here...",
    "topic_cluster": "2-3 word topic label",
    "key_entities": ["Entity1", "Entity2"]
}}"""


# Fact distillation
FACT_DISTILLATION_PROMPT = """Synthesize these facts about {entity_name} into a brief narrative profile.

Facts:
{facts}

Requirements:
- Create a natural language profile
- Maximum {max_words} words
- Highlight the most important attributes
- Make it useful for understanding who/what {entity_name} is

Return JSON:
{{
    "profile": "Your profile summary here...",
    "key_attributes": ["attr1", "attr2", "attr3"]
}}"""
