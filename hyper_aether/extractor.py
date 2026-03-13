import ollama
from pydantic import BaseModel

class MemoryEntry(BaseModel):
    node_id: str
    abstraction: str
    value: str
    contexts: list[str]

class ExtractionResult(BaseModel):
    memories: list[MemoryEntry]

def extract_hypergraph_nodes(transcript_chunk: str, model_name="llama3.2") -> list[MemoryEntry]:
    """
    Uses a local LLM via Ollama to extract nodes and Hypergraph contexts.
    """
    prompt = f"""
    Analyze the following conversation chunk. Extract factual memories that could be useful for a persistent AI assistant.
    For each memory, provide:
    1. node_id: A unique snake_case identifier
    2. abstraction: A high-level summary (Primary Abstraction)
    3. value: The raw, specific detail mentioned (Memory Value)
    4. contexts: A list of 1-3 situational envelopes or themes this belongs to. Use consistent theme names.
    
    Conversation Chunk:
    "{transcript_chunk}"
    """
    
    try:
        #Use Ollama's structured output capability to guarantee a strict JSON schema
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            format=ExtractionResult.model_json_schema(),
            options={'temperature': 0.0} # Deterministic output for data extraction
        )
        
        # Validate and parse the JSON response
        result = ExtractionResult.model_validate_json(response.message.content)
        return result.memories
    except Exception as e:
        print(f"[Extractor Error] {e}")
        return []