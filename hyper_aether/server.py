from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import pickle

# Import your existing logic
from mem_kernel2_0 import HypergraphMemoryOSV2
from gemini_implementation import GeminiAPI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPH_FILE_PATH = os.path.join(os.path.dirname(__file__), "saved_graphs", "chrome_memory.pkl")
gemini_api = GeminiAPI(os.getenv("GEMINI_API_KEY"))

# Ensure directory exists
os.makedirs(os.path.dirname(GRAPH_FILE_PATH), exist_ok=True)

# Load existing graph from disk if it exists, otherwise create a new one
if os.path.exists(GRAPH_FILE_PATH):
    print("Loading existing memory graph from disk...")
    with open(GRAPH_FILE_PATH, 'rb') as f:
        memos = pickle.load(f)
else:
    print("No existing graph found. Starting fresh...")
    memos = HypergraphMemoryOSV2()

@app.post("/ingest")
async def ingest_memory(request: Request):
    data = await request.json()
    text = data.get("text", "")
    
    # Process and save to hypergraph
    response = gemini_api.process_text(text)
    
    if "blocks" in response and response["blocks"]:
        for block in response['blocks']:
            memos.ingest_block(block)
            
        # Save the updated graph back to the disk
        with open(GRAPH_FILE_PATH, 'wb') as f:
            pickle.dump(memos, f)
            
        return {"status": "success", "message": "Memory ingested and saved to disk!"}
    
    return {"status": "ignored", "message": "No long-term memories found."}

@app.post("/retrieve")
async def retrieve_memory(request: Request):
    data = await request.json()
    query = data.get("query", "")
    
    context = memos.retrieve_envelope(query)
    return {"context": context}

if __name__ == "__main__":
    import uvicorn
    # Runs the server locally on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)