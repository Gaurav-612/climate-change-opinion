from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import json
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse




load_dotenv()

app = FastAPI()

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return jsonable_encoder(obj)
        except:
            return str(obj)

app.json_encoder = CustomJSONEncoder

def limit_depth(obj, max_depth=5):
    if isinstance(obj, dict) and max_depth > 0:
        return {k: limit_depth(v, max_depth - 1) for k, v in obj.items()}
    elif isinstance(obj, list) and max_depth > 0:
        return [limit_depth(i, max_depth - 1) for i in obj]
    else:
        return str(obj) if max_depth <= 0 else obj

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("reddit-comments-sample")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/query")
async def query_index(request: QueryRequest):
    try:
        # Here you'd typically convert the query to an embedding
        # For simplicity, let's assume we're querying with a random vector
        results = index.query(
            vector=[0]*384,  # Replace with actual query vector
            top_k=1,#request.top_k,
            include_metadata=False
        )
        # limited_results = limit_depth(results)
        return JSONResponse(content=results, encoder=CustomJSONEncoder)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    try:
        stats = index.describe_index_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)