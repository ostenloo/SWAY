#!/usr/bin/env python3
"""Llama 3.1 8B server using Transformers library"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Model configuration
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16
)
logger.info("Model loaded successfully!")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "llama-3.1-8b",
                "object": "model",
                "owned_by": "meta",
                "permission": []
            }
        ]
    }


@app.post("/v1/completions")
async def completions(request: GenerateRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "object": "text_completion",
            "model": "llama-3.1-8b",
            "choices": [
                {
                    "text": response,
                    "index": 0,
                    "finish_reason": "stop"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)
