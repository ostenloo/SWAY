#!/usr/bin/env python3
"""Ornith-9B server using Transformers library"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Model configuration
MODEL_ID = "cyankiwi/Ornith-1.0-9B-AWQ-FP8"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
logger.info("Model loaded successfully!")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95


@app.get("/health")
async def health():
    # Verify model is actually loaded in VRAM
    if model is None:
        return {"status": "error", "reason": "model not loaded"}, 503
    try:
        # Quick inference to verify model is ready
        inputs = tokenizer("test", return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs, max_new_tokens=1)
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        return {"status": "error", "reason": str(e)}, 503


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "ornith-9b", "object": "model"}]
    }


@app.post("/v1/completions")
async def completions(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=True,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8130)
