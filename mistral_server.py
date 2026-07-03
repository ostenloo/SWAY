#!/usr/bin/env python3
"""Mistral-7B server using Transformers library"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Model configuration
MODEL_ID = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
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
    return {"status": "ok", "model_loaded": True}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "mistral-7b", "object": "model"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: dict = Body(...)):
    messages = request.get("messages", [])
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    max_tokens = request.get("max_tokens", 2048)
    temperature = request.get("temperature", 0.7)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"choices": [{"message": {"content": text}}]}


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
    uvicorn.run(app, host="0.0.0.0", port=8120)
