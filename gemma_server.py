#!/usr/bin/env python3
"""Gemma 4 12B server using Transformers library"""

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Model configuration
MODEL_ID = "/mnt/data/models/gemma-4-12B-it-AWQ-INT4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model {MODEL_ID} on {DEVICE}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
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
                "id": "gemma-4-12b",
                "object": "model",
                "owned_by": "google",
                "permission": []
            }
        ]
    }


@app.post("/v1/completions")
async def completions(request: GenerateRequest):
    try:
        messages = [
            {"role": "user", "content": request.prompt},
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )

        input_len = inputs["input_ids"].shape[-1]
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        return {
            "object": "text_completion",
            "model": "gemma-4-12b",
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
    uvicorn.run(app, host="0.0.0.0", port=8010)
