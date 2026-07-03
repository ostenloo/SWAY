import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

model_id = "cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit"
model_path = "/mnt/data/models/Qwen3-4B-Instruct-2507-AWQ-4bit"
logger.info(f"Loading model from {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

logger.info("Model loaded successfully!")


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": model_id, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    messages = request.get("messages", [])
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    max_tokens = request.get("max_tokens", 2048)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"choices": [{"message": {"content": text}}]}


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=request.max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
