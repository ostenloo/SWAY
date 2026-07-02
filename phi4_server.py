import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

model_id = "cyankiwi/Qwen3.5-9B-AWQ-INT8-INT4"
logger.info(f"Loading model {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

logger.info("Model loaded successfully!")


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "qwen3.5-9b", "object": "model"}]}


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=request.max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8120)
