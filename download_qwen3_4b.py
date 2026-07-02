#!/usr/bin/env python3
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = 'cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit'
save_path = '/mnt/data/models/Qwen3-4B-Instruct-2507-AWQ-4bit'

print(f'Downloading {model_id}...')
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map='cpu',
    trust_remote_code=True,
)

print(f'Saving to {save_path}...')
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print('Download complete!')
