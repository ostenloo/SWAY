# Model Specifications

## Storage Summary (Production-Ready Models)

| Model | Storage |
|-------|---------|
| Qwen3-4B | ~4 GB |
| Mistral-7B BNB-4bit | ~6 GB |
| Llama 3.1 8B | ~9 GB |
| Ornith-9B AWQ-FP8 | ~9 GB |
| Gemma 4 12B | ~14 GB |
| **Total Production** | **~42 GB** |
| GLM-4.5-Air | ~10 GB (incompatible) |
| GLM-4.6V | ~15 GB (incompatible) |
| **Total All** | **~67 GB** |

---

## Qwen3-4B

| Property | Value |
|----------|-------|
| **Model ID** | cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit |
| **HuggingFace URL** | https://huggingface.co/cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit |
| **Parameters** | 4 Billion |
| **Architecture** | Decoder-only Transformer |
| **Quantization** | AWQ 4-bit (compressed-tensors) |
| **VRAM Usage** | 3.8 GB |
| **Storage** | ~4 GB |
| **Context Length** | 128K tokens |
| **Port** | 8100 |

## Gemma 4 12B

| Property | Value |
|----------|-------|
| **Model ID** | cyankiwi/gemma-4-12B-it-AWQ-INT4 |
| **HuggingFace URL** | https://huggingface.co/cyankiwi/gemma-4-12B-it-AWQ-INT4 |
| **Parameters** | 12 Billion |
| **Architecture** | Multimodal LM |
| **Quantization** | INT4 (compressed-tensors) |
| **VRAM Usage** | 8.9 GB |
| **Storage** | ~14 GB |
| **Context Length** | 8K tokens |
| **Port** | 8010 |

## Llama 3.1 8B

| Property | Value |
|----------|-------|
| **Model ID** | unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit |
| **HuggingFace URL** | https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit |
| **Parameters** | 8 Billion |
| **Architecture** | Decoder-only Transformer |
| **Quantization** | BNB 4-bit |
| **VRAM Usage** | 5.9 GB |
| **Storage** | ~9 GB |
| **Context Length** | 128K tokens |
| **Port** | 8020 |

## Mistral-7B BNB-4bit

| Property | Value |
|----------|-------|
| **Model ID** | unsloth/mistral-7b-instruct-v0.3-bnb-4bit |
| **HuggingFace URL** | https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit |
| **Parameters** | 7 Billion |
| **Architecture** | Decoder-only Transformer |
| **Quantization** | BNB 4-bit |
| **VRAM Usage** | 4.3 GB |
| **Storage** | ~6 GB |
| **Context Length** | 32K tokens |
| **Port** | 8120 |

## Ornith-9B AWQ-FP8

| Property | Value |
|----------|-------|
| **Model ID** | cyankiwi/Ornith-1.0-9B-AWQ-FP8 |
| **HuggingFace URL** | https://huggingface.co/cyankiwi/Ornith-1.0-9B-AWQ-FP8 |
| **Parameters** | 9 Billion |
| **Architecture** | Decoder-only Transformer |
| **Quantization** | AWQ-FP8 |
| **VRAM Usage** | 10.8 GB |
| **Storage** | ~9 GB |
| **Context Length** | 4K tokens |
| **Port** | 8130 |

## GLM-4.5-Air

| Property | Value |
|----------|-------|
| **Model ID** | cyankiwi/GLM-4.5-Air-AWQ-4bit |
| **HuggingFace URL** | https://huggingface.co/cyankiwi/GLM-4.5-Air-AWQ-4bit |
| **Parameters** | ~9 Billion |
| **Architecture** | Decoder-only Transformer (GLM4-MoE) |
| **Quantization** | AWQ 4-bit (compressed-tensors) |
| **VRAM Usage** | ~7 GB (estimated) |
| **Storage** | ~10 GB |
| **Context Length** | 128K tokens |
| **Port** | 8070 |

## GLM-4.6V

| Property | Value |
|----------|-------|
| **Model ID** | cyankiwi/GLM-4.6V-AWQ-4bit |
| **HuggingFace URL** | https://huggingface.co/cyankiwi/GLM-4.6V-AWQ-4bit |
| **Parameters** | ~13 Billion |
| **Architecture** | Vision-Language Transformer |
| **Quantization** | AWQ 4-bit (compressed-tensors) |
| **VRAM Usage** | ~10 GB (estimated) |
| **Storage** | ~15 GB |
| **Context Length** | 128K tokens |
| **Port** | 8080 |

---

## VRAM Summary (Measured - Individual Tests)

| Model | VRAM | Params | GB/B | Status |
|-------|------|--------|------|--------|
| Qwen3-4B | 3.8 GB | 4B | 0.95 | ✓ Confirmed |
| Mistral-7B BNB-4bit | 4.3 GB | 7B | 0.61 | ✓ Confirmed |
| Llama 3.1 8B | 5.9 GB | 8B | 0.74 | ✓ Confirmed |
| Gemma 4 12B | 8.9 GB | 12B | 0.74 | ✓ Confirmed |
| Ornith-9B AWQ-FP8 | 10.8 GB | 9B | 1.20 | ✓ Confirmed |
| GLM-4.5-Air | ~7 GB | 9B | — | ❌ Incompatible |
| GLM-4.6V | ~10 GB | 13B | — | ❌ Incompatible |
| **Total (production-ready)** | **33.7 GB** | 38B | — | 5 models working |

**Testing Results:**
- ✓ **5 models production-ready**: Qwen3-4B, Mistral-7B BNB-4bit, Llama 3.1 8B, Ornith-9B, Gemma 4 12B
- ❌ **2 models with library incompatibility (Not Recommended)**:
  - **GLM-4.5-Air**: Requires Glm4MoeForCausalLM + torchvision incompatibility
  - **GLM-4.6V**: Requires AriaTextConfig + torchvision compatibility issues

**Model Distribution Strategy (32GB GPU):**

**Tier 1 - Lightweight (≤5GB each):**
- Qwen3-4B (3.8GB) - Most efficient, fast inference
- Mistral-7B BNB-4bit (4.3GB) - Best efficiency/capability ratio

**Tier 2 - Standard (5-9GB each):**
- Llama 3.1 8B (5.9GB) - Versatile general purpose
- Gemma 4 12B (8.9GB) - Multimodal capable

**Tier 3 - Large (10GB+):**
- Ornith-9B AWQ-FP8 (10.8GB) - Good reasoning capability

**Recommended Deployment Profiles:**

1. **Simultaneous Small + Medium (18.6GB used, 13.4GB free)**
   - Qwen3-4B + Mistral-7B BNB + Llama 3.1 8B
   - Best for mixed workloads needing variety

2. **Multi-Small Stack (13.5GB used, 18.5GB free)**
   - Qwen3-4B + Mistral-7B BNB + Llama 3.1 8B (lite mode)
   - Most headroom, lightweight tasks

3. **Balanced Pair (19.7GB used, 12.3GB free)**
   - Ornith-9B + Gemma 4 12B
   - Heavy reasoning + multimodal

4. **Premium Pair (16.7GB used, 15.3GB free)**
   - Ornith-9B + Llama 3.1 8B
   - Balanced capability and efficiency

5. **Solo Mode**
   - Run any single model alone for maximum VRAM
   - Fastest inference, dedicated resources

