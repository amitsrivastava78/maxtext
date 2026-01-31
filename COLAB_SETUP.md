# Running Kascade on Google Colab

## Complete Setup for Colab (TPU or CPU)

### Step 1: Clone Repository

```bash
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/maxtext.git
%cd maxtext
```

### Step 2: Install Dependencies

```bash
# Install required packages
!pip install -q jax[tpu] flax numpy transformers huggingface_hub torch
```

For CPU runtime:
```bash
!pip install -q jax flax numpy transformers huggingface_hub torch
```

### Step 3: Download LLaMA 3.2-1B from Hugging Face

You have two options:

**Option A: Using Hugging Face Hub (Recommended)**

```python
from huggingface_hub import snapshot_download
import os

# Login to Hugging Face (you'll need a token with Llama access)
from huggingface_hub import login
login()  # Will prompt for token

# Download LLaMA 3.2-1B
checkpoint_dir = snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B",
    local_dir="./llama-checkpoint",
    local_dir_use_symlinks=False
)
print(f"✓ Downloaded to: {checkpoint_dir}")
```

**Option B: Using transformers library**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download model (requires HF token with Llama access)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype="float16",
    cache_dir="./llama-checkpoint"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Save in Meta format for conversion
model.save_pretrained("./llama-checkpoint/meta-format")
print("✓ Model downloaded and saved")
```

### Step 4: Convert Weights to JAX Format

```bash
# Convert PyTorch weights to JAX chunked format
!python convert_llama_weights.py \
    --input ./llama-checkpoint/meta-format \
    --chunked
```

This creates `llama_weights_chunked/` directory (~3GB).

### Step 5: Run Benchmark

**On TPU:**
```bash
!python benchmark_kascade_final.py --device tpu
```

**On CPU:**
```bash
!python benchmark_kascade_final.py --device cpu
```

**With custom parameters:**
```bash
!python benchmark_kascade_final.py \
    --device tpu \
    --top_k 12 \
    --threshold 0.65 \
    --tile_size 16
```

---

## Complete Colab Notebook Template

```python
# ========================================
# Kascade Sparse Attention Benchmark
# Google Colab Setup
# ========================================

# 1. Setup
!git clone https://github.com/YOUR_USERNAME/maxtext.git
%cd maxtext

# 2. Install dependencies (TPU runtime)
!pip install -q jax[tpu] flax numpy transformers huggingface_hub torch

# 3. Login to Hugging Face
from huggingface_hub import login
login()  # Enter your HF token

# 4. Download LLaMA 3.2-1B
from huggingface_hub import snapshot_download
checkpoint_dir = snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B",
    local_dir="./llama-checkpoint",
    local_dir_use_symlinks=False
)

# 5. Convert weights
!python convert_llama_weights.py \
    --input {checkpoint_dir} \
    --chunked

# 6. Run benchmark on TPU
!python benchmark_kascade_final.py --device tpu

# 7. View results
# Check the output for:
# - Dense Perplexity
# - Sparse Perplexity  
# - Degradation %
```

---

## Quick Commands Summary

```bash
# 1. Clone & setup
!git clone https://github.com/YOUR_USERNAME/maxtext.git && cd maxtext
!pip install -q jax[tpu] flax numpy transformers huggingface_hub torch

# 2. Download (requires HF token)
!huggingface-cli login
!huggingface-cli download meta-llama/Llama-3.2-1B --local-dir llama-checkpoint

# 3. Convert weights
!python convert_llama_weights.py --input llama-checkpoint --chunked

# 4. Run on TPU
!python benchmark_kascade_final.py --device tpu
```

---

## Troubleshooting

### "No TPU devices found"
```python
import jax
print(jax.devices())  # Should show TPU devices
```

If no TPUs, check:
- Runtime → Change runtime type → Hardware accelerator → TPU v2

### "Out of memory"
Reduce sequence length or use CPU:
```bash
!python benchmark_kascade_final.py --device cpu
```

### "Hugging Face token required"
Get token from: https://huggingface.co/settings/tokens
- Need "Read" access
- Must accept Meta Llama license

### "Model not found"
Ensure you've accepted the Llama license:
1. Visit: https://huggingface.co/meta-llama/Llama-3.2-1B
2. Click "Agree and access repository"
3. Wait for approval (usually instant)

---

## Storage Requirements

- LLaMA checkpoint: ~2.5 GB
- Converted weights: ~3 GB
- **Total: ~5.5 GB**

Google Colab free tier provides 12GB disk, so this should fit comfortably.

---

## Expected Runtime

- Weight conversion: ~2 minutes
- Benchmark execution:
  - CPU: ~15-20 minutes
  - TPU v2: ~3-5 minutes ⚡

---

## Saving Results

To save results to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Run benchmark and save output
!python benchmark_kascade_final.py --device tpu 2>&1 | tee /content/drive/MyDrive/kascade_results.log
```
