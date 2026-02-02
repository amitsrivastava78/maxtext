# Guide 4: Weight Conversion - From HuggingFace to JAX

## Overview
This guide explains how to convert pretrained LLaMA weights from HuggingFace's PyTorch format to JAX-compatible NumPy arrays. Understanding this is crucial for loading any pretrained model into the Kascade implementation.

---

## Part 1: Weight Storage Architecture

### HuggingFace Format
```
HuggingFace Hub: meta-llama/Llama-3.2-1B
├── config.json                (Model architecture config)
├── model.safetensors.index.json  (Weight shard mapping)
├── model-00001-of-00002.safetensors  (Weights part 1)
└── model-00002-of-00002.safetensors  (Weights part 2)

Total size: ~5 GB
Format: PyTorch tensors (float32, bfloat16)
Device: CPU tensors by default
```

### MaxText Chunked Format
```
llama_weights_chunked/
├── layer_0.pkl   (128 MB)  ← Layer 0 weights only
├── layer_1.pkl   (128 MB)  ← Layer 1 weights only
├── layer_2.pkl   (128 MB)
├── ...
└── layer_15.pkl  (128 MB)  ← Layer 15 weights only

Total size: ~2 GB
Format: NumPy arrays (float32)
Device: Device-agnostic (JAX handles placement)
```

### Why Convert?
1. **Framework Compatibility** - PyTorch → JAX requires NumPy intermediate
2. **Memory Efficiency** - Load one layer at a time instead of full model
3. **Device Agnostic** - NumPy arrays can be placed on CPU/TPU/GPU by JAX
4. **Fast Loading** - Pickle files load faster than safetensors

---

## Part 2: Download Model from HuggingFace

### Code
```python
from huggingface_hub import snapshot_download

def download_llama_model(model_id="meta-llama/Llama-3.2-1B", local_dir="./llama_model"):
    """
    Download LLaMA model from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model identifier
        local_dir: Local directory to save model
    
    Returns:
        path: Path to downloaded model
    """
    print(f"📥 Downloading {model_id} from HuggingFace Hub...")
    
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    print(f"✓ Downloaded to {path}")
    return path
```

### Example Usage
```python
model_path = download_llama_model()

# Output:
# 📥 Downloading meta-llama/Llama-3.2-1B from HuggingFace Hub...
# Downloading model-00001-of-00002.safetensors: 100%|█| 2.5GB/2.5GB
# Downloading model-00002-of-00002.safetensors: 100%|█| 2.5GB/2.5GB
# ✓ Downloaded to ./llama_model
```

### Directory Structure After Download
```
llama_model/
├── config.json
│   {
│     "hidden_size": 2048,
│     "num_hidden_layers": 16,
│     "num_attention_heads": 32,
│     "num_key_value_heads": 8,
│     "intermediate_size": 5632,
│     "vocab_size": 128256,
│     ...
│   }
├── model.safetensors.index.json
│   {
│     "weight_map": {
│       "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
│       "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
│       ...
│     }
│   }
├── model-00001-of-00002.safetensors  (2.5 GB)
└── model-00002-of-00002.safetensors  (2.5 GB)
```

---

## Part 3: Load PyTorch Weights

### Code
```python
from transformers import LlamaForCausalLM
import torch

def load_pytorch_model(model_path):
    """
    Load LLaMA model from disk using HuggingFace transformers.
    
    Args:
        model_path: Path to model directory
    
    Returns:
        model: LlamaForCausalLM instance with loaded weights
    """
    print("📦 Loading PyTorch model...")
    
    # Load model (weights are on CPU by default)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu"  # Keep on CPU for conversion
    )
    
    # Set to eval mode (disable dropout, etc.)
    model.eval()
    
    print(f"✓ Loaded {model.config.num_hidden_layers} layers")
    return model
```

### Example Model Structure
```python
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-15): 16 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(2048, 2048, bias=False)
          (k_proj): Linear(2048, 512, bias=False)  ← GQA: 8 heads × 64
          (v_proj): Linear(2048, 512, bias=False)  ← GQA: 8 heads × 64
          (o_proj): Linear(2048, 2048, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(2048, 5632, bias=False)
          (up_proj): Linear(2048, 5632, bias=False)
          (down_proj): Linear(5632, 2048, bias=False)
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(2048, 128256, bias=False)
)

Total parameters: ~1.24 billion
```

---

## Part 4: Weight Name Mapping

### HuggingFace → MaxText Naming
```python
# HuggingFace naming convention:
model.layers.0.self_attn.q_proj.weight  → [2048, 2048]
model.layers.0.self_attn.k_proj.weight  → [2048, 512]
model.layers.0.self_attn.v_proj.weight  → [2048, 512]
model.layers.0.self_attn.o_proj.weight  → [2048, 2048]

model.layers.0.mlp.gate_proj.weight     → [5632, 2048]
model.layers.0.mlp.up_proj.weight       → [5632, 2048]
model.layers.0.mlp.down_proj.weight     → [2048, 5632]

model.layers.0.input_layernorm.weight   → [2048]
model.layers.0.post_attention_layernorm.weight → [2048]

# MaxText naming convention:
layer_0['attention']['wq']  → [2048, 2048]
layer_0['attention']['wk']  → [2048, 512]
layer_0['attention']['wv']  → [2048, 512]
layer_0['attention']['wo']  → [2048, 2048]

layer_0['feed_forward']['w1']  → [2048, 5632]  (gate)
layer_0['feed_forward']['w2']  → [5632, 2048]  (down)
layer_0['feed_forward']['w3']  → [2048, 5632]  (up)

layer_0['attention_norm']['weight']  → [2048]
layer_0['ffn_norm']['weight']        → [2048]
```

### Mapping Function
```python
def get_maxtext_name(hf_name):
    """
    Convert HuggingFace name to MaxText name.
    
    Examples:
        'model.layers.0.self_attn.q_proj.weight' → 'layer_0/attention/wq'
        'model.layers.0.mlp.gate_proj.weight' → 'layer_0/feed_forward/w1'
    """
    if 'self_attn.q_proj' in hf_name:
        return hf_name.replace('model.layers.', 'layer_') \
                     .replace('.self_attn.q_proj.weight', '/attention/wq')
    elif 'self_attn.k_proj' in hf_name:
        return hf_name.replace('model.layers.', 'layer_') \
                     .replace('.self_attn.k_proj.weight', '/attention/wk')
    elif 'self_attn.v_proj' in hf_name:
        return hf_name.replace('model.layers.', 'layer_') \
                     .replace('.self_attn.v_proj.weight', '/attention/wv')
    elif 'self_attn.o_proj' in hf_name:
        return hf_name.replace('model.layers.', 'layer_') \
                     .replace('.self_attn.o_proj.weight', '/attention/wo')
    elif 'mlp.gate_proj' in hf_name:
        return hf_name.replace('model.layers.', 'layer_') \
                     .replace('.mlp.gate_proj.weight', '/feed_forward/w1')
    elif 'mlp.down_proj' in hf_name:
        return hf_name.replace('model.layers.', 'layer_') \
                     .replace('.mlp.down_proj.weight', '/feed_forward/w2')
    elif 'mlp.up_proj' in hf_name:
        return hf_name.replace('model.layers.', 'layer_') \
                     .replace('.mlp.up_proj.weight', '/feed_forward/w3')
    elif 'input_layernorm' in hf_name:
        return hf_name.replace('model.layers.', 'layer_') \
                     .replace('.input_layernorm.weight', '/attention_norm/weight')
    elif 'post_attention_layernorm' in hf_name:
        return hf_name.replace('model.layers.', 'layer_') \
                     .replace('.post_attention_layernorm.weight', '/ffn_norm/weight')
    else:
        return hf_name
```

---

## Part 5: PyTorch → NumPy Conversion

### Why `.cpu()` is Needed

**PyTorch Tensor Properties:**
```python
# PyTorch tensor on GPU
tensor_gpu = torch.randn(2048, 2048, device='cuda')
print(tensor_gpu.device)  # cuda:0
print(tensor_gpu.numpy())  # ERROR! Can't convert GPU tensor directly

# Must move to CPU first
tensor_cpu = tensor_gpu.cpu()
print(tensor_cpu.device)  # cpu
numpy_array = tensor_cpu.numpy()  # ✓ Works!
print(type(numpy_array))  # <class 'numpy.ndarray'>
```

**Important Note:**
- `.cpu()` moves PyTorch tensor to CPU memory
- `.numpy()` creates NumPy view (zero-copy on CPU)
- NumPy arrays are device-agnostic
- JAX will place them on the target device (CPU/TPU/GPU)

### Conversion Code
```python
def convert_layer_weights(pytorch_model, layer_idx):
    """
    Convert one layer's weights from PyTorch to NumPy.
    
    Args:
        pytorch_model: Loaded LlamaForCausalLM
        layer_idx: Layer index (0-15)
    
    Returns:
        layer_dict: Dictionary of NumPy arrays
    """
    layer = pytorch_model.model.layers[layer_idx]
    
    # Extract attention weights
    attention = {
        'wq': layer.self_attn.q_proj.weight.cpu().numpy().T,  # Transpose!
        'wk': layer.self_attn.k_proj.weight.cpu().numpy().T,
        'wv': layer.self_attn.v_proj.weight.cpu().numpy().T,
        'wo': layer.self_attn.o_proj.weight.cpu().numpy().T,
    }
    
    # Extract feed-forward weights
    feed_forward = {
        'w1': layer.mlp.gate_proj.weight.cpu().numpy().T,  # Transpose!
        'w2': layer.mlp.down_proj.weight.cpu().numpy().T,
        'w3': layer.mlp.up_proj.weight.cpu().numpy().T,
    }
    
    # Extract normalization weights
    attention_norm = {
        'weight': layer.input_layernorm.weight.cpu().numpy()
    }
    ffn_norm = {
        'weight': layer.post_attention_layernorm.weight.cpu().numpy()
    }
    
    layer_dict = {
        'attention': attention,
        'feed_forward': feed_forward,
        'attention_norm': attention_norm,
        'ffn_norm': ffn_norm
    }
    
    return layer_dict
```

### Why Transpose?

**PyTorch Convention:**
```python
# Linear layer: out = input @ weight.T + bias
# weight shape: [out_features, in_features]

q_proj.weight.shape = [2048, 2048]  # [out, in]

# Forward:
# x: [batch, seq, 2048]
# output = x @ weight.T = [batch, seq, 2048] @ [2048, 2048] = [batch, seq, 2048]
```

**JAX Convention:**
```python
# Linear layer: out = input @ weight + bias
# weight shape: [in_features, out_features]

wq.shape = [2048, 2048]  # [in, out]

# Forward:
# x: [batch, seq, 2048]
# output = x @ weight = [batch, seq, 2048] @ [2048, 2048] = [batch, seq, 2048]
```

**Example:**
```python
# PyTorch weight (before transpose)
q_proj.weight: [2048, 2048]  # [out, in]
[[w00, w01, w02, ..., w0_2047],  ← Output neuron 0
 [w10, w11, w12, ..., w1_2047],  ← Output neuron 1
 ...
 [w2047_0, w2047_1, ..., w2047_2047]]  ← Output neuron 2047

# After .T (transpose)
wq: [2048, 2048]  # [in, out]
[[w00, w10, ..., w2047_0],  ← Input feature 0
 [w01, w11, ..., w2047_1],  ← Input feature 1
 ...
 [w0_2047, w1_2047, ..., w2047_2047]]  ← Input feature 2047
```

---

## Part 6: Save Chunked Weights

### Code
```python
import pickle
import os

def save_layer_weights(layer_dict, output_dir, layer_idx):
    """
    Save layer weights to pickle file.
    
    Args:
        layer_dict: Dictionary of NumPy arrays
        output_dir: Output directory (e.g., 'llama_weights_chunked')
        layer_idx: Layer index (0-15)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'layer_{layer_idx}.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(layer_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Check file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Layer {layer_idx}: {size_mb:.1f} MB")
```

### Example Output
```
Converting LLaMA weights to MaxText format...

Processing Layer 0...
  attention/wq: [2048, 2048] float32
  attention/wk: [2048, 512] float32
  attention/wv: [2048, 512] float32
  attention/wo: [2048, 2048] float32
  feed_forward/w1: [2048, 5632] float32
  feed_forward/w2: [5632, 2048] float32
  feed_forward/w3: [2048, 5632] float32
  Layer 0: 127.8 MB

Processing Layer 1...
  Layer 1: 127.8 MB

...

Processing Layer 15...
  Layer 15: 127.8 MB

✓ Conversion complete!
Total size: 2044.8 MB (2.0 GB)
Saved to: llama_weights_chunked/
```

---

## Part 7: Complete Conversion Script

### Full Pipeline
```python
import torch
from transformers import LlamaForCausalLM
from huggingface_hub import snapshot_download
import pickle
import os

def convert_llama_weights(
    model_id="meta-llama/Llama-3.2-1B",
    output_dir="llama_weights_chunked"
):
    """
    Complete pipeline: Download → Load → Convert → Save
    
    Args:
        model_id: HuggingFace model ID
        output_dir: Output directory for chunked weights
    """
    print("=" * 70)
    print("LLAMA WEIGHT CONVERSION PIPELINE")
    print("=" * 70)
    
    # Step 1: Download from HuggingFace Hub
    print("\n📥 Step 1: Downloading from HuggingFace Hub...")
    model_path = snapshot_download(
        repo_id=model_id,
        local_dir="./llama_model",
        local_dir_use_symlinks=False
    )
    print(f"✓ Downloaded to {model_path}")
    
    # Step 2: Load PyTorch model
    print("\n📦 Step 2: Loading PyTorch model...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    print(f"✓ Loaded {model.config.num_hidden_layers} layers")
    
    # Step 3: Convert each layer
    print("\n🔄 Step 3: Converting weights...")
    os.makedirs(output_dir, exist_ok=True)
    
    num_layers = model.config.num_hidden_layers
    total_size = 0
    
    for layer_idx in range(num_layers):
        print(f"\nProcessing Layer {layer_idx}...")
        
        # Extract layer
        layer = model.model.layers[layer_idx]
        
        # Convert to NumPy (with transpose)
        layer_dict = {
            'attention': {
                'wq': layer.self_attn.q_proj.weight.cpu().numpy().T,
                'wk': layer.self_attn.k_proj.weight.cpu().numpy().T,
                'wv': layer.self_attn.v_proj.weight.cpu().numpy().T,
                'wo': layer.self_attn.o_proj.weight.cpu().numpy().T,
            },
            'feed_forward': {
                'w1': layer.mlp.gate_proj.weight.cpu().numpy().T,
                'w2': layer.mlp.down_proj.weight.cpu().numpy().T,
                'w3': layer.mlp.up_proj.weight.cpu().numpy().T,
            },
            'attention_norm': {
                'weight': layer.input_layernorm.weight.cpu().numpy()
            },
            'ffn_norm': {
                'weight': layer.post_attention_layernorm.weight.cpu().numpy()
            }
        }
        
        # Print shapes
        print(f"  attention/wq: {layer_dict['attention']['wq'].shape}")
        print(f"  attention/wk: {layer_dict['attention']['wk'].shape}")
        print(f"  attention/wv: {layer_dict['attention']['wv'].shape}")
        print(f"  attention/wo: {layer_dict['attention']['wo'].shape}")
        print(f"  feed_forward/w1: {layer_dict['feed_forward']['w1'].shape}")
        print(f"  feed_forward/w2: {layer_dict['feed_forward']['w2'].shape}")
        print(f"  feed_forward/w3: {layer_dict['feed_forward']['w3'].shape}")
        
        # Save to pickle
        output_path = os.path.join(output_dir, f'layer_{layer_idx}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(layer_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        total_size += size_mb
        print(f"  Saved: {size_mb:.1f} MB")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"Total layers: {num_layers}")
    print(f"Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(f"Output directory: {output_dir}/")
    print("\nFiles created:")
    for i in range(num_layers):
        print(f"  - layer_{i}.pkl")
    print("=" * 70)

# Run conversion
if __name__ == "__main__":
    convert_llama_weights()
```

### Usage
```bash
python convert_llama_weights.py

# Output:
======================================================================
LLAMA WEIGHT CONVERSION PIPELINE
======================================================================

📥 Step 1: Downloading from HuggingFace Hub...
Downloading model-00001-of-00002.safetensors: 100%|█| 2.5GB/2.5GB
Downloading model-00002-of-00002.safetensors: 100%|█| 2.5GB/2.5GB
✓ Downloaded to ./llama_model

📦 Step 2: Loading PyTorch model...
✓ Loaded 16 layers

🔄 Step 3: Converting weights...

Processing Layer 0...
  attention/wq: (2048, 2048)
  attention/wk: (2048, 512)
  attention/wv: (2048, 512)
  attention/wo: (2048, 2048)
  feed_forward/w1: (2048, 5632)
  feed_forward/w2: (5632, 2048)
  feed_forward/w3: (2048, 5632)
  Saved: 127.8 MB

...

======================================================================
✅ CONVERSION COMPLETE!
======================================================================
Total layers: 16
Total size: 2044.8 MB (2.00 GB)
Output directory: llama_weights_chunked/

Files created:
  - layer_0.pkl
  - layer_1.pkl
  ...
  - layer_15.pkl
======================================================================
```

---

## Part 8: Loading Converted Weights in JAX

### Code
```python
import jax
import jax.numpy as jnp
import pickle

def load_layer_for_jax(weights_dir, layer_idx, device='cpu'):
    """
    Load layer weights and place on JAX device.
    
    Args:
        weights_dir: Directory with .pkl files
        layer_idx: Layer to load
        device: 'cpu', 'tpu', or 'gpu'
    
    Returns:
        layer_params: Dictionary of JAX arrays
    """
    # Load NumPy arrays from pickle
    pkl_path = f"{weights_dir}/layer_{layer_idx}.pkl"
    with open(pkl_path, 'rb') as f:
        layer_dict = pickle.load(f)
    
    # Convert to JAX arrays (automatically placed on configured device)
    layer_params = jax.tree_map(lambda x: jnp.array(x), layer_dict)
    
    return layer_params
```

### Example
```python
# Configure JAX device
jax.config.update('jax_platform_name', 'cpu')

# Load layer 0
params = load_layer_for_jax('llama_weights_chunked', layer_idx=0)

print("Layer 0 weights:")
print(f"  wq: {params['attention']['wq'].shape} on {params['attention']['wq'].device()}")
# Output: wq: (2048, 2048) on CpuDevice(id=0)

# If configured for TPU:
jax.config.update('jax_platform_name', 'tpu')
params = load_layer_for_jax('llama_weights_chunked', layer_idx=0)
print(f"  wq: {params['attention']['wq'].shape} on {params['attention']['wq'].device()}")
# Output: wq: (2048, 2048) on TpuDevice(id=0)
```

---

## Summary: Conversion Pipeline

```
1. Download from HuggingFace Hub
   ├─ Model: meta-llama/Llama-3.2-1B
   ├─ Format: PyTorch safetensors
   └─ Size: ~5 GB

2. Load PyTorch Model
   ├─ Framework: transformers.LlamaForCausalLM
   ├─ Device: CPU (for conversion)
   └─ 16 layers loaded

3. Convert Each Layer
   ├─ Extract weights: q_proj, k_proj, v_proj, o_proj, gate, up, down
   ├─ Move to CPU: .cpu()  ← Required for .numpy()
   ├─ Convert to NumPy: .numpy()
   ├─ Transpose: .T  ← PyTorch vs JAX convention
   └─ Validate shapes

4. Save Chunked Files
   ├─ Format: Pickle (fast, native Python)
   ├─ One file per layer: layer_0.pkl, layer_1.pkl, ...
   ├─ Size per layer: ~128 MB
   └─ Total: ~2 GB (50% compression from FP16→FP32)

5. Load in JAX
   ├─ Load NumPy from pickle
   ├─ Convert to JAX arrays
   ├─ Device placement: Automatic based on jax.config
   └─ Ready for inference!
```

---

## Key Insights

1. **`.cpu()` is for PyTorch→NumPy** - Not a JAX device lock
2. **NumPy is device-agnostic** - JAX handles device placement
3. **Transpose weights** - PyTorch [out, in] → JAX [in, out]
4. **Chunked storage** - Load one layer at a time (memory efficient)
5. **Pickle for speed** - Faster than safetensors for small files

---

## Troubleshooting

### Issue: "Can't convert CUDA tensor to NumPy"
```python
# ❌ Wrong:
weight = model.layer.weight.numpy()  # Error if on GPU

# ✓ Correct:
weight = model.layer.weight.cpu().numpy()  # Move to CPU first
```

### Issue: "Shape mismatch in linear layer"
```python
# ❌ Wrong: Forgot transpose
wq = q_proj.weight.cpu().numpy()  # [2048, 2048] wrong order

# ✓ Correct: Transpose for JAX
wq = q_proj.weight.cpu().numpy().T  # [2048, 2048] correct order
```

### Issue: "Out of memory during conversion"
```python
# ✓ Solution: Convert one layer at a time
for layer_idx in range(16):
    layer_dict = convert_layer(model, layer_idx)
    save_layer(layer_dict, layer_idx)
    del layer_dict  # Free memory
```

This completes the weight conversion guide. You now understand the full pipeline from HuggingFace to JAX-compatible format!
