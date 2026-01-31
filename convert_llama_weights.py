"""
LLaMA-3.1-8B Weight Converter for JAX/Flax
-------------------------------------------
Converts PyTorch checkpoints to JAX format with:
1. Weight transposition (.T) for Dense layers
2. GQA expansion (8 KV heads → 32 Q heads)
3. Chunked saving for <24GB RAM

Usage:
    python convert_llama_weights.py --input /path/to/llama-3.1-8b --output converted_weights.pkl
"""

import torch
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm


def expand_gqa_weights(weight, num_q_heads=32, num_kv_heads=8):
    """
    Expand GQA K/V weights from 8 heads to 32 heads.
    
    Args:
        weight: Weight tensor [embed_dim, num_kv_heads, head_dim]
        num_q_heads: Target number of query heads (32 for LLaMA-3.1-8B)
        num_kv_heads: Current number of KV heads (8 for LLaMA-3.1-8B)
    
    Returns:
        Expanded weight [embed_dim, num_q_heads, head_dim]
    """
    assert num_q_heads % num_kv_heads == 0, "Q heads must be divisible by KV heads"
    repeat_factor = num_q_heads // num_kv_heads  # 32 / 8 = 4
    
    # Reshape and repeat
    embed_dim = weight.shape[0]
    head_dim = weight.shape[2]
    weight_reshaped = weight.reshape(embed_dim, num_kv_heads, head_dim)
    weight_expanded = np.repeat(weight_reshaped, repeat_factor, axis=1)
    
    return weight_expanded


def convert_layer_weights(layer_idx, state_dict, config):
    """
    Convert a single transformer layer from PyTorch to JAX format.
    
    Args:
        layer_idx: Layer index
        state_dict: PyTorch state dict
        config: Model configuration dict
    
    Returns:
        Flax-compatible nested dict for this layer
    """
    prefix = f"model.layers.{layer_idx}"
    
    # Attention weights
    wq = state_dict[f"{prefix}.self_attn.q_proj.weight"].cpu().numpy()
    wk = state_dict[f"{prefix}.self_attn.k_proj.weight"].cpu().numpy()
    wv = state_dict[f"{prefix}.self_attn.v_proj.weight"].cpu().numpy()
    wo = state_dict[f"{prefix}.self_attn.o_proj.weight"].cpu().numpy()
    
    # Config
    embed_dim = config['hidden_size']  # 4096
    num_q_heads = config['num_attention_heads']  # 32
    num_kv_heads = config['num_key_value_heads']  # 8
    head_dim = embed_dim // num_q_heads  # 128
    
    # Reshape K/V for GQA expansion
    wk_reshaped = wk.reshape(num_kv_heads * head_dim, embed_dim)
    wv_reshaped = wv.reshape(num_kv_heads * head_dim, embed_dim)
    
    # Expand GQA: 8 KV heads → 32 Q heads
    wk_expanded = expand_gqa_weights(
        wk_reshaped.T.reshape(embed_dim, num_kv_heads, head_dim),
        num_q_heads, num_kv_heads
    ).reshape(embed_dim, num_q_heads * head_dim)
    
    wv_expanded = expand_gqa_weights(
        wv_reshaped.T.reshape(embed_dim, num_kv_heads, head_dim),
        num_q_heads, num_kv_heads
    ).reshape(embed_dim, num_q_heads * head_dim)
    
    # Transpose for JAX Dense layers (PyTorch: [Out, In] → JAX: [In, Out])
    wq = wq.T
    wk = wk_expanded.T
    wv = wv_expanded.T
    wo = wo.T
    
    # MLP weights
    w_gate = state_dict[f"{prefix}.mlp.gate_proj.weight"].cpu().numpy().T
    w_up = state_dict[f"{prefix}.mlp.up_proj.weight"].cpu().numpy().T
    w_down = state_dict[f"{prefix}.mlp.down_proj.weight"].cpu().numpy().T
    
    # LayerNorm weights
    ln_attn = state_dict[f"{prefix}.input_layernorm.weight"].cpu().numpy()
    ln_mlp = state_dict[f"{prefix}.post_attention_layernorm.weight"].cpu().numpy()
    
    # Build Flax nested dict
    return {
        'attention': {
            'q_proj': {'kernel': wq},
            'k_proj': {'kernel': wk},
            'v_proj': {'kernel': wv},
            'o_proj': {'kernel': wo},
        },
        'mlp': {
            'gate_proj': {'kernel': w_gate},
            'up_proj': {'kernel': w_up},
            'down_proj': {'kernel': w_down},
        },
        'input_layernorm': {'scale': ln_attn},
        'post_attention_layernorm': {'scale': ln_mlp},
    }


def convert_llama_weights(input_path, output_path, chunked=True):
    """
    Convert full LLaMA-3.1-8B model from PyTorch to JAX.
    
    Args:
        input_path: Path to PyTorch checkpoint directory
        output_path: Path to save converted weights
        chunked: If True, save per-layer to avoid OOM
    """
    input_path = Path(input_path)
    
    print("Loading PyTorch checkpoint...")
    # Load consolidated checkpoint
    ckpt_files = list(input_path.glob("consolidated*.pth")) or \
                 list(input_path.glob("pytorch_model*.bin"))
    
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {input_path}")
    
    # Load model
    state_dict = {}
    for ckpt_file in tqdm(ckpt_files, desc="Loading checkpoints"):
        state_dict.update(torch.load(ckpt_file, map_location='cpu'))
    
    # Load config
    config_file = input_path / "config.json"
    if not config_file.exists():
        # Default LLaMA-3.1-8B config
        config = {
            'hidden_size': 4096,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'num_hidden_layers': 32,
            'intermediate_size': 14336,
            'vocab_size': 128256,
            'rope_theta': 500000.0,
        }
        print("Warning: config.json not found, using default LLaMA-3.1-8B config")
    else:
        import json
        with open(config_file) as f:
            config = json.load(f)
    
    print(f"Model config: {config['num_hidden_layers']} layers, "
          f"{config['num_attention_heads']} Q heads, "
          f"{config['num_key_value_heads']} KV heads")
    
    # Convert embeddings
    print("\nConverting embeddings...")
    embed_tokens = state_dict['model.embed_tokens.weight'].cpu().numpy()
    final_ln = state_dict['model.norm.weight'].cpu().numpy()
    lm_head = state_dict.get('lm_head.weight', embed_tokens).cpu().numpy().T
    
    # Convert layers
    print("\nConverting transformer layers...")
    if chunked:
        # Save per-layer to avoid OOM
        output_dir = Path(output_path).parent / "llama_weights_chunked"
        output_dir.mkdir(exist_ok=True)
        
        # Save embeddings
        with open(output_dir / "embeddings.pkl", 'wb') as f:
            pickle.dump({
                'embed_tokens': embed_tokens,
                'norm': final_ln,
                'lm_head': lm_head,
                'config': config,
            }, f)
        
        # Save each layer separately
        for layer_idx in tqdm(range(config['num_hidden_layers']), desc="Converting layers"):
            layer_weights = convert_layer_weights(layer_idx, state_dict, config)
            with open(output_dir / f"layer_{layer_idx:02d}.pkl", 'wb') as f:
                pickle.dump(layer_weights, f)
        
        print(f"\n✓ Saved chunked weights to {output_dir}/")
        print(f"  - embeddings.pkl: {embed_tokens.nbytes / 1e9:.2f} GB")
        print(f"  - layer_XX.pkl: {config['num_hidden_layers']} files")
        
    else:
        # Save full model (requires ~16GB RAM)
        layers = []
        for layer_idx in tqdm(range(config['num_hidden_layers']), desc="Converting layers"):
            layers.append(convert_layer_weights(layer_idx, state_dict, config))
        
        full_weights = {
            'embed_tokens': embed_tokens,
            'layers': layers,
            'norm': final_ln,
            'lm_head': lm_head,
            'config': config,
        }
        
        print(f"\nSaving full weights to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(full_weights, f)
        
        print(f"✓ Saved {len(layers)} layers to {output_path}")
    
    print("\n✅ Conversion complete!")
    print("\nKey transformations applied:")
    print("  1. Weight transpose: PyTorch [Out, In] → JAX [In, Out]")
    print("  2. GQA expansion: 8 KV heads → 32 Q heads (4x repeat)")
    print("  3. RoPE theta: 500000 (LLaMA-3.1 extended context)")


def load_layer_params(layer_idx, weights_dir):
    """
    Lazy load a single layer's parameters.
    
    Args:
        layer_idx: Layer index to load
        weights_dir: Directory with chunked weights
    
    Returns:
        Layer parameters dict
    """
    weights_dir = Path(weights_dir)
    with open(weights_dir / f"layer_{layer_idx:02d}.pkl", 'rb') as f:
        return pickle.load(f)


def load_embeddings(weights_dir):
    """Load embedding weights and config."""
    weights_dir = Path(weights_dir)
    with open(weights_dir / "embeddings.pkl", 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert LLaMA-3.1-8B weights to JAX')
    parser.add_argument('--input', required=True, help='Path to PyTorch checkpoint directory')
    parser.add_argument('--output', default='llama_weights.pkl', help='Output path')
    parser.add_argument('--chunked', action='store_true', default=True,
                        help='Save per-layer (recommended for <24GB RAM)')
    
    args = parser.parse_args()
    
    convert_llama_weights(args.input, args.output, chunked=args.chunked)
