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


def permute_for_rope(weight, num_heads, head_dim):
    """Permutes weights to convert from 'Half-Split' (HF) to 'Interleaved' (JAX) RoPE.
    
    HF RoPE pairs: (0, 64), (1, 65)... (pairs dimension i with i + dim/2)
    JAX RoPE pairs: (0, 1), (2, 3)... (pairs dimension i with i + 1)
    
    We need to reorder the OUTPUT dimension of the weights so that 
    HF's (0, 64) become JAX's (0, 1).
    
    Args:
        weight: Weight tensor [Embed_Dim, Num_Heads * Head_Dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each head
    
    Returns:
        Permuted weight with interleaved RoPE pairing
    """
    embed_dim = weight.shape[0]
    
    # 1. Reshape to isolate heads: [Embed, Heads, Head_Dim]
    w = weight.reshape(embed_dim, num_heads, head_dim)
    
    # 2. Split Head_Dim into two halves (HF style): [Embed, Heads, 2, Half_Dim]
    w = w.reshape(embed_dim, num_heads, 2, head_dim // 2)
    
    # 3. Transpose to interleave (Swap axis 2 and 3)
    # From [2, Half_Dim] (0..63, 64..127) -> [Half_Dim, 2] ((0,64), (1,65)...)
    w = w.transpose(0, 1, 3, 2)
    
    # 4. Flatten back: [Embed, Heads * Head_Dim]
    w = w.reshape(embed_dim, num_heads * head_dim)
    return w


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
    # Support both HuggingFace and Meta formats
    if f"model.layers.{layer_idx}.self_attn.q_proj.weight" in state_dict:
        # HuggingFace format
        prefix = f"model.layers.{layer_idx}"
        wq = state_dict[f"{prefix}.self_attn.q_proj.weight"].half().cpu().numpy()
        wk = state_dict[f"{prefix}.self_attn.k_proj.weight"].half().cpu().numpy()
        wv = state_dict[f"{prefix}.self_attn.v_proj.weight"].half().cpu().numpy()
        wo = state_dict[f"{prefix}.self_attn.o_proj.weight"].half().cpu().numpy()
        w_gate = state_dict[f"{prefix}.mlp.gate_proj.weight"].half().cpu().numpy()
        w_up = state_dict[f"{prefix}.mlp.up_proj.weight"].half().cpu().numpy()
        w_down = state_dict[f"{prefix}.mlp.down_proj.weight"].half().cpu().numpy()
        ln_attn = state_dict[f"{prefix}.input_layernorm.weight"].half().cpu().numpy()
        ln_mlp = state_dict[f"{prefix}.post_attention_layernorm.weight"].half().cpu().numpy()
    else:
        # Meta format
        prefix = f"layers.{layer_idx}"
        wq = state_dict[f"{prefix}.attention.wq.weight"].half().cpu().numpy()
        wk = state_dict[f"{prefix}.attention.wk.weight"].half().cpu().numpy()
        wv = state_dict[f"{prefix}.attention.wv.weight"].half().cpu().numpy()
        wo = state_dict[f"{prefix}.attention.wo.weight"].half().cpu().numpy()
        w_gate = state_dict[f"{prefix}.feed_forward.w3.weight"].half().cpu().numpy()  # Note: w3 is gate in Meta
        w_up = state_dict[f"{prefix}.feed_forward.w1.weight"].half().cpu().numpy()
        w_down = state_dict[f"{prefix}.feed_forward.w2.weight"].half().cpu().numpy()
        ln_attn = state_dict[f"{prefix}.attention_norm.weight"].half().cpu().numpy()
        ln_mlp = state_dict[f"{prefix}.ffn_norm.weight"].half().cpu().numpy()
    
    # Attention weights (now loaded above based on format)
    
    # Config
    embed_dim = config['hidden_size']  # 4096 or 2048
    num_q_heads = config['num_attention_heads']  # 32
    num_kv_heads = config['num_key_value_heads']  # 8
    head_dim = embed_dim // num_q_heads  # 128 or 64
    
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
    wk = wk_expanded  # Already correct shape from expand_gqa_weights
    wv = wv_expanded  # Already correct shape from expand_gqa_weights
    wo = wo.T
    
    # NOTE: RoPE permutation NOT applied - Meta weights already compatible with JAX
    
    # MLP weights (already loaded above based on format)
    w_gate = w_gate.T
    w_up = w_up.T
    w_down = w_down.T
    
    # LayerNorm weights (already loaded above based on format)
    
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
    params_file = input_path / "params.json"
    
    if config_file.exists():
        # HuggingFace format
        import json
        with open(config_file) as f:
            hf_config = json.load(f)
        config = {
            'hidden_size': hf_config.get('hidden_size', 4096),
            'num_attention_heads': hf_config.get('num_attention_heads', 32),
            'num_key_value_heads': hf_config.get('num_key_value_heads', 8),
            'num_hidden_layers': hf_config.get('num_hidden_layers', 32),
            'intermediate_size': hf_config.get('intermediate_size', 14336),
            'vocab_size': hf_config.get('vocab_size', 128256),
            'rope_theta': hf_config.get('rope_theta', 500000.0),
        }
    elif params_file.exists():
        # Meta format
        import json
        with open(params_file) as f:
            meta_params = json.load(f)
        config = {
            'hidden_size': meta_params['dim'],
            'num_attention_heads': meta_params['n_heads'],
            'num_key_value_heads': meta_params['n_kv_heads'],
            'num_hidden_layers': meta_params['n_layers'],
            'intermediate_size': int(meta_params['dim'] * 4 * meta_params.get('ffn_dim_multiplier', 1.0)),
            'vocab_size': meta_params['vocab_size'],
            'rope_theta': meta_params.get('rope_theta', 500000.0),
        }
    else:
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
        print("Warning: No config file found, using default LLaMA-3.1-8B config")
    
    print(f"Model config: {config['num_hidden_layers']} layers, "
          f"{config['num_attention_heads']} Q heads, "
          f"{config['num_key_value_heads']} KV heads")
    
    # Convert embeddings
    print("\nConverting embeddings...")
    
    # Support both HuggingFace and Meta formats
    if 'model.embed_tokens.weight' in state_dict:
        # HuggingFace format
        embed_tokens = state_dict['model.embed_tokens.weight'].half().cpu().numpy()
        final_ln = state_dict['model.norm.weight'].half().cpu().numpy()
        lm_head = state_dict.get('lm_head.weight', embed_tokens).half().cpu().numpy().T
    else:
        # Meta format
        embed_tokens = state_dict['tok_embeddings.weight'].half().cpu().numpy()
        final_ln = state_dict['norm.weight'].half().cpu().numpy()
        lm_head = state_dict.get('output.weight', embed_tokens).half().cpu().numpy().T
    
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
