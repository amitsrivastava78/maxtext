#!/usr/bin/env python3
"""
Kascade Benchmark - Long Sequence Edition (up to 32K+)
=======================================================
Supports:
  - seq_len up to 32K on single TPU v5e/v6e (16GB)
  - Memory-efficient attention (no S*S matrix) for DENSE/ANCHOR
  - Tokamax SplashAttention with dynamic grid for REUSE on TPU
  - Chunked logits for perplexity (no 33GB logit tensor)
  - bf16 activations on TPU
  - Auto-adapting tile_size and top_k based on seq_len
"""

import sys
import os
import pickle
import argparse
from pathlib import Path

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

# Import Kascade layers
import importlib.util
spec = importlib.util.spec_from_file_location("kascade_layers", 
    os.path.join(src_path, "MaxText/layers/kascade_layers.py"))
kascade_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kascade_module)

KascadeAnchorAttention = kascade_module.KascadeAnchorAttention
KascadeReuseAttention = kascade_module.KascadeReuseAttention
DenseFullAttention = kascade_module.DenseFullAttention
KASCADE_CACHE = kascade_module.KASCADE_CACHE
precompute_freqs_cis = kascade_module.precompute_freqs_cis
apply_rope = kascade_module.apply_rope
BLOCK_SPARSE_AVAILABLE = getattr(kascade_module, 'BLOCK_SPARSE_AVAILABLE', False)

# Try to import Tokamax availability flag
try:
    _kernel_spec = importlib.util.spec_from_file_location("kascade_kernel",
        os.path.join(src_path, "MaxText/kernels/kascade_block_sparse_kernel.py"))
    _kernel_mod = importlib.util.module_from_spec(_kernel_spec)
    _kernel_spec.loader.exec_module(_kernel_mod)
    TOKAMAX_SPLASH_AVAILABLE = getattr(_kernel_mod, 'TOKAMAX_SPLASH_AVAILABLE', False)
except Exception:
    TOKAMAX_SPLASH_AVAILABLE = False

# Model Configuration (Fixed for LLaMA 3.2-1B)
WEIGHTS_DIR = "llama_weights_chunked"
NUM_LAYERS = 16
NUM_HEADS = 32
HEAD_DIM = 64
EMBED_DIM = 2048
MLP_DIM = 8192
VOCAB_SIZE = 128256

# LLaMA-3.2 RoPE Scaling
ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

# Defaults
DEFAULT_SEQ_LEN = 512
DEFAULT_THRESHOLD = 0.65
DEFAULT_MAX_REUSE_DIST = 4
DEFAULT_DEVICE = 'cpu'

# Global variables (set by parse_args)
TILE_SIZE = 16
TOP_K_OPTIMIZED = 8
SEQ_LEN = DEFAULT_SEQ_LEN
USE_SPLASH_KERNEL = False


def auto_params(seq_len):
    """Auto-select tile_size and top_k based on sequence length."""
    if seq_len >= 4096:
        tile_size = 128  # Match TPU block size
        num_tiles = seq_len // tile_size
        top_k = max(2, num_tiles // 8)  # ~12.5% of tiles
    elif seq_len >= 1024:
        tile_size = 64
        num_tiles = seq_len // tile_size
        top_k = max(2, num_tiles // 4)  # ~25% of tiles
    else:
        tile_size = 16
        num_tiles = seq_len // tile_size
        top_k = max(2, num_tiles // 2)  # ~50% of tiles
    return tile_size, top_k


def solve_head_mapping_corrected(reuse_tiles, anchor_tiles, num_heads):
    """Correct head mapping with proper dimensions."""
    mapping = {}
    total_score = 0
    for r_h in range(num_heads):
        r_set = set(np.array(reuse_tiles[:, r_h, :]).flatten())
        best_score = -1.0
        best_a_h = 0
        for a_h in range(num_heads):
            a_set = set(np.array(anchor_tiles[:, a_h, :]).flatten())
            intersection = len(r_set & a_set)
            union = len(r_set | a_set)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard > best_score:
                best_score = jaccard
                best_a_h = a_h
        mapping[r_h] = best_a_h
        total_score += best_score
    avg_score = total_score / num_heads
    return avg_score, mapping


def _build_schedule(consecutive_similarities, num_layers, threshold, max_reuse_dist):
    """Build schedule with given threshold."""
    schedule = {0: {"type": "DENSE"}}
    schedule[1] = {"type": "ANCHOR"}
    last_anchor = 1
    for i in range(2, num_layers):
        if i not in consecutive_similarities:
            schedule[i] = {"type": "ANCHOR"}
            last_anchor = i
            continue
        score = consecutive_similarities[i]
        distance = i - last_anchor
        if score >= threshold and distance < max_reuse_dist:
            schedule[i] = {"type": "REUSE", "anchor_id": last_anchor, "head_map": {}}
        else:
            schedule[i] = {"type": "ANCHOR"}
            last_anchor = i
    reuse_count = sum(1 for v in schedule.values() if v["type"] == "REUSE")
    return schedule, reuse_count


def generate_schedule_structure(consecutive_similarities, num_layers, threshold=0.65, max_reuse_dist=4):
    """Generate schedule with auto-adaptive threshold."""
    print(f"\n  Generating Optimized Schedule:")
    sim_values = [v for v in consecutive_similarities.values()]
    if sim_values:
        median_sim = sorted(sim_values)[len(sim_values) // 2]
        print(f"   Layer similarities: min={min(sim_values):.2%}, median={median_sim:.2%}, max={max(sim_values):.2%}")
    else:
        median_sim = 0.0
    schedule, reuse_count = _build_schedule(consecutive_similarities, num_layers, threshold, max_reuse_dist)
    if reuse_count == 0 and sim_values:
        adaptive_threshold = median_sim * 0.90
        schedule_adaptive, reuse_adaptive = _build_schedule(consecutive_similarities, num_layers, adaptive_threshold, max_reuse_dist)
        if reuse_adaptive > 0:
            print(f"   Auto-adapted threshold: {adaptive_threshold:.2%} -> {reuse_adaptive} REUSE layers")
            schedule = schedule_adaptive
            reuse_count = reuse_adaptive
    else:
        print(f"   Threshold: {threshold:.2%} -> {reuse_count} REUSE layers")
    print(f"  Layer 0: DENSE")
    print(f"  Layer 1: ANCHOR")
    for i in range(2, num_layers):
        plan = schedule[i]
        score = consecutive_similarities.get(i, 0)
        if plan["type"] == "REUSE":
            print(f"  Layer {i}: REUSE L{plan['anchor_id']} (sim: {score:.2%})")
        else:
            print(f"  Layer {i}: ANCHOR (sim: {score:.2%})")
    return schedule


def calibrate_on_real_text_optimized(params, calib_ids, threshold, max_reuse_dist):
    """Calibrate with proper anchor-to-reuse head mappings."""
    print("\n  Calibrating on real text...")
    print(f"   Data shape: {calib_ids.shape}")
    all_anchor = {i: {"type": "ANCHOR"} for i in range(NUM_LAYERS)}
    model = LlamaModel(schedule=all_anchor)
    KASCADE_CACHE.clear()
    _ = model.apply(params, calib_ids)
    all_tiles = {}
    for i in range(NUM_LAYERS):
        tiles = KASCADE_CACHE.get(f"layer_{i}_indices_calib")
        if tiles is None:
            per_query = KASCADE_CACHE.get(f"layer_{i}_indices")
            if per_query is not None:
                tiles = per_query[:, :, -1, :] if per_query.ndim == 4 else per_query
        if tiles is not None:
            all_tiles[i] = tiles
    consecutive_similarities = {}
    for i in range(2, NUM_LAYERS):
        if i in all_tiles and i-1 in all_tiles:
            score, _ = solve_head_mapping_corrected(all_tiles[i], all_tiles[i-1], NUM_HEADS)
            consecutive_similarities[i] = score
    schedule = generate_schedule_structure(consecutive_similarities, NUM_LAYERS, threshold, max_reuse_dist)
    print("\n  Computing Head Mappings...")
    for layer_id, plan in schedule.items():
        if plan["type"] == "REUSE":
            anchor_id = plan["anchor_id"]
            if layer_id in all_tiles and anchor_id in all_tiles:
                score, head_map = solve_head_mapping_corrected(all_tiles[layer_id], all_tiles[anchor_id], NUM_HEADS)
                plan["head_map"] = head_map
                print(f"   Layer {layer_id} -> Anchor {anchor_id}: {score:.2%}")
    return schedule


# --- WEIGHT LOADING ---
def load_embeddings(weights_dir=WEIGHTS_DIR):
    with open(Path(weights_dir) / "embeddings.pkl", 'rb') as f:
        return pickle.load(f)

def load_layer_params(layer_idx, weights_dir=WEIGHTS_DIR):
    with open(Path(weights_dir) / f"layer_{layer_idx:02d}.pkl", 'rb') as f:
        return pickle.load(f)

def load_all_weights(weights_dir=WEIGHTS_DIR):
    print("Loading pretrained weights...")
    emb_data = load_embeddings(weights_dir)
    print(f"   Original dtypes: embed={emb_data['embed_tokens'].dtype}")
    emb_data['embed_tokens'] = emb_data['embed_tokens'].astype(jnp.float32)
    emb_data['norm'] = emb_data['norm'].astype(jnp.float32)
    emb_data['lm_head'] = emb_data['lm_head'].astype(jnp.float32)
    params = {
        'tok_embeddings': {'embedding': emb_data['embed_tokens']},
        'norm': {'scale': emb_data['norm']},
        'output': {'kernel': emb_data['lm_head']},
    }
    for i in range(emb_data['config']['num_hidden_layers']):
        lw = load_layer_params(i, weights_dir)
        wq = lw['attention']['q_proj']['kernel'].astype(jnp.float32)
        wk = lw['attention']['k_proj']['kernel'].astype(jnp.float32)
        wv = lw['attention']['v_proj']['kernel'].astype(jnp.float32)
        wo = lw['attention']['o_proj']['kernel'].astype(jnp.float32)
        params[f'layer_{i}'] = {
            'attention_norm': {'scale': lw['input_layernorm']['scale'].astype(jnp.float32)},
            'ffn_norm': {'scale': lw['post_attention_layernorm']['scale'].astype(jnp.float32)},
            'DenseFullAttention_0': {
                'Dense_0': {'kernel': wq}, 'Dense_1': {'kernel': wk},
                'Dense_2': {'kernel': wv}, 'Dense_3': {'kernel': wo}
            },
            'KascadeAnchorAttention_0': {
                'Dense_0': {'kernel': wq}, 'Dense_1': {'kernel': wk},
                'Dense_2': {'kernel': wv}, 'Dense_3': {'kernel': wo}
            },
            'KascadeReuseAttention_0': {
                'Dense_0': {'kernel': wq}, 'Dense_1': {'kernel': wk},
                'Dense_2': {'kernel': wv}, 'Dense_3': {'kernel': wo}
            },
            'gate_proj': {'kernel': lw['mlp']['gate_proj']['kernel'].astype(jnp.float32)},
            'up_proj': {'kernel': lw['mlp']['up_proj']['kernel'].astype(jnp.float32)},
            'down_proj': {'kernel': lw['mlp']['down_proj']['kernel'].astype(jnp.float32)},
        }
    print(f"  Loaded {emb_data['config']['num_hidden_layers']} layers")
    return {'params': params}, emb_data['config']


def calculate_full_sequence_perplexity_chunked(model, params, input_ids, chunk_size=1024):
    """Calculate perplexity using CHUNKED LOGITS to avoid OOM.
    
    Instead of materializing [1, seq_len, 128256] logits (33GB at 32K bf16),
    computes logits in chunks of chunk_size and accumulates NLL.
    
    Args:
        model: The LlamaModel (returns hidden states if return_hidden=True)
        params: Model parameters
        input_ids: [B, seq_len]
        chunk_size: Tokens to process at once for the final projection
    Returns:
        perplexity (float)
    """
    seq_len = input_ids.shape[1]
    
    # For short sequences, just compute normally
    if seq_len <= 4096:
        logits = model.apply(params, input_ids)
        return _ppl_from_logits(logits, input_ids)
    
    # For long sequences, get hidden states then chunk the lm_head projection
    hidden = model.apply(params, input_ids, method=model.forward_hidden)
    # hidden: [B, seq_len, embed_dim]
    
    lm_head_kernel = params['params']['output']['kernel']  # [embed_dim, vocab]
    
    # Compute NLL in chunks
    total_nll = 0.0
    total_tokens = 0
    shift_targets = input_ids[:, 1:]  # [B, seq_len-1]
    
    for start in range(0, seq_len - 1, chunk_size):
        end = min(start + chunk_size, seq_len - 1)
        # hidden[:, start:end, :] predicts targets[:, start:end]
        hidden_chunk = hidden[:, start:end, :]  # [B, chunk, embed_dim]
        logits_chunk = hidden_chunk @ lm_head_kernel  # [B, chunk, vocab]
        targets_chunk = shift_targets[:, start:end]  # [B, chunk]
        
        log_probs = jax.nn.log_softmax(logits_chunk, axis=-1)
        one_hot = jax.nn.one_hot(targets_chunk, logits_chunk.shape[-1])
        token_log_probs = jnp.sum(one_hot * log_probs, axis=-1)
        
        total_nll += float(-jnp.sum(token_log_probs))
        total_tokens += (end - start)
    
    avg_nll = total_nll / total_tokens
    ppl = np.exp(avg_nll)
    print(f"   Chunked PPL: {total_tokens} tokens, avg_NLL={avg_nll:.4f}, PPL={ppl:.4f}")
    return ppl


def _ppl_from_logits(logits, targets):
    """Standard perplexity from full logits tensor."""
    shift_logits = logits[:, :-1, :]
    shift_targets = targets[:, 1:]
    seq_len_eval = shift_logits.shape[1]
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    one_hot = jax.nn.one_hot(shift_targets, logits.shape[-1])
    token_log_probs = jnp.sum(one_hot * log_probs, axis=-1)
    avg_nll = -jnp.mean(token_log_probs)
    ppl = jnp.exp(avg_nll)
    last_nll = -token_log_probs[0, -1]
    last_ppl = jnp.exp(last_nll)
    print(f"   {seq_len_eval} tokens, avg_NLL={float(avg_nll):.4f}, PPL={float(ppl):.4f}")
    return ppl


def calculate_full_sequence_perplexity(logits, targets):
    """Calculate perplexity from pre-computed logits (short sequences)."""
    return _ppl_from_logits(logits, targets)


# --- MODEL CLASSES ---
class LlamaBlock(nn.Module):
    layer_id: int = 0
    schedule: dict = None
    use_splash: bool = False
    
    @nn.compact
    def __call__(self, x):
        normed = nn.RMSNorm(epsilon=1e-5, name="attention_norm")(x)
        seq_len = x.shape[1]
        freq_cis = precompute_freqs_cis(HEAD_DIM, seq_len, theta=500000.0, rope_scaling=ROPE_SCALING)
        plan = self.schedule.get(self.layer_id, {"type": "ANCHOR"})
        if plan["type"] == "DENSE":
            attn = DenseFullAttention(NUM_HEADS, HEAD_DIM)
            attn_out = attn(normed, freq_cis=freq_cis)
        elif plan["type"] == "ANCHOR":
            attn = KascadeAnchorAttention(
                NUM_HEADS, HEAD_DIM, self.layer_id,
                top_k_tiles=TOP_K_OPTIMIZED, tile_size=TILE_SIZE,
                use_splash=self.use_splash and USE_SPLASH_KERNEL)
            attn_out = attn(normed, freq_cis=freq_cis)
        else:
            attn = KascadeReuseAttention(
                NUM_HEADS, HEAD_DIM, plan["anchor_id"],
                tile_size=TILE_SIZE, head_map=plan["head_map"],
                use_splash=self.use_splash and USE_SPLASH_KERNEL)
            attn_out = attn(normed, freq_cis=freq_cis)
        x = x + attn_out
        normed = nn.RMSNorm(epsilon=1e-5, name="ffn_norm")(x)
        gate = nn.Dense(MLP_DIM, use_bias=False, name="gate_proj")(normed)
        up = nn.Dense(MLP_DIM, use_bias=False, name="up_proj")(normed)
        mlp_out = nn.Dense(EMBED_DIM, use_bias=False, name="down_proj")(nn.silu(gate) * up)
        return x + mlp_out


class LlamaModel(nn.Module):
    schedule: dict = None
    use_splash: bool = False
    
    @nn.compact
    def __call__(self, input_ids):
        hidden = self._body(input_ids)
        logits = nn.Dense(VOCAB_SIZE, use_bias=False, name="output")(hidden)
        return logits
    
    @nn.compact
    def forward_hidden(self, input_ids):
        """Return hidden states before lm_head (for chunked logits).
        Can be called via model.apply(params, ids, method=model.forward_hidden)."""
        return self._body(input_ids)
    
    def _body(self, input_ids):
        """Shared body: embed -> transformer blocks -> norm."""
        x = nn.Embed(VOCAB_SIZE, EMBED_DIM, name="tok_embeddings")(input_ids)
        for i in range(NUM_LAYERS):
            x = LlamaBlock(layer_id=i, schedule=self.schedule,
                          use_splash=self.use_splash, name=f"layer_{i}")(x)
        x = nn.RMSNorm(epsilon=1e-5, name="norm")(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kascade Sparse Attention Benchmark (Long Sequence Edition)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tile_size", type=int, default=None,
        help="Tile size (auto-selected if not set: 128 for >=4K, 64 for >=1K, 16 for <1K)")
    parser.add_argument("--top_k", type=int, default=None,
        help="Top-K tiles (auto-selected if not set)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max_reuse_dist", type=int, default=DEFAULT_MAX_REUSE_DIST)
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--weights_dir", type=str, default=WEIGHTS_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=['cpu', 'tpu', 'gpu'])
    parser.add_argument("--use_splash_kernel", action="store_true", default=False,
        help="Use SplashAttention for REUSE (Tokamax on TPU, masked-dense on CPU for short seq)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"  Configuring JAX for {args.device.upper()}...")
    jax.config.update('jax_platform_name', args.device)
    devices = jax.devices()
    print(f"  JAX: {len(devices)} {devices[0].platform.upper()} device(s)")
    
    # Auto-select tile_size and top_k
    global TILE_SIZE, TOP_K_OPTIMIZED, SEQ_LEN, USE_SPLASH_KERNEL
    SEQ_LEN = args.seq_len
    auto_ts, auto_tk = auto_params(SEQ_LEN)
    TILE_SIZE = args.tile_size if args.tile_size is not None else auto_ts
    TOP_K_OPTIMIZED = args.top_k if args.top_k is not None else auto_tk
    USE_SPLASH_KERNEL = args.use_splash_kernel
    
    # Ensure seq_len is divisible by tile_size
    if SEQ_LEN % TILE_SIZE != 0:
        old = SEQ_LEN
        SEQ_LEN = (SEQ_LEN // TILE_SIZE) * TILE_SIZE
        print(f"   Adjusted seq_len {old} -> {SEQ_LEN} (divisible by tile_size={TILE_SIZE})")
    
    num_tiles = SEQ_LEN // TILE_SIZE
    
    print("\n" + "=" * 70)
    print("  KASCADE BENCHMARK (Long-Sequence Edition)")
    print("=" * 70)
    print(f"\n  Configuration:")
    print(f"   Device:       {args.device.upper()}")
    print(f"   Seq Length:   {SEQ_LEN:,}")
    print(f"   Tile Size:    {TILE_SIZE} ({'auto' if args.tile_size is None else 'manual'})")
    print(f"   Top-K Tiles:  {TOP_K_OPTIMIZED}/{num_tiles} = {TOP_K_OPTIMIZED/num_tiles:.0%} ({'auto' if args.top_k is None else 'manual'})")
    print(f"   Threshold:    {args.threshold:.2%}")
    print(f"   Splash:       {'YES' if USE_SPLASH_KERNEL else 'NO'}")
    if USE_SPLASH_KERNEL:
        print(f"   Tokamax:      {'Available' if TOKAMAX_SPLASH_AVAILABLE else 'Not installed (will use fallback)'}")
        if args.device == 'tpu' and TOKAMAX_SPLASH_AVAILABLE:
            print(f"                 -> Tokamax SplashAttention with dynamic grid (real block skipping)")
        elif args.device == 'cpu' and SEQ_LEN <= 8192:
            print(f"                 -> Masked dense (CPU, short seq)")
        else:
            print(f"                 -> Gather fallback")
    
    # Memory estimate
    attn_mem_gb = SEQ_LEN * SEQ_LEN * 4 / 1e9  # float32 S*S matrix
    print(f"\n  Memory:")
    print(f"   S*S matrix (explicit): {attn_mem_gb:.1f} GB <- AVOIDED by memory-efficient attention")
    print(f"   Model weights: ~2.0 GB (float32)")
    if SEQ_LEN >= 8192:
        print(f"   Using chunked logits for perplexity (avoids {SEQ_LEN * VOCAB_SIZE * 4 / 1e9:.1f} GB)")
    
    # Load weights
    print("\n  Loading Weights...")
    params_dict, config = load_all_weights(args.weights_dir)
    
    # Prepare data
    print("\n  Loading C4 Dataset...")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not hf_token:
        print("  ERROR: Set HF_TOKEN environment variable")
        sys.exit(1)
    
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    
    all_tokens = []
    doc_count = 0
    needed_tokens = 3 * SEQ_LEN
    for example in dataset:
        if len(all_tokens) >= needed_tokens:
            break
        tokens = tokenizer.encode(example['text'][:100000], add_special_tokens=False)
        all_tokens.extend(tokens)
        doc_count += 1
        if doc_count >= 500:  # Enough docs for 32K+ tokens
            break
    
    print(f"   Collected {len(all_tokens):,} tokens from {doc_count} documents")
    
    if len(all_tokens) < 3 * SEQ_LEN:
        raise ValueError(f"Not enough tokens: {len(all_tokens)} < {3 * SEQ_LEN}")
    
    all_tokens = [min(t, VOCAB_SIZE - 1) for t in all_tokens]
    skip = SEQ_LEN // 2
    calib_ids = jnp.array([all_tokens[:SEQ_LEN]], dtype=jnp.int32)
    test_ids = jnp.array([all_tokens[SEQ_LEN+skip:2*SEQ_LEN+skip]], dtype=jnp.int32)
    print(f"   Calibration: {calib_ids.shape[1]:,} tokens")
    print(f"   Test:        {test_ids.shape[1]:,} tokens")
    
    # Calibrate
    schedule = calibrate_on_real_text_optimized(params_dict, calib_ids, args.threshold, args.max_reuse_dist)
    reuse_count = sum(1 for v in schedule.values() if v["type"] == "REUSE")
    anchor_count = sum(1 for v in schedule.values() if v["type"] == "ANCHOR")
    dense_count = sum(1 for v in schedule.values() if v["type"] == "DENSE")
    print(f"\n  Schedule: {dense_count} DENSE + {anchor_count} ANCHOR + {reuse_count} REUSE")
    
    # Dense baseline
    print("\n  Running DENSE Baseline...")
    dense_schedule = {i: {"type": "DENSE"} for i in range(NUM_LAYERS)}
    model_dense = LlamaModel(schedule=dense_schedule, use_splash=False)
    KASCADE_CACHE.clear()
    
    if SEQ_LEN > 4096:
        ppl_dense = calculate_full_sequence_perplexity_chunked(model_dense, params_dict, test_ids)
    else:
        logits_dense = model_dense.apply(params_dict, test_ids)
        ppl_dense = calculate_full_sequence_perplexity(logits_dense, test_ids)
    
    # Sparse Kascade
    print("\n  Running KASCADE Sparse...")
    model_sparse = LlamaModel(schedule=schedule, use_splash=USE_SPLASH_KERNEL)
    KASCADE_CACHE.clear()
    
    if SEQ_LEN > 4096:
        ppl_sparse = calculate_full_sequence_perplexity_chunked(model_sparse, params_dict, test_ids)
    else:
        logits_sparse = model_sparse.apply(params_dict, test_ids)
        ppl_sparse = calculate_full_sequence_perplexity(logits_sparse, test_ids)
    
    # Results
    print("\n" + "=" * 70)
    print("  RESULTS:")
    print("=" * 70)
    diff_pct = abs(ppl_sparse - ppl_dense) / ppl_dense * 100
    print(f"\n   Dense PPL:    {ppl_dense:.4f}")
    print(f"   Sparse PPL:   {ppl_sparse:.4f}")
    print(f"   Degradation:  {diff_pct:.4f}%")
    
    if diff_pct < 2.0:
        print(f"\n  SUCCESS! <2% degradation")
    elif diff_pct < 5.0:
        print(f"\n  Good! <5% degradation (paper target)")
    else:
        print(f"\n  Gap is {diff_pct:.2f}%")
    
    # Speedup benchmark
    print("\n" + "=" * 70)
    print("  SPEEDUP BENCHMARK")
    print("=" * 70)
    
    import time
    n_runs = 5 if SEQ_LEN <= 8192 else 3  # Fewer runs for long sequences
    
    print(f"\n  {n_runs} runs each (warmup 2)...")
    
    # Use forward_hidden for timing to avoid materializing full logits
    # At 32K: hidden=[1,32768,2048]=256MB vs logits=[1,32768,128256]=16.8GB
    use_hidden = SEQ_LEN > 4096
    
    # Free PPL intermediates before speedup benchmark
    import gc
    del ppl_dense, ppl_sparse
    gc.collect()
    
    # Warmup (2 rounds each)
    for _ in range(2):
        KASCADE_CACHE.clear()
        if use_hidden:
            out = model_dense.apply(params_dict, test_ids, method=model_dense.forward_hidden)
        else:
            out = model_dense.apply(params_dict, test_ids)
        jax.block_until_ready(out)
        del out
        
        KASCADE_CACHE.clear()
        if use_hidden:
            out = model_sparse.apply(params_dict, test_ids, method=model_sparse.forward_hidden)
        else:
            out = model_sparse.apply(params_dict, test_ids)
        jax.block_until_ready(out)
        del out
    
    # Dense timing
    dense_times = []
    for i in range(n_runs):
        KASCADE_CACHE.clear()
        start = time.perf_counter()
        if use_hidden:
            out = model_dense.apply(params_dict, test_ids, method=model_dense.forward_hidden)
        else:
            out = model_dense.apply(params_dict, test_ids)
        jax.block_until_ready(out)
        dense_times.append(time.perf_counter() - start)
        del out
    
    # Sparse timing
    sparse_times = []
    for i in range(n_runs):
        KASCADE_CACHE.clear()
        start = time.perf_counter()
        if use_hidden:
            out = model_sparse.apply(params_dict, test_ids, method=model_sparse.forward_hidden)
        else:
            out = model_sparse.apply(params_dict, test_ids)
        jax.block_until_ready(out)
        sparse_times.append(time.perf_counter() - start)
        del out
    
    dense_avg = sum(dense_times) / len(dense_times) * 1000
    sparse_avg = sum(sparse_times) / len(sparse_times) * 1000
    speedup = dense_avg / sparse_avg if sparse_avg > 0 else 0
    dense_min = min(dense_times) * 1000
    sparse_min = min(sparse_times) * 1000
    speedup_best = dense_min / sparse_min if sparse_min > 0 else 0
    
    print(f"\n  Timing (avg of {n_runs}):")
    print(f"   Dense:   {dense_avg:.1f} ms  (best: {dense_min:.1f} ms)")
    print(f"   Sparse:  {sparse_avg:.1f} ms  (best: {sparse_min:.1f} ms)")
    print(f"   Speedup: {speedup:.2f}x avg, {speedup_best:.2f}x best")
    
    # Analysis
    attn_fraction = (SEQ_LEN * SEQ_LEN) / (SEQ_LEN * SEQ_LEN + 3 * SEQ_LEN * EMBED_DIM + 3 * SEQ_LEN * MLP_DIM)
    sparse_ratio = TOP_K_OPTIMIZED / num_tiles
    
    # Estimate theoretical speedup
    if USE_SPLASH_KERNEL and args.device == 'tpu' and TOKAMAX_SPLASH_AVAILABLE:
        # Tokamax path: actual block skipping
        reuse_attn_speedup = 1.0 / sparse_ratio if sparse_ratio > 0 else 1.0  # e.g. 8x for 12.5%
        reuse_ratio = 1.0 / reuse_attn_speedup
        reuse_desc = f"Tokamax SplashAttention ({sparse_ratio:.0%} blocks, {reuse_attn_speedup:.1f}x attn speedup)"
    elif USE_SPLASH_KERNEL and SEQ_LEN <= 8192:
        reuse_ratio = 1.02
        reuse_desc = f"masked dense (short seq, ~dense speed)"
    else:
        reuse_ratio = max(0.3, sparse_ratio * 3)
        reuse_desc = f"gather sparse ({sparse_ratio:.0%} of tokens)"
    
    total_attn_cost = dense_count * 1.0 + anchor_count * 1.05 + reuse_count * reuse_ratio
    full_cost = NUM_LAYERS * 1.0
    theoretical_attn_speedup = full_cost / total_attn_cost if total_attn_cost > 0 else 1.0
    theoretical_total = 1.0 / (1 - attn_fraction + attn_fraction / theoretical_attn_speedup)
    
    print(f"\n  Analysis:")
    print(f"   Tiles: {num_tiles}, Top-K: {TOP_K_OPTIMIZED} ({sparse_ratio:.0%})")
    print(f"   Schedule: {dense_count}D + {anchor_count}A + {reuse_count}R")
    print(f"   REUSE: {reuse_desc}")
    print(f"   Attention fraction: {attn_fraction:.1%}")
    print(f"   Theoretical speedup: {theoretical_total:.2f}x")
    
    if SEQ_LEN >= 32768:
        print(f"\n   At seq_len={SEQ_LEN:,}, attention is {attn_fraction:.1%} of FLOPs")
        print(f"   This is the sweet spot for Kascade sparse attention!")
    elif SEQ_LEN >= 16384:
        print(f"\n   At seq_len={SEQ_LEN:,}, attention is {attn_fraction:.1%} -> moderate speedup expected")
    else:
        print(f"\n   At seq_len={SEQ_LEN:,}, attention is only {attn_fraction:.1%} of FLOPs")
        print(f"   Run with --seq_len 32768 for >10% speedup")
    
    if speedup > 1.1:
        print(f"\n  Sparse is {speedup:.2f}x faster!")
    elif speedup > 0.95:
        print(f"\n  Near-dense speed ({speedup:.2f}x) with {diff_pct:.1f}% quality degradation")
    else:
        print(f"\n  Sparse is {speedup:.2f}x (overhead from tiling/gather)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
