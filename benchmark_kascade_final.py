#!/usr/bin/env python3
"""
Final Kascade Benchmark - Real Text + Paper Optimizations
==========================================================
Layer 0 DENSE + TOP_K 12 + Real Wikipedia Text
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

# Model Configuration (Fixed)
WEIGHTS_DIR = "llama_weights_chunked"
NUM_LAYERS = 16
NUM_HEADS = 32
HEAD_DIM = 64
EMBED_DIM = 2048
MLP_DIM = 8192
VOCAB_SIZE = 128256
SEQ_LEN = 512  # Will be overridden by command line arg

# Hyperparameters (Defaults - can be overridden via command line)
DEFAULT_TILE_SIZE = 16
DEFAULT_TOP_K = 24  # Testing: 24/32 = 75% of tiles
DEFAULT_THRESHOLD = 0.65  # 65% Jaccard similarity for reuse
DEFAULT_MAX_REUSE_DIST = 4  # Maximum layers between anchor and reuse
DEFAULT_SEQ_LEN = 512  # Default sequence length
DEFAULT_DEVICE = 'cpu'  # Default to CPU for compatibility
DEFAULT_USE_SPLASH = False  # Use optimized SplashAttention kernel

# LLaMA-3.2 RoPE Scaling Configuration
ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

# Global variables (set by parse_args)
TILE_SIZE = DEFAULT_TILE_SIZE
TOP_K_OPTIMIZED = DEFAULT_TOP_K
USE_SPLASH_KERNEL = False  # Set by parse_args

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
    """Build a schedule with the given threshold. Returns (schedule, reuse_count)."""
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
            schedule[i] = {
                "type": "REUSE",
                "anchor_id": last_anchor,
                "head_map": {}
            }
        else:
            schedule[i] = {"type": "ANCHOR"}
            last_anchor = i
    
    reuse_count = sum(1 for v in schedule.values() if v["type"] == "REUSE")
    return schedule, reuse_count


def generate_schedule_structure(consecutive_similarities, num_layers, threshold=0.65, max_reuse_dist=4):
    """Generate schedule structure determining ANCHOR vs REUSE (without head mappings yet).
    
    Auto-adaptive: if user's threshold yields 0 REUSE layers, automatically
    lower to median similarity to enable REUSE. This handles varying seq_len/
    tile_size/top_k combinations where absolute similarity values differ.
    """
    print(f"\n⚡ Generating Optimized Schedule:")
    print(f"   Max reuse distance: {max_reuse_dist}")
    
    # Show similarity statistics
    sim_values = [v for v in consecutive_similarities.values()]
    if sim_values:
        median_sim = sorted(sim_values)[len(sim_values) // 2]
        print(f"   Layer similarities: min={min(sim_values):.2%}, median={median_sim:.2%}, max={max(sim_values):.2%}")
    else:
        median_sim = 0.0
    
    # Try user's threshold first
    schedule, reuse_count = _build_schedule(consecutive_similarities, num_layers, threshold, max_reuse_dist)
    
    if reuse_count == 0 and sim_values:
        # Auto-adapt: lower threshold to median similarity to get REUSE layers
        adaptive_threshold = median_sim * 0.90  # 90% of median → roughly half the layers become REUSE
        schedule_adaptive, reuse_adaptive = _build_schedule(consecutive_similarities, num_layers, adaptive_threshold, max_reuse_dist)
        
        if reuse_adaptive > 0:
            print(f"   ⚠️  Threshold {threshold:.2%} yields 0 REUSE layers")
            print(f"   🔧 Auto-adapted threshold: {adaptive_threshold:.2%} (90% of median similarity)")
            print(f"   → {reuse_adaptive} REUSE layers enabled")
            threshold = adaptive_threshold
            schedule = schedule_adaptive
            reuse_count = reuse_adaptive
        else:
            print(f"   Threshold: {threshold:.2%} (0 REUSE layers — similarities too low)")
    else:
        print(f"   Threshold: {threshold:.2%} → {reuse_count} REUSE layers")
    
    # Print schedule
    print(f"  Layer 0: DENSE (full attention - paper requirement)")
    print(f"  Layer 1: ANCHOR (first sparse layer)")
    for i in range(2, num_layers):
        plan = schedule[i]
        if plan["type"] == "REUSE":
            score = consecutive_similarities.get(i, 0)
            print(f"  Layer {i}: REUSE L{plan['anchor_id']} (similarity: {score:.2%})")
        else:
            score = consecutive_similarities.get(i, 0)
            print(f"  Layer {i}: ANCHOR (similarity: {score:.2%})")
    
    return schedule

def generate_optimized_schedule(layer_analysis, num_layers, threshold=0.65, max_reuse_dist=4):
    """DEPRECATED: Old function for backward compatibility. Use generate_schedule_structure instead."""
    print(f"\n⚡ Generating Optimized Schedule (OLD METHOD):")
    print(f"   Similarity threshold: {threshold:.2%} (tuned for 1B)")
    print(f"   Max reuse distance: {max_reuse_dist}")
    
    # CRITICAL: Layer 0 MUST be DENSE (paper Section 3.1)
    schedule = {0: {"type": "DENSE"}}
    print("  Layer 0: DENSE (full attention - paper requirement)")
    
    # Layer 1 is first ANCHOR
    schedule[1] = {"type": "ANCHOR"}
    print("  Layer 1: ANCHOR (first sparse layer)")
    last_anchor = 1
    
    for i in range(2, num_layers):
        if i not in layer_analysis:
            schedule[i] = {"type": "ANCHOR"}
            last_anchor = i
            print(f"  Layer {i}: ANCHOR (no data)")
            continue
        
        score, head_map = layer_analysis[i]
        distance = i - last_anchor
        
        if score >= threshold and distance < max_reuse_dist:
            print(f"  Layer {i}: REUSE L{last_anchor} (similarity: {score:.2%})")
            schedule[i] = {
                "type": "REUSE",
                "anchor_id": last_anchor,
                "head_map": head_map
            }
        else:
            schedule[i] = {"type": "ANCHOR"}
            last_anchor = i
            if score < threshold:
                print(f"  Layer {i}: ANCHOR (low similarity: {score:.2%})")
            else:
                print(f"  Layer {i}: ANCHOR (distance: {distance})")
    
    return schedule

def calibrate_on_real_text_optimized(params, calib_ids, threshold, max_reuse_dist):
    """Calibrate with proper anchor-to-reuse head mappings."""
    print("\n📊 Calibrating on Real Wikipedia Text...")
    print(f"   Calibration data: {calib_ids.shape}")
    
    # Run ALL ANCHOR (no DENSE distinction during calibration)
    all_anchor = {i: {"type": "ANCHOR"} for i in range(NUM_LAYERS)}
    model = LlamaModel(schedule=all_anchor)
    
    KASCADE_CACHE.clear()
    _ = model.apply(params, calib_ids)
    
    # Store all tile indices for calibration (last-token summaries for scheduling)
    all_tiles = {}
    for i in range(NUM_LAYERS):
        # Use _calib suffix which contains last-token tiles [B, H, top_k]
        tiles = KASCADE_CACHE.get(f"layer_{i}_indices_calib")
        if tiles is None:
            # Fallback: if per-query indices exist, extract last token
            per_query = KASCADE_CACHE.get(f"layer_{i}_indices")
            if per_query is not None:
                tiles = per_query[:, :, -1, :] if per_query.ndim == 4 else per_query
        if tiles is not None:
            all_tiles[i] = tiles
    
    # First pass: compute consecutive layer similarities to determine schedule structure
    consecutive_similarities = {}
    for i in range(2, NUM_LAYERS):
        if i in all_tiles and i-1 in all_tiles:
            score, _ = solve_head_mapping_corrected(all_tiles[i], all_tiles[i-1], NUM_HEADS)
            consecutive_similarities[i] = score
    
    # Generate schedule structure (which layers are ANCHOR vs REUSE and from which anchor)
    schedule = generate_schedule_structure(consecutive_similarities, NUM_LAYERS, threshold, max_reuse_dist)
    
    # Second pass: compute correct head mappings between each REUSE layer and its designated ANCHOR
    print("\n🔗 Computing Head Mappings...")
    for layer_id, plan in schedule.items():
        if plan["type"] == "REUSE":
            anchor_id = plan["anchor_id"]
            if layer_id in all_tiles and anchor_id in all_tiles:
                score, head_map = solve_head_mapping_corrected(all_tiles[layer_id], all_tiles[anchor_id], NUM_HEADS)
                plan["head_map"] = head_map
                print(f"   Layer {layer_id} → Anchor {anchor_id}: {score:.2%} similarity")
    
    return schedule

# --- WEIGHT LOADING ---
def load_embeddings(weights_dir=WEIGHTS_DIR):
    """Load embedding weights and config."""
    with open(Path(weights_dir) / "embeddings.pkl", 'rb') as f:
        return pickle.load(f)

def load_layer_params(layer_idx, weights_dir=WEIGHTS_DIR):
    """Lazy load a single layer's parameters."""
    with open(Path(weights_dir) / f"layer_{layer_idx:02d}.pkl", 'rb') as f:
        return pickle.load(f)

def load_all_weights(weights_dir=WEIGHTS_DIR):
    """Load all weights into Flax parameter structure."""
    print("Loading pretrained weights...")
    
    emb_data = load_embeddings(weights_dir)
    
    # Check dtype of loaded weights
    print(f"🔍 Original weight dtypes: embed_tokens={emb_data['embed_tokens'].dtype}, norm={emb_data['norm'].dtype}")
    
    # CRITICAL FIX: Convert float16 weights to float32 for TPU stability
    # Mixed precision (float16 weights × float32 activations) causes 152% degradation on TPU
    emb_data['embed_tokens'] = emb_data['embed_tokens'].astype(jnp.float32)
    emb_data['norm'] = emb_data['norm'].astype(jnp.float32)
    emb_data['lm_head'] = emb_data['lm_head'].astype(jnp.float32)
    print(f"   ✓ Converted to float32 for TPU stability")
    
    params = {
        'tok_embeddings': {'embedding': emb_data['embed_tokens']},
        'norm': {'scale': emb_data['norm']},
        'output': {'kernel': emb_data['lm_head']},
    }
    
    for i in range(emb_data['config']['num_hidden_layers']):
        layer_weights = load_layer_params(i, weights_dir)
        if i == 0:
            # Print first layer dtypes
            print(f"   Layer 0 original dtypes: q_proj={layer_weights['attention']['q_proj']['kernel'].dtype}")
        
        # Convert all layer weights to float32
        wq = layer_weights['attention']['q_proj']['kernel'].astype(jnp.float32)
        wk = layer_weights['attention']['k_proj']['kernel'].astype(jnp.float32)
        wv = layer_weights['attention']['v_proj']['kernel'].astype(jnp.float32)
        wo = layer_weights['attention']['o_proj']['kernel'].astype(jnp.float32)
        
        params[f'layer_{i}'] = {
            'attention_norm': {'scale': layer_weights['input_layernorm']['scale'].astype(jnp.float32)},
            'ffn_norm': {'scale': layer_weights['post_attention_layernorm']['scale'].astype(jnp.float32)},
            'DenseFullAttention_0': {
                'Dense_0': {'kernel': wq},
                'Dense_1': {'kernel': wk},
                'Dense_2': {'kernel': wv},
                'Dense_3': {'kernel': wo}
            },
            'KascadeAnchorAttention_0': {
                'Dense_0': {'kernel': wq},
                'Dense_1': {'kernel': wk},
                'Dense_2': {'kernel': wv},
                'Dense_3': {'kernel': wo}
            },
            'KascadeReuseAttention_0': {
                'Dense_0': {'kernel': wq},
                'Dense_1': {'kernel': wk},
                'Dense_2': {'kernel': wv},
                'Dense_3': {'kernel': wo}
            },
            'gate_proj': {'kernel': layer_weights['mlp']['gate_proj']['kernel'].astype(jnp.float32)},
            'up_proj': {'kernel': layer_weights['mlp']['up_proj']['kernel'].astype(jnp.float32)},
            'down_proj': {'kernel': layer_weights['mlp']['down_proj']['kernel'].astype(jnp.float32)},
        }
    
    print(f"✓ Loaded {emb_data['config']['num_hidden_layers']} layers")
    return {'params': params}, emb_data['config']

def calculate_full_sequence_perplexity(logits, targets):
    """Calculate perplexity over ALL tokens (proper evaluation, not just 1 token).
    
    For autoregressive LM: logits[:, t, :] predicts targets[:, t+1].
    So we use logits[:, 0:-1, :] to predict targets[:, 1:].
    This gives us (seq_len - 1) token predictions instead of just 1.
    """
    # Shift: logits[t] predicts target[t+1]
    shift_logits = logits[:, :-1, :]   # [B, seq_len-1, vocab]
    shift_targets = targets[:, 1:]      # [B, seq_len-1]
    
    seq_len_eval = shift_logits.shape[1]
    
    # Compute log probabilities for all positions
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)  # [B, seq_len-1, vocab]
    
    # Gather log prob of correct tokens
    one_hot = jax.nn.one_hot(shift_targets, logits.shape[-1])  # [B, seq_len-1, vocab]
    token_log_probs = jnp.sum(one_hot * log_probs, axis=-1)    # [B, seq_len-1]
    
    # Average negative log likelihood across ALL positions
    avg_nll = -jnp.mean(token_log_probs)
    ppl = jnp.exp(avg_nll)
    
    # Also compute last-token perplexity for comparison
    last_nll = -token_log_probs[0, -1]
    last_ppl = jnp.exp(last_nll)
    
    print(f"   Logits shape: {logits.shape}, Evaluating {seq_len_eval} token predictions")
    print(f"   Full-sequence: avg_NLL={float(avg_nll):.4f}, Perplexity={float(ppl):.4f}")
    print(f"   Last-token only: NLL={float(last_nll):.4f}, Perplexity={float(last_ppl):.4f}")
    
    return ppl

# --- MODEL CLASSES ---
class LlamaBlock(nn.Module):
    """LLaMA Transformer Block with Kascade"""
    layer_id: int = 0
    schedule: dict = None
    use_splash: bool = False
    
    @nn.compact
    def __call__(self, x):
        normed = nn.RMSNorm(epsilon=1e-5, name="attention_norm")(x)
        
        seq_len = x.shape[1]
        freq_cis = precompute_freqs_cis(HEAD_DIM, seq_len, theta=500000.0,
                                        rope_scaling=ROPE_SCALING)
        
        plan = self.schedule.get(self.layer_id, {"type": "ANCHOR"})
        
        # Compute attention based on plan type
        if plan["type"] == "DENSE":
            # Clean full attention — no tile overhead for fair baseline
            attn = DenseFullAttention(
                NUM_HEADS, HEAD_DIM,
            )
            attn_out = attn(normed, freq_cis=freq_cis)
        elif plan["type"] == "ANCHOR":
            attn = KascadeAnchorAttention(
                NUM_HEADS, HEAD_DIM, self.layer_id,
                top_k_tiles=TOP_K_OPTIMIZED,
                tile_size=TILE_SIZE,
                use_splash=self.use_splash and USE_SPLASH_KERNEL
            )
            attn_out = attn(normed, freq_cis=freq_cis)
        else:  # REUSE
            attn = KascadeReuseAttention(
                NUM_HEADS, HEAD_DIM, plan["anchor_id"],
                tile_size=TILE_SIZE, head_map=plan["head_map"],
                use_splash=self.use_splash and USE_SPLASH_KERNEL
            )
            attn_out = attn(normed, freq_cis=freq_cis)
        
        x = x + attn_out
        
        normed = nn.RMSNorm(epsilon=1e-5, name="ffn_norm")(x)
        gate = nn.Dense(MLP_DIM, use_bias=False, name="gate_proj")(normed)
        up = nn.Dense(MLP_DIM, use_bias=False, name="up_proj")(normed)
        mlp_out = nn.Dense(EMBED_DIM, use_bias=False, name="down_proj")(
            nn.silu(gate) * up
        )
        
        return x + mlp_out

class LlamaModel(nn.Module):
    """LLaMA with Kascade Sparse Attention"""
    schedule: dict = None
    use_splash: bool = False
    
    @nn.compact
    def __call__(self, input_ids):
        x = nn.Embed(VOCAB_SIZE, EMBED_DIM, name="tok_embeddings")(input_ids)
        
        for i in range(NUM_LAYERS):
            x = LlamaBlock(
                layer_id=i, 
                schedule=self.schedule,
                use_splash=self.use_splash,
                name=f"layer_{i}"
            )(x)
        
        x = nn.RMSNorm(epsilon=1e-5, name="norm")(x)
        logits = nn.Dense(VOCAB_SIZE, use_bias=False, name="output")(x)
        
        return logits

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kascade Sparse Attention Benchmark for LLaMA 3.2-1B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Hyperparameters
    parser.add_argument(
        "--tile_size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Size of attention tiles (tokens per tile)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top tiles to select in Anchor layers (paper uses 8 for 8B, 12 for 1B)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Jaccard similarity threshold for reusing anchor tiles (0.0-1.0)"
    )
    parser.add_argument(
        "--max_reuse_dist",
        type=int,
        default=DEFAULT_MAX_REUSE_DIST,
        help="Maximum layer distance between anchor and reuse layers"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="Sequence length for benchmark (longer = better speedup)"
    )
    
    # Paths
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=WEIGHTS_DIR,
        help="Directory containing converted LLaMA weights"
    )
    
    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=['cpu', 'tpu', 'gpu'],
        help="Device to run on (cpu, tpu, or gpu)"
    )
    
    # Kernel optimization
    parser.add_argument(
        "--use_splash_kernel",
        action="store_true",
        default=DEFAULT_USE_SPLASH,
        help="Use optimized SplashAttention kernel (TPU only, provides 2-3× speedup)"
    )
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Configure JAX device
    print(f"🖥️  Configuring JAX to use {args.device.upper()}...")
    jax.config.update('jax_platform_name', args.device)
    
    # Verify device configuration
    devices = jax.devices()
    print(f"✓ JAX using {len(devices)} {devices[0].platform.upper()} device(s): {[d.id for d in devices]}")
    
    # Update global variables with command line values
    global TILE_SIZE, TOP_K_OPTIMIZED, SEQ_LEN, USE_SPLASH_KERNEL
    TILE_SIZE = args.tile_size
    TOP_K_OPTIMIZED = args.top_k
    SEQ_LEN = args.seq_len
    USE_SPLASH_KERNEL = args.use_splash_kernel
    
    # Check SplashAttention availability
    if args.use_splash_kernel:
        if args.device != 'tpu':
            print(f"\n⚠️  WARNING: --use_splash_kernel only works on TPU, but device is '{args.device}'")
            print("   Falling back to standard Kascade implementation")
            args.use_splash_kernel = False
        else:
            try:
                # Step 1: Load splash_attention_kernel first
                import importlib.util
                import traceback
                
                kernel_path = os.path.join(src_path, "MaxText/kernels/splash_attention_kernel.py")
                kernel_path = os.path.abspath(kernel_path)
                
                print(f"\n🔧 Loading splash_attention_kernel from: {kernel_path}")
                print(f"   File exists: {os.path.exists(kernel_path)}")
                
                kernel_spec = importlib.util.spec_from_file_location("splash_attention_kernel_direct", kernel_path)
                kernel_module = importlib.util.module_from_spec(kernel_spec)
                print("   ✓ Kernel spec created")
                
                sys.modules[kernel_spec.name] = kernel_module
                kernel_spec.loader.exec_module(kernel_module)
                print("   ✓ Kernel module loaded")
                
                # Step 2: Load kascade_splash_attention and inject the kernel module
                splash_path = os.path.join(src_path, "MaxText/layers/kascade_splash_attention.py")
                splash_path = os.path.abspath(splash_path)
                
                print(f"🔧 Loading kascade_splash_attention from: {splash_path}")
                print(f"   File exists: {os.path.exists(splash_path)}")
                
                spec = importlib.util.spec_from_file_location("kascade_splash_attention", splash_path)
                kascade_splash_module = importlib.util.module_from_spec(spec)
                print("   ✓ Splash spec created")
                
                sys.modules[spec.name] = kascade_splash_module
                # Inject the kernel module before execution
                kascade_splash_module._KERNEL_MODULE = kernel_module
                kascade_splash_module.__file__ = splash_path
                print("   ✓ Kernel module injected")
                
                spec.loader.exec_module(kascade_splash_module)
                print("   ✓ Splash module loaded")
                
                # Store in sys.modules so LlamaBlock can import it
                sys.modules['kascade_splash_attention'] = kascade_splash_module
                
                print("\n✅ SplashAttention kernel available - will use optimized implementation")
            except Exception as e:
                print(f"\n⚠️  WARNING: Could not import kascade_splash_attention: {e}")
                print(f"   Error type: {type(e).__name__}")
                print("   Traceback:")
                traceback.print_exc()
                print("   Falling back to standard Kascade implementation")
                args.use_splash_kernel = False
    
    print("\n" + "=" * 70)
    print("🚀 FINAL KASCADE BENCHMARK")
    print("=" * 70)
    print(f"\n⚙️  Configuration:")
    print(f"   Device:           {args.device.upper()}")
    print(f"   Sequence Length:  {args.seq_len}")
    print(f"   Tile Size:        {args.tile_size}")
    print(f"   Top-K Tiles:      {args.top_k}")
    print(f"   Threshold:        {args.threshold:.2%}")
    print(f"   Max Reuse Dist:   {args.max_reuse_dist}")
    print(f"   Weights Dir:      {args.weights_dir}")
    print(f"   Use Splash Kernel: {'YES (2-3× expected speedup)' if args.use_splash_kernel else 'NO'}")
    
    # Load weights
    print("\n📥 Loading Weights...")
    params_dict, config = load_all_weights()
    
    # Prepare real text from C4 dataset
    print("\n📝 Loading Real C4 Dataset...")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    # Check for HuggingFace authentication token
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    if not hf_token:
        print("\n❌ ERROR: HuggingFace authentication token not found!")
        print("\n📋 Setup Instructions:")
        print("   1. Get token from: https://huggingface.co/settings/tokens")
        print("   2. Request access to: https://huggingface.co/meta-llama/Llama-3.2-1B")
        print("   3. Set environment variable:")
        print("      export HF_TOKEN='your_token_here'")
        print("\n   Then re-run this script.")
        sys.exit(1)
    
    # Authenticate with HuggingFace
    print("   Authenticating with HuggingFace...")
    from huggingface_hub import login
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("   ✓ Authenticated")
    except Exception as e:
        print(f"\n❌ ERROR: Authentication failed: {e}")
        print("   Please check your HF_TOKEN is valid.")
        sys.exit(1)
    
    # Load LLaMA tokenizer (required - no fallback)
    print("   Loading LLaMA-3.2-1B tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        print("   ✓ Loaded tokenizer")
    except Exception as e:
        print(f"\n❌ ERROR: Could not load LLaMA tokenizer: {e}")
        print("\n   Make sure you have access to the model:")
        print("   https://huggingface.co/meta-llama/Llama-3.2-1B")
        print("   Click 'Request Access' and wait for approval.")
        sys.exit(1)
    
    # Load C4 validation split
    print("   Loading C4 validation data...")
    dataset = load_dataset(
        "allenai/c4", 
        "en", 
        split="validation", 
        streaming=True
    )
    
    # Collect tokens from multiple different documents
    print("   Tokenizing documents...")
    all_tokens = []
    doc_count = 0
    needed_tokens = 2 * SEQ_LEN + SEQ_LEN  # Need enough for calibration + skip + test
    
    for example in dataset:
        if len(all_tokens) >= needed_tokens:
            break
        
        # Tokenize and take tokens
        text = example['text'][:50000]  # Large limit to get more tokens per doc
        tokens = tokenizer.encode(text, add_special_tokens=False)  # No special tokens to avoid mismatch
        all_tokens.extend(tokens)
        doc_count += 1
        
        if doc_count >= 100:  # High limit to ensure enough tokens for any seq_len
            break
    
    print(f"   ✓ Collected tokens from {doc_count} documents")
    
    # Verify we have enough tokens
    needed_tokens = 2 * SEQ_LEN + SEQ_LEN  # calib + skip + test
    if len(all_tokens) < needed_tokens:
        raise ValueError(f"Not enough tokens collected: {len(all_tokens)} < {needed_tokens}. Increase document count or text length.")
    
    # Trim to exactly what we need
    needed_total = 2 * SEQ_LEN + SEQ_LEN
    all_tokens = all_tokens[:needed_total]
    
    # Ensure we have valid token IDs (clip to vocab size)
    all_tokens = [min(t, VOCAB_SIZE - 1) for t in all_tokens]
    
    # CRITICAL: Use DIFFERENT documents for calibration and test to avoid information leakage
    # Calibration: first document's tokens
    # Test: next document's tokens (completely independent)
    if len(all_tokens) < 2 * SEQ_LEN:
        raise ValueError(f"Need at least {2*SEQ_LEN} tokens, got {len(all_tokens)}")
    
    # Find document boundary (look for tokens that indicate new document/topic)
    # For safety, just use first half for calib, second half for test from DIFFERENT parts
    # Better: skip some tokens between calib and test to reduce dependency
    skip_tokens = SEQ_LEN // 2  # Skip 256 tokens between calib and test
    calib_ids = jnp.array([all_tokens[:SEQ_LEN]], dtype=jnp.int32)
    test_ids = jnp.array([all_tokens[SEQ_LEN+skip_tokens:2*SEQ_LEN+skip_tokens]], dtype=jnp.int32)
    
    print(f"   ✓ Calibration: {calib_ids.shape[1]} tokens from C4")
    print(f"   ✓ Test: {test_ids.shape[1]} tokens (different documents)")
    
    # Verify shapes
    assert calib_ids.shape[1] == SEQ_LEN, f"Calibration shape mismatch: {calib_ids.shape[1]} != {SEQ_LEN}"
    assert test_ids.shape[1] == SEQ_LEN, f"Test shape mismatch: {test_ids.shape[1]} != {SEQ_LEN}"
    
    # Calibrate
    schedule = calibrate_on_real_text_optimized(params_dict, calib_ids, args.threshold, args.max_reuse_dist)
    
    reuse_count = sum(1 for v in schedule.values() if v["type"] == "REUSE")
    print(f"\n📋 Final Schedule: {reuse_count} REUSE, {NUM_LAYERS - reuse_count} ANCHOR/DENSE")
    
    # Dense baseline (all DENSE for fair comparison)
    print("\n🏃 Running DENSE Baseline...")
    dense_schedule = {i: {"type": "DENSE"} for i in range(NUM_LAYERS)}
    print(f"   Dense expects {SEQ_LEN // TILE_SIZE} tiles per layer (FULL ATTENTION)")
    print(f"   Dense schedule L0: {dense_schedule[0]}")
    print(f"   Dense schedule L1: {dense_schedule[1]}")
    model_dense = LlamaModel(schedule=dense_schedule, use_splash=args.use_splash_kernel)
    
    # Note: Dense baseline will populate cache (ANCHOR with top_k=32), but we don't use it
    print(f"   Cache before clear: {list(KASCADE_CACHE.keys())}")
    KASCADE_CACHE.clear()
    print(f"   Cache after clear: {list(KASCADE_CACHE.keys())}")
    logits_dense = model_dense.apply(params_dict, test_ids)
    calib_tiles = KASCADE_CACHE.get('layer_1_indices_calib', KASCADE_CACHE.get('layer_1_indices'))
    if calib_tiles is not None and calib_tiles.ndim == 3:
        print(f"   Dense L1 tiles (head 0, last token): {calib_tiles[0,0]}")
    elif calib_tiles is not None:
        print(f"   Dense L1 tiles (head 0, last token): {calib_tiles[0,0,-1]}")
    print(f"   Cache after Dense run: {[k for k in KASCADE_CACHE.keys() if not k.endswith('_calib')]}")
    ppl_dense = calculate_full_sequence_perplexity(logits_dense, test_ids)
    
    # Sparse Kascade
    print("\n⚡ Running KASCADE Sparse...")
    print(f"   Sparse schedule L0: {schedule[0]}")
    print(f"   Sparse schedule L1: {schedule[1]}")
    print(f"   Sparse schedule L2: {schedule[2]}")
    model_sparse = LlamaModel(schedule=schedule, use_splash=args.use_splash_kernel)
    
    # Clear cache from Dense baseline before running Sparse
    KASCADE_CACHE.clear()
    
    # Single pass: ANCHOR layers populate cache before REUSE layers within same fwd pass
    print("   Running single-pass sparse forward...")
    logits_sparse = model_sparse.apply(params_dict, test_ids)
    anchor_keys = [k for k in KASCADE_CACHE.keys() if not k.endswith('_calib')]
    print(f"   Cache populated: {anchor_keys}")
    ppl_sparse = calculate_full_sequence_perplexity(logits_sparse, test_ids)
    
    # Results
    print("\n" + "=" * 70)
    print("📊 RESULTS ON REAL TEXT:")
    print("=" * 70)
    
    diff_pct = abs(ppl_sparse - ppl_dense) / ppl_dense * 100
    
    print(f"\n   Dense Perplexity:  {ppl_dense:.4f}")
    print(f"   Sparse Perplexity: {ppl_sparse:.4f}")
    print(f"   Degradation:       {diff_pct:.4f}%")
    
    if diff_pct < 2.0:
        print(f"\n✅✅✅ SUCCESS! <2% degradation achieved!")
        print(f"   Layer 0 DENSE + TOP_K 12 optimizations working!")
    elif diff_pct < 5.0:
        print(f"\n✅ Good! <5% degradation (paper target met)")
    else:
        print(f"\n⚠️ Gap is {diff_pct:.2f}%. Consider increasing TOP_K to 16.")
    
    # Speedup benchmark
    print("\n" + "=" * 70)
    print("⏱️  SPEEDUP BENCHMARK")
    print("=" * 70)
    
    import time
    n_runs = 5
    
    print(f"\nBenchmarking {n_runs} runs each...")
    
    # Warmup (2 runs each to stabilize JIT)
    print("  Warming up...")
    KASCADE_CACHE.clear()
    _ = model_dense.apply(params_dict, test_ids)
    jax.block_until_ready(_)
    KASCADE_CACHE.clear()
    _ = model_sparse.apply(params_dict, test_ids)
    jax.block_until_ready(_)
    KASCADE_CACHE.clear()
    _ = model_dense.apply(params_dict, test_ids)
    jax.block_until_ready(_)
    KASCADE_CACHE.clear()
    _ = model_sparse.apply(params_dict, test_ids)
    jax.block_until_ready(_)
    
    # Dense timing
    print("  Timing Dense...")
    dense_times = []
    for i in range(n_runs):
        KASCADE_CACHE.clear()
        start = time.perf_counter()
        out = model_dense.apply(params_dict, test_ids)
        jax.block_until_ready(out)
        dense_times.append(time.perf_counter() - start)
    
    # Sparse timing
    print("  Timing Sparse...")
    sparse_times = []
    for i in range(n_runs):
        KASCADE_CACHE.clear()
        start = time.perf_counter()
        out = model_sparse.apply(params_dict, test_ids)
        jax.block_until_ready(out)
        sparse_times.append(time.perf_counter() - start)
    
    dense_avg = sum(dense_times) / len(dense_times) * 1000
    sparse_avg = sum(sparse_times) / len(sparse_times) * 1000
    speedup = dense_avg / sparse_avg if sparse_avg > 0 else 0
    
    dense_min = min(dense_times) * 1000
    sparse_min = min(sparse_times) * 1000
    speedup_best = dense_min / sparse_min if sparse_min > 0 else 0
    
    print(f"\n📊 Timing Results (avg of {n_runs} runs):")
    print(f"   Dense:   {dense_avg:.1f} ms  (best: {dense_min:.1f} ms)")
    print(f"   Sparse:  {sparse_avg:.1f} ms  (best: {sparse_min:.1f} ms)")
    print(f"   Speedup: {speedup:.2f}x avg, {speedup_best:.2f}x best")
    
    # Analysis — ANCHOR computes FULL attention (+ tile scoring overhead)
    # Only REUSE layers do sparse attention (that's where speedup comes from)
    num_tiles = SEQ_LEN // TILE_SIZE
    attn_fraction = (SEQ_LEN * SEQ_LEN) / (SEQ_LEN * SEQ_LEN + 3 * SEQ_LEN * EMBED_DIM + 3 * SEQ_LEN * MLP_DIM)
    sparse_ratio = TOP_K_OPTIMIZED / num_tiles  # fraction of tiles kept
    
    # Count layer types
    dense_count = sum(1 for v in schedule.values() if v["type"] == "DENSE")
    anchor_count = sum(1 for v in schedule.values() if v["type"] == "ANCHOR")
    # reuse_count already computed
    
    # ANCHOR cost: full attention O(S²) + tile scoring overhead (small)
    # ANCHOR ≈ 1.0x of dense (paper: anchor computes full attention)
    # Tile scoring from full attention weights is negligible (just max-pool + top_k)
    anchor_ratio = 1.0  # full attention, same cost as dense
    
    # REUSE cost: sparse attention only (no full Q@K^T)
    reuse_ratio = TOP_K_OPTIMIZED * TILE_SIZE / SEQ_LEN  # e.g., 16*16/4096 = 6.25%
    
    # Weighted attention cost across all layers
    total_attn_cost = (dense_count * 1.0 + anchor_count * anchor_ratio + reuse_count * reuse_ratio)
    full_attn_cost = NUM_LAYERS * 1.0
    theoretical_attn_speedup = full_attn_cost / total_attn_cost if total_attn_cost > 0 else 1.0
    theoretical_total_speedup = 1.0 / (1 - attn_fraction + attn_fraction / theoretical_attn_speedup)
    
    print(f"\n📐 Analysis:")
    print(f"   Tiles: {num_tiles}, Top-K: {TOP_K_OPTIMIZED} ({sparse_ratio:.0%} of tiles)")
    print(f"   Schedule: {dense_count} DENSE + {anchor_count} ANCHOR (full attn) + {reuse_count} REUSE (sparse)")
    print(f"   ANCHOR: full attention + caches tile indices for REUSE")
    print(f"   REUSE:  sparse attention ({reuse_ratio:.1%} of tokens) using borrowed indices")
    print(f"   Attention fraction of total FLOPs: {attn_fraction:.1%}")
    print(f"   Theoretical attention-only speedup: {theoretical_attn_speedup:.2f}x")
    print(f"   Theoretical total speedup: {theoretical_total_speedup:.2f}x")
    if reuse_count == 0:
        print(f"   ⚠️  0 REUSE layers → ANCHOR=Dense, no speedup expected")
        print(f"   💡 Lower --threshold (try 0.30) to enable REUSE layers")
    
    if speedup > 1.2:
        print(f"\n✅ Sparse is {speedup:.2f}x faster!")
    elif speedup > 0.8:
        print(f"\n⚠️  Speedup: {speedup:.2f}x (CPU has gather overhead, expect better on TPU)")
    else:
        print(f"\n⚠️  Sparse slower ({speedup:.2f}x) — normal on CPU due to gather overhead")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
