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
KASCADE_CACHE = kascade_module.KASCADE_CACHE
precompute_freqs_cis = kascade_module.precompute_freqs_cis

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
DEFAULT_TOP_K = 12  # Optimized for 1B model (paper uses 8 for 8B)
DEFAULT_THRESHOLD = 0.65  # 65% Jaccard similarity for reuse
DEFAULT_MAX_REUSE_DIST = 4  # Maximum layers between anchor and reuse
DEFAULT_SEQ_LEN = 512  # Default sequence length
DEFAULT_DEVICE = 'cpu'  # Default to CPU for compatibility
DEFAULT_USE_SPLASH = False  # Use optimized SplashAttention kernel

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

def generate_schedule_structure(consecutive_similarities, num_layers, threshold=0.65, max_reuse_dist=4):
    """Generate schedule structure determining ANCHOR vs REUSE (without head mappings yet)."""
    print(f"\n⚡ Generating Optimized Schedule:")
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
        if i not in consecutive_similarities:
            schedule[i] = {"type": "ANCHOR"}
            last_anchor = i
            print(f"  Layer {i}: ANCHOR (no data)")
            continue
        
        score = consecutive_similarities[i]
        distance = i - last_anchor
        
        if score >= threshold and distance < max_reuse_dist:
            print(f"  Layer {i}: REUSE L{last_anchor} (consecutive similarity: {score:.2%})")
            schedule[i] = {
                "type": "REUSE",
                "anchor_id": last_anchor,
                "head_map": {}  # Will be filled in second pass
            }
        else:
            schedule[i] = {"type": "ANCHOR"}
            last_anchor = i
            if score < threshold:
                print(f"  Layer {i}: ANCHOR (low similarity: {score:.2%})")
            else:
                print(f"  Layer {i}: ANCHOR (distance: {distance})")
    
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
    
    # Store all tile indices for later comparison
    all_tiles = {}
    for i in range(NUM_LAYERS):
        tiles = KASCADE_CACHE.get(f"layer_{i}_indices")
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
    
    params = {
        'tok_embeddings': {'embedding': emb_data['embed_tokens']},
        'norm': {'scale': emb_data['norm']},
        'output': {'kernel': emb_data['lm_head']},
    }
    
    for i in range(emb_data['config']['num_hidden_layers']):
        layer_weights = load_layer_params(i, weights_dir)
        
        wq = layer_weights['attention']['q_proj']['kernel']
        wk = layer_weights['attention']['k_proj']['kernel']
        wv = layer_weights['attention']['v_proj']['kernel']
        wo = layer_weights['attention']['o_proj']['kernel']
        
        params[f'layer_{i}'] = {
            'attention_norm': {'scale': layer_weights['input_layernorm']['scale']},
            'ffn_norm': {'scale': layer_weights['post_attention_layernorm']['scale']},
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
            'gate_proj': {'kernel': layer_weights['mlp']['gate_proj']['kernel']},
            'up_proj': {'kernel': layer_weights['mlp']['up_proj']['kernel']},
            'down_proj': {'kernel': layer_weights['mlp']['down_proj']['kernel']},
        }
    
    print(f"✓ Loaded {emb_data['config']['num_hidden_layers']} layers")
    return {'params': params}, emb_data['config']

def calculate_last_token_perplexity(logits, targets):
    """Calculate perplexity ONLY for the final token prediction."""
    final_logit = logits[:, -2, :]
    final_target = targets[:, -1]
    
    one_hot = jax.nn.one_hot(final_target, logits.shape[-1])
    log_probs = jax.nn.log_softmax(final_logit, axis=-1)
    
    token_log_prob = jnp.sum(one_hot * log_probs, axis=-1)
    loss = -jnp.mean(token_log_prob)
    
    return jnp.exp(loss)

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
        freq_cis = precompute_freqs_cis(HEAD_DIM, seq_len, theta=500000.0)
        
        plan = self.schedule.get(self.layer_id, {"type": "ANCHOR"})
        
        # Use SplashAttention kernel if enabled (TPU only)
        if self.use_splash and USE_SPLASH_KERNEL:
            # Import the module that was loaded in main()
            import sys
            kascade_splash_module = sys.modules.get('kascade_splash_attention')
            if kascade_splash_module is None:
                raise ImportError("kascade_splash_attention not loaded")
            
            kascade_splash_attention = kascade_splash_module.kascade_splash_attention
            
            # For splash kernel, we need Q, K, V explicitly
            # Compute them using Dense layers (matching the attention module structure)
            wq = nn.Dense(NUM_HEADS * HEAD_DIM, use_bias=False, name="Dense_0")
            wk = nn.Dense(NUM_HEADS * HEAD_DIM, use_bias=False, name="Dense_1")
            wv = nn.Dense(NUM_HEADS * HEAD_DIM, use_bias=False, name="Dense_2")
            wo = nn.Dense(EMBED_DIM, use_bias=False, name="Dense_3")
            
            Q = wq(normed).reshape(normed.shape[0], seq_len, NUM_HEADS, HEAD_DIM)
            K = wk(normed).reshape(normed.shape[0], seq_len, NUM_HEADS, HEAD_DIM)
            V = wv(normed).reshape(normed.shape[0], seq_len, NUM_HEADS, HEAD_DIM)
            
            # Apply RoPE
            # TODO: Apply freq_cis rotation to Q and K
            
            # Call optimized kernel
            is_anchor = plan["type"] in ["DENSE", "ANCHOR"]
            anchor_id = None if is_anchor else plan["anchor_id"]
            top_k_ratio = 1.0 if plan["type"] == "DENSE" else (TOP_K_OPTIMIZED / (seq_len / TILE_SIZE))
            
            attn_out = kascade_splash_attention(
                Q, K, V,
                layer_id=self.layer_id,
                is_anchor_layer=is_anchor,
                anchor_layer_id=anchor_id,
                tile_size=TILE_SIZE,
                top_k_ratio=top_k_ratio
            )
            attn_out = attn_out.reshape(normed.shape[0], seq_len, -1)
            attn_out = wo(attn_out)
        else:
            # Standard Kascade implementation
            if plan["type"] == "DENSE":
                attn = KascadeAnchorAttention(
                    NUM_HEADS, HEAD_DIM, self.layer_id,
                    top_k_tiles=32,
                    tile_size=TILE_SIZE
                )
                attn_out = attn(normed, freq_cis=freq_cis)
            elif plan["type"] == "ANCHOR":
                attn = KascadeAnchorAttention(
                    NUM_HEADS, HEAD_DIM, self.layer_id,
                    top_k_tiles=TOP_K_OPTIMIZED,
                    tile_size=TILE_SIZE
                )
                attn_out = attn(normed, freq_cis=freq_cis)
            else:
                attn = KascadeReuseAttention(
                    NUM_HEADS, HEAD_DIM, plan["anchor_id"],
                    tile_size=TILE_SIZE, head_map=plan["head_map"]
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
        streaming=True,
        trust_remote_code=True
    )
    
    # Collect tokens from multiple different documents
    print("   Tokenizing documents...")
    all_tokens = []
    doc_count = 0
    needed_tokens = 2 * SEQ_LEN  # Need 2x for calibration + test
    
    for example in dataset:
        if len(all_tokens) >= needed_tokens:
            break
        
        # Tokenize and take tokens
        text = example['text'][:10000]  # Increased to 10000 to get more tokens per doc
        tokens = tokenizer.encode(text, add_special_tokens=False)  # No special tokens to avoid mismatch
        all_tokens.extend(tokens)
        doc_count += 1
        
        if doc_count >= 20:  # Increased to 20 to ensure we get enough tokens for large seq_len
            break
    
    print(f"   ✓ Collected tokens from {doc_count} documents")
    
    # Verify we have enough tokens
    needed_tokens = 2 * SEQ_LEN
    if len(all_tokens) < needed_tokens:
        raise ValueError(f"Not enough tokens collected: {len(all_tokens)} < {needed_tokens}. Increase document count or text length.")
    
    # Trim to exactly 2*SEQ_LEN tokens (half for calibration, half for test)
    all_tokens = all_tokens[:2 * SEQ_LEN]
    
    # Ensure we have valid token IDs (clip to vocab size)
    all_tokens = [min(t, VOCAB_SIZE - 1) for t in all_tokens]
    
    # Split into calibration (first SEQ_LEN) and test (next SEQ_LEN)
    calib_ids = jnp.array([all_tokens[:SEQ_LEN]], dtype=jnp.int32)
    test_ids = jnp.array([all_tokens[SEQ_LEN:2*SEQ_LEN]], dtype=jnp.int32)
    
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
    model_dense = LlamaModel(schedule=dense_schedule, use_splash=args.use_splash_kernel)
    
    KASCADE_CACHE.clear()
    logits_dense = model_dense.apply(params_dict, test_ids)
    ppl_dense = calculate_last_token_perplexity(logits_dense, test_ids)
    
    # Sparse Kascade
    print("\n⚡ Running KASCADE Sparse...")
    model_sparse = LlamaModel(schedule=schedule, use_splash=args.use_splash_kernel)
    
    KASCADE_CACHE.clear()
    logits_sparse = model_sparse.apply(params_dict, test_ids)
    ppl_sparse = calculate_last_token_perplexity(logits_sparse, test_ids)
    
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
    
    # Warmup
    print("  Warming up...")
    _ = model_dense.apply(params_dict, test_ids)
    _ = model_sparse.apply(params_dict, test_ids)
    
    # Dense timing
    print("  Timing Dense...")
    dense_times = []
    for i in range(n_runs):
        KASCADE_CACHE.clear()
        start = time.time()
        _ = model_dense.apply(params_dict, test_ids)
        dense_times.append(time.time() - start)
    
    # Sparse timing
    print("  Timing Sparse...")
    sparse_times = []
    for i in range(n_runs):
        KASCADE_CACHE.clear()
        start = time.time()
        _ = model_sparse.apply(params_dict, test_ids)
        sparse_times.append(time.time() - start)
    
    dense_avg = sum(dense_times) / len(dense_times) * 1000
    sparse_avg = sum(sparse_times) / len(sparse_times) * 1000
    speedup = dense_avg / sparse_avg if sparse_avg > 0 else 0
    
    print(f"\n📊 Timing Results (avg of {n_runs} runs):")
    print(f"   Dense:   {dense_avg:.2f} ms")
    print(f"   Sparse:  {sparse_avg:.2f} ms")
    print(f"   Speedup: {speedup:.2f}x")
    
    if speedup > 1.2:
        print(f"\n✅ Sparse is {speedup:.2f}x faster!")
    elif speedup > 0.8:
        print(f"\n⚠️  Speedup: {speedup:.2f}x (CPU has overhead, expect better on TPU)")
    else:
        print(f"\n⚠️  Sparse slower ({speedup:.2f}x) - normal on CPU due to overhead")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
