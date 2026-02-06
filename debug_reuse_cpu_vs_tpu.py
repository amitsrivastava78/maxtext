"""
Debug REUSE layer: CPU vs TPU comparison
Minimal test to find where divergence occurs
"""

import jax
import jax.numpy as jnp
import sys

# Import REUSE components
sys.path.insert(0, 'src')
from MaxText.layers.kascade_layers import (
    KascadeReuseAttention, 
    apply_rope, 
    precompute_freqs_cis,
    KASCADE_CACHE
)
from flax import linen as nn

# Configuration
BATCH = 1
SEQ_LEN = 64  # Small for debugging
NUM_HEADS = 8
HEAD_DIM = 128
TILE_SIZE = 16
NUM_TILES = SEQ_LEN // TILE_SIZE  # 4 tiles
TOP_K = 2  # Keep 2 out of 4 tiles

def run_test(device='cpu'):
    """Run REUSE test on specified device"""
    print(f"\n{'='*70}")
    print(f"Testing on {device.upper()}")
    print(f"{'='*70}")
    
    # Configure device
    jax.config.update('jax_platform_name', device)
    print(f"Device: {jax.devices()[0]}")
    
    # Fixed random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create input
    x = jax.random.normal(key, (BATCH, SEQ_LEN, NUM_HEADS * HEAD_DIM))
    print(f"\nInput shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input stats: mean={float(jnp.mean(x)):.6f}, std={float(jnp.std(x)):.6f}")
    
    # Simulate anchor indices (fixed for reproducibility)
    # Anchor selected tiles [0, 2] for all heads
    anchor_indices = jnp.array([[[0, 2]]] * NUM_HEADS).transpose(0, 2, 1)  # [1, 8, 2]
    print(f"\nAnchor indices shape: {anchor_indices.shape}")
    print(f"Anchor indices: {anchor_indices[0, 0, :]}")  # Show first head
    
    # Store in cache
    KASCADE_CACHE.clear()
    KASCADE_CACHE['layer_0_indices'] = anchor_indices
    
    # Create REUSE attention layer
    model = KascadeReuseAttention(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        anchor_layer_id=0,
        tile_size=TILE_SIZE,
        head_map=None,
        use_splash=False
    )
    
    # Initialize parameters
    params = model.init(key, x)
    
    # Create RoPE
    freq_cis = precompute_freqs_cis(HEAD_DIM, SEQ_LEN, theta=500000.0)
    print(f"\nRoPE freq_cis shape: {freq_cis.shape}, dtype: {freq_cis.dtype}")
    
    # Run forward pass
    print(f"\n{'='*70}")
    print("Running forward pass...")
    print(f"{'='*70}")
    
    output = model.apply(params, x, freq_cis=freq_cis)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output stats: mean={float(jnp.mean(output)):.6f}, std={float(jnp.std(output)):.6f}")
    print(f"Output range: [{float(jnp.min(output)):.6f}, {float(jnp.max(output)):.6f}]")
    
    # Check for NaN/Inf
    has_nan = jnp.any(jnp.isnan(output))
    has_inf = jnp.any(jnp.isinf(output))
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")
    
    return output, x, params

def main():
    print("\n" + "="*70)
    print("REUSE LAYER DEBUG: CPU vs TPU")
    print("="*70)
    
    # Test on CPU
    out_cpu, x_cpu, params_cpu = run_test('cpu')
    
    # Test on TPU
    out_tpu, x_tpu, params_tpu = run_test('tpu')
    
    # Compare outputs
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    # Input should be identical (same seed)
    input_diff = float(jnp.max(jnp.abs(x_cpu - x_tpu)))
    print(f"\nInput difference (should be ~0): {input_diff:.10f}")
    
    # Output difference
    output_diff = float(jnp.max(jnp.abs(out_cpu - out_tpu)))
    output_rel_diff = float(output_diff / (jnp.abs(out_cpu).mean() + 1e-8))
    
    print(f"\nOutput absolute difference: {output_diff:.10f}")
    print(f"Output relative difference: {output_rel_diff:.10f}")
    
    if output_rel_diff < 0.01:
        print("\n✅ SUCCESS: CPU and TPU outputs match (< 1% diff)")
    elif output_rel_diff < 0.1:
        print("\n⚠️  WARNING: Small difference between CPU and TPU (1-10% diff)")
    else:
        print("\n❌ FAILURE: Large difference between CPU and TPU (> 10% diff)")
        print("\nThis explains the perplexity degradation!")
        
        # Show sample values
        print(f"\nCPU output [0, 0, :5]: {out_cpu[0, 0, :5]}")
        print(f"TPU output [0, 0, :5]: {out_tpu[0, 0, :5]}")

if __name__ == "__main__":
    main()
