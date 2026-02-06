"""
Debug REUSE layer: CPU vs TPU comparison
Minimal test to find where divergence occurs
"""

import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())

# Configuration
BATCH = 1
SEQ_LEN = 64
NUM_HEADS = 8
HEAD_DIM = 128
TILE_SIZE = 16

def apply_rope_simple(xq, xk, freq_cis):
    """Simplified RoPE without Flax dependencies"""
    xq = xq.astype(jnp.float32)
    xk = xk.astype(jnp.float32)
    
    xq_pairs = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_pairs = xk.reshape(*xk.shape[:-1], -1, 2)
    
    xq_complex = jax.lax.complex(xq_pairs[..., 0], xq_pairs[..., 1])
    xk_complex = jax.lax.complex(xk_pairs[..., 0], xk_pairs[..., 1])
    
    freq_cis = freq_cis[None, None, :xq.shape[2], :]
    
    xq_rotated = xq_complex * freq_cis
    xk_rotated = xk_complex * freq_cis
    
    xq_out = jnp.stack([xq_rotated.real, xq_rotated.imag], axis=-1).reshape(xq.shape)
    xk_out = jnp.stack([xk_rotated.real, xk_rotated.imag], axis=-1).reshape(xk.shape)
    
    return xq_out, xk_out

def reuse_attention_minimal(q, k, v, tile_indices):
    """Minimal REUSE attention without Flax"""
    batch, num_heads, seq_len, head_dim = q.shape
    
    # Expand tile indices to token indices
    offsets = jnp.arange(TILE_SIZE)[None, None, None, :]
    tile_starts = tile_indices[..., None] * TILE_SIZE
    token_indices = tile_starts + offsets
    flat_indices = token_indices.reshape(batch, num_heads, -1)
    flat_indices = jnp.clip(flat_indices, 0, seq_len - 1)
    
    # Gather sparse K, V
    k_sparse = jnp.take_along_axis(k, flat_indices[..., None], axis=2)
    v_sparse = jnp.take_along_axis(v, flat_indices[..., None], axis=2)
    
    # Compute attention
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k_sparse) / jnp.sqrt(head_dim)
    
    # Causal mask
    query_idx = jnp.arange(seq_len)[None, None, :, None]
    key_idx = flat_indices[:, :, None, :]
    future_mask = (key_idx > query_idx)
    logits = jnp.where(future_mask, -1e10, logits)
    
    # Softmax and output
    weights = jax.nn.softmax(logits, axis=-1)
    weights = jnp.where(jnp.isnan(weights), 0.0, weights)
    
    output = jnp.einsum('bhqk,bhkd->bhqd', weights, v_sparse)
    return output



def run_test(device='cpu'):
    """Run REUSE test on specified device"""
    print(f"\n{'='*70}")
    print(f"Testing on {device.upper()}")
    print(f"{'='*70}")
    
    jax.config.update('jax_platform_name', device)
    print(f"Device: {jax.devices()[0]}")
    
    key = jax.random.PRNGKey(42)
    
    # Create Q, K, V directly
    q = jax.random.normal(key, (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
    k = jax.random.normal(jax.random.split(key)[0], (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
    v = jax.random.normal(jax.random.split(key)[1], (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
    
    print(f"\nQ shape: {q.shape}, dtype: {q.dtype}")
    print(f"Q stats: mean={float(jnp.mean(q)):.6f}, std={float(jnp.std(q)):.6f}")
    
    # Apply RoPE
    freqs = 1.0 / (500000.0 ** (jnp.arange(0, HEAD_DIM, 2).astype(jnp.float32) / HEAD_DIM))
    t = jnp.arange(SEQ_LEN, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    freq_cis = jnp.exp(1j * freqs)
    
    q_rope, k_rope = apply_rope_simple(q, k, freq_cis)
    print(f"\nAfter RoPE - Q dtype: {q_rope.dtype}")
    print(f"Q_rope stats: mean={float(jnp.mean(q_rope)):.6f}, std={float(jnp.std(q_rope)):.6f}")
    
    # Tile indices (keep tiles 0 and 2)
    tile_indices = jnp.array([[[0, 2]]] * NUM_HEADS).transpose(0, 2, 1)
    print(f"\nTile indices: {tile_indices[0, 0, :]}")
    
    # Run REUSE attention
    output = reuse_attention_minimal(q_rope, k_rope, v, tile_indices)
    
    print(f"\nOutput shape: {output.shape}, dtype: {output.dtype}")
    print(f"Output stats: mean={float(jnp.mean(output)):.6f}, std={float(jnp.std(output)):.6f}")
    print(f"Output range: [{float(jnp.min(output)):.6f}, {float(jnp.max(output)):.6f}]")
    
    has_nan = jnp.any(jnp.isnan(output))
    has_inf = jnp.any(jnp.isinf(output))
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")
    
    return output, q, k, v

def main():
    print("\n" + "="*70)
    print("REUSE LAYER DEBUG: CPU vs TPU")
    print("="*70)
    
    out_cpu, q_cpu, k_cpu, v_cpu = run_test('cpu')
    out_tpu, q_tpu, k_tpu, v_tpu = run_test('tpu')
    
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    input_diff = float(jnp.max(jnp.abs(q_cpu - q_tpu)))
    print(f"\nInput Q difference (should be ~0): {input_diff:.10f}")
    
    output_diff = float(jnp.max(jnp.abs(out_cpu - out_tpu)))
    output_rel_diff = float(output_diff / (jnp.abs(out_cpu).mean() + 1e-8))
    
    print(f"\nOutput absolute difference: {output_diff:.10f}")
    print(f"Output relative difference: {output_rel_diff:.10f}")
    
    if output_rel_diff < 0.01:
        print("\n✅ SUCCESS: CPU and TPU outputs match (< 1% diff)")
    elif output_rel_diff < 0.1:
        print("\n⚠️  WARNING: Small difference (1-10% diff)")
    else:
        print("\n❌ FAILURE: Large difference (> 10% diff)")
        print(f"\nCPU output sample: {out_cpu[0, 0, 0, :5]}")
        print(f"TPU output sample: {out_tpu[0, 0, 0, :5]}")

if __name__ == "__main__":
    main()
