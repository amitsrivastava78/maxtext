"""
Debug script to understand kernel vs reference divergence
"""
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import importlib.util
kernel_path = Path(__file__).parent / "src" / "MaxText" / "kernels" / "kascade_kernel.py"
spec = importlib.util.spec_from_file_location("kascade_kernel", kernel_path)
kascade_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kascade_kernel)

kascade_attention_forward = kascade_kernel.kascade_attention_forward
kascade_attention_reference = kascade_kernel.kascade_attention_reference
KascadeBlockSizes = kascade_kernel.KascadeBlockSizes

print("="*70)
print("DEBUG: Manual computation vs kernel")
print("="*70)

# Small test case - must be multiples of 128!
num_heads = 2
q_seq_len = 256  # Changed to 256 (multiple of 128)
sparse_len = 128  # Already 128
head_dim = 128

key = jax.random.PRNGKey(42)
key, *subkeys = jax.random.split(key, 4)

q = jax.random.normal(subkeys[0], (num_heads, q_seq_len, head_dim))
k_sparse = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
v_sparse = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))

print(f"\nInput shapes:")
print(f"  Q: {q.shape}")
print(f"  K_sparse: {k_sparse.shape}")
print(f"  V_sparse: {v_sparse.shape}")

# Manual computation (what kernel should do)
print("\n" + "="*70)
print("MANUAL COMPUTATION (step by step)")
print("="*70)

# Pick first head, first query
h, i = 0, 0
q_single = q[h, i:i+1, :]  # [1, head_dim]
k_single = k_sparse[h, :, :]  # [sparse_len, head_dim]
v_single = v_sparse[h, :, :]  # [sparse_len, head_dim]

print(f"\nProcessing head={h}, query={i}")
print(f"  q_single: {q_single.shape}")
print(f"  k_single: {k_single.shape}")
print(f"  v_single: {v_single.shape}")

# Step 1: Q @ K^T
qk = jnp.dot(q_single, k_single.T)  # [1, sparse_len]
print(f"\n1. QK (before scaling): {qk.shape}")
print(f"   Values: {qk[0, :5]}")  # First 5 values

# Step 2: Scale
qk_scaled = qk / jnp.sqrt(float(head_dim))
print(f"\n2. QK (after scaling by sqrt({head_dim})):")
print(f"   Values: {qk_scaled[0, :5]}")

# Step 3: Softmax
m = qk_scaled.max()
exp_qk = jnp.exp(qk_scaled - m)
l = exp_qk.sum()
weights = exp_qk / l

print(f"\n3. Softmax computation:")
print(f"   max (m): {m:.6f}")
print(f"   exp values: {exp_qk[0, :5]}")
print(f"   sum (l): {l:.6f}")
print(f"   weights: {weights[0, :5]}")
print(f"   weights sum: {weights.sum():.6f} (should be 1.0)")

# Step 4: Weighted sum
output_manual = jnp.dot(weights, v_single)
print(f"\n4. Output @ V:")
print(f"   Shape: {output_manual.shape}")
print(f"   First 5 values: {output_manual[0, :5]}")

# Reference implementation
print("\n" + "="*70)
print("REFERENCE IMPLEMENTATION")
print("="*70)
out_ref = kascade_attention_reference(q, k_sparse, v_sparse)
print(f"Reference output shape: {out_ref.shape}")
print(f"Reference output [h={h}, i={i}, first 5]: {out_ref[h, i, :5]}")
print(f"Reference output range: [{out_ref.min():.4f}, {out_ref.max():.4f}]")

# Check if manual matches reference
manual_vs_ref_diff = jnp.abs(output_manual - out_ref[h:h+1, i:i+1, :]).max()
print(f"\nManual vs Reference diff: {manual_vs_ref_diff:.6e}")
if manual_vs_ref_diff < 1e-5:
    print("✅ Manual computation matches reference!")
else:
    print("❌ Manual computation differs from reference!")

# Kernel implementation
print("\n" + "="*70)
print("KERNEL IMPLEMENTATION")
print("="*70)
block_sizes = KascadeBlockSizes(
    block_q=256,  # Match q_seq_len
    block_kv_sparse=128,  # Match sparse_len
    block_kv_compute=128,  # Single iteration
)
out_kernel = kascade_attention_forward(q, k_sparse, v_sparse, block_sizes)
print(f"Kernel output shape: {out_kernel.shape}")
print(f"Kernel output [h={h}, i={i}, first 5]: {out_kernel[h, i, :5]}")
print(f"Kernel output range: [{out_kernel.min():.4f}, {out_kernel.max():.4f}]")

# Compare
kernel_vs_ref_diff = jnp.abs(out_kernel - out_ref).max()
print(f"\nKernel vs Reference diff: {kernel_vs_ref_diff:.6e}")
if kernel_vs_ref_diff < 1e-3:
    print("✅ Kernel matches reference!")
else:
    print("❌ Kernel differs from reference!")
    
    # More details
    print(f"\nDifference details:")
    print(f"  Max diff: {kernel_vs_ref_diff:.6e}")
    print(f"  Mean diff: {jnp.abs(out_kernel - out_ref).mean():.6e}")
    print(f"  Kernel / Reference ratio (first elem): {out_kernel[h, i, 0] / out_ref[h, i, 0]:.4f}")
    
    # Check if it's a constant scaling factor
    ratio = out_kernel / (out_ref + 1e-10)
    print(f"  Ratio statistics:")
    print(f"    Mean: {ratio.mean():.4f}")
    print(f"    Std: {ratio.std():.4f}")
    print(f"    Min: {ratio.min():.4f}")
    print(f"    Max: {ratio.max():.4f}")

print("\n" + "="*70)
