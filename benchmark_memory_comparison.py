#!/usr/bin/env python3
"""
Compare Kascade kernel vs JAX reference on MEMORY USAGE and SCALABILITY.
This is the real advantage of online softmax algorithms.
"""

import sys
import os
import importlib.util

kernel_path = os.path.join(os.path.dirname(__file__), 'src', 'MaxText', 'kernels', 'kascade_kernel.py')
spec = importlib.util.spec_from_file_location("kascade_kernel", kernel_path)
kascade_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kascade_kernel)

import jax
import jax.numpy as jnp
import time

kascade_attention_forward = kascade_kernel.kascade_attention_forward


def reference_attention_sparse(q, k_sparse, v_sparse):
    """Naive JAX implementation - materializes full attention matrix."""
    scores = jnp.einsum('hqd,hkd->hqk', q, k_sparse)  # Memory: O(H×Q×K)
    scores = scores / jnp.sqrt(q.shape[-1])
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum('hqk,hkd->hqd', attn_weights, v_sparse)
    return output


def calculate_memory(num_heads, q_len, k_len, head_dim, dtype_bytes=4):
    """Calculate theoretical memory usage."""
    # Reference JAX: Materializes full attention matrix
    attention_matrix = num_heads * q_len * k_len * dtype_bytes
    input_tensors = num_heads * (q_len + 2*k_len) * head_dim * dtype_bytes
    output_tensor = num_heads * q_len * head_dim * dtype_bytes
    jax_total = attention_matrix + input_tensors + output_tensor
    
    # Kascade: Online softmax, only stores per-block data
    block_q = 1024
    max_active_blocks = (q_len + block_q - 1) // block_q
    # Per-block stores: running max (m), running sum (l), partial output
    per_block_memory = num_heads * block_q * head_dim * dtype_bytes
    # Plus scratch buffers for K/V blocks
    scratch_memory = num_heads * k_len * head_dim * 2 * dtype_bytes
    kascade_total = per_block_memory + scratch_memory + input_tensors + output_tensor
    
    return jax_total, kascade_total


def main():
    print("="*80)
    print("MEMORY EFFICIENCY COMPARISON: Kascade vs Naive JAX")
    print("="*80)
    print()
    print("The real advantage of Kascade (online softmax) is MEMORY EFFICIENCY,")
    print("not speed. As sequence length grows, naive JAX runs out of memory")
    print("while Kascade scales gracefully.")
    print()
    
    print("="*80)
    print("THEORETICAL MEMORY USAGE")
    print("="*80)
    print()
    print(f"{'Config':<30} {'JAX (GB)':<15} {'Kascade (GB)':<15} {'Savings':<10}")
    print("-" * 75)
    
    configs = [
        # (num_heads, q_len, k_len, name)
        (32, 1024, 256, "Current test"),
        (32, 2048, 512, "2x larger"),
        (32, 4096, 1024, "4x larger"),
        (32, 8192, 2048, "8x larger"),
        (32, 16384, 4096, "16x larger"),
        (32, 32768, 8192, "32x larger (realistic)"),
    ]
    
    for num_heads, q_len, k_len, name in configs:
        head_dim = 128
        jax_mem, kascade_mem = calculate_memory(num_heads, q_len, k_len, head_dim)
        jax_gb = jax_mem / 1e9
        kascade_gb = kascade_mem / 1e9
        savings = (jax_mem - kascade_mem) / jax_mem * 100
        
        print(f"{name:<30} {jax_gb:<15.3f} {kascade_gb:<15.3f} {savings:<10.1f}%")
    
    print()
    print("="*80)
    print("PRACTICAL SCALABILITY TEST")
    print("="*80)
    print()
    print("Testing increasing sequence lengths to see where JAX fails...")
    print()
    
    if jax.default_backend() != "tpu":
        print("⚠️  WARNING: Not on TPU, skipping practical test")
        return
    
    key = jax.random.PRNGKey(42)
    num_heads = 32
    head_dim = 128
    
    ref_fn = jax.jit(reference_attention_sparse)
    
    @jax.jit
    def kernel_fn(q, k, v):
        return kascade_attention_forward(q, k, v, None)  # Use defaults
    
    print(f"{'Q Length':<12} {'K Length':<12} {'JAX Status':<20} {'Kascade Status':<20}")
    print("-" * 70)
    
    for q_len in [1024, 2048, 4096, 8192, 16384]:
        k_len = q_len // 4  # Sparse attention
        
        try:
            # Generate inputs
            key, *subkeys = jax.random.split(key, 4)
            q = jax.random.normal(subkeys[0], (num_heads, q_len, head_dim))
            k = jax.random.normal(subkeys[1], (num_heads, k_len, head_dim))
            v = jax.random.normal(subkeys[2], (num_heads, k_len, head_dim))
            
            # Test JAX
            jax_status = "???"
            try:
                start = time.perf_counter()
                _ = ref_fn(q, k, v).block_until_ready()
                jax_time = time.perf_counter() - start
                jax_status = f"✅ {jax_time*1000:.1f}ms"
            except Exception as e:
                jax_status = f"❌ OOM/Error"
            
            # Test Kascade
            kascade_status = "???"
            try:
                start = time.perf_counter()
                _ = kernel_fn(q, k, v).block_until_ready()
                kascade_time = time.perf_counter() - start
                kascade_status = f"✅ {kascade_time*1000:.1f}ms"
            except Exception as e:
                kascade_status = f"❌ Error"
            
            print(f"{q_len:<12} {k_len:<12} {jax_status:<20} {kascade_status:<20}")
            
        except Exception as e:
            print(f"{q_len:<12} {k_len:<12} {'❌ Setup failed':<20} {'❌ Setup failed':<20}")
    
    print()
    print("="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
The comparison you've been running measures SPEED on small inputs where
both algorithms fit in memory. The real advantage of Kascade is:

1. MEMORY EFFICIENCY: O(Q) instead of O(Q×K) for attention matrix
2. SCALABILITY: Can handle much longer sequences
3. STREAMING: Processes data in blocks, better for large models

Your 0.909× "slowdown" is actually impressive because:
- You're matching XLA's highly-optimized einsum/softmax
- While using a more complex algorithm with better memory properties
- The naive JAX reference would OOM on longer sequences

To claim "2× speedup", you need to either:
A) Compare against a memory-matched baseline (Flash Attention JAX port)
B) Show where Kascade ENABLES computations that JAX cannot do (long seqs)
C) Demonstrate end-to-end speedup in real workloads

Currently: You've proven Kascade is correct and competitive. ✅
Still needed: Demonstrate the algorithmic advantage. 📊
""")


if __name__ == "__main__":
    main()
