#!/usr/bin/env python3
"""
Debug why Pallas kernel is slower than JAX reference.
Systematic investigation of performance bottlenecks.
"""

import sys
import os
import importlib.util

# Load kernel directly
kernel_path = os.path.join(os.path.dirname(__file__), 'src', 'MaxText', 'kernels', 'kascade_kernel.py')
spec = importlib.util.spec_from_file_location("kascade_kernel", kernel_path)
kascade_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kascade_kernel)

import jax
import jax.numpy as jnp
import time

kascade_attention_forward = kascade_kernel.kascade_attention_forward
KascadeBlockSizes = kascade_kernel.KascadeBlockSizes


def reference_attention_sparse(q, k_sparse, v_sparse):
    """Reference JAX implementation."""
    scores = jnp.einsum('hqd,hkd->hqk', q, k_sparse)
    scores = scores / jnp.sqrt(q.shape[-1])
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum('hqk,hkd->hqd', attn_weights, v_sparse)
    return output


def benchmark_config(q_shape, k_shape, v_shape, block_q, warmup=5, runs=20):
    """Benchmark a specific configuration."""
    num_heads, q_seq_len, head_dim = q_shape
    _, sparse_len, _ = k_shape
    
    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, 4)
    
    q = jax.random.normal(subkeys[0], q_shape)
    k_sparse = jax.random.normal(subkeys[1], k_shape)
    v_sparse = jax.random.normal(subkeys[2], v_shape)
    
    # JIT compile
    ref_fn = jax.jit(reference_attention_sparse)
    
    block_sizes = KascadeBlockSizes(
        block_q=block_q,
        block_kv_sparse=sparse_len,  # Use full sparse length
        block_kv_compute=min(128, sparse_len)
    )
    
    @jax.jit
    def kernel_fn(q, k, v):
        return kascade_attention_forward(q, k, v, block_sizes)
    
    # Warmup
    for _ in range(warmup):
        _ = ref_fn(q, k_sparse, v_sparse).block_until_ready()
        _ = kernel_fn(q, k_sparse, v_sparse).block_until_ready()
    
    # Benchmark reference
    times_ref = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = ref_fn(q, k_sparse, v_sparse).block_until_ready()
        times_ref.append(time.perf_counter() - start)
    
    # Benchmark kernel
    times_kernel = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = kernel_fn(q, k_sparse, v_sparse).block_until_ready()
        times_kernel.append(time.perf_counter() - start)
    
    median_ref = sorted(times_ref)[len(times_ref)//2]
    median_kernel = sorted(times_kernel)[len(times_kernel)//2]
    speedup = median_ref / median_kernel
    
    return speedup, median_ref * 1000, median_kernel * 1000


def main():
    print("="*80)
    print("PALLAS KERNEL PERFORMANCE INVESTIGATION")
    print("="*80)
    print(f"\nDevice: {jax.devices()[0]}")
    print(f"Backend: {jax.default_backend()}")
    
    if jax.default_backend() != "tpu":
        print("\n⚠️  WARNING: Not on TPU! Results won't be meaningful.")
        return
    
    print("\n" + "="*80)
    print("HYPOTHESIS 1: Block size sensitivity")
    print("="*80)
    print("\nTesting different block_q values with fixed input size:")
    print("Input: Q=(32, 1024, 128), K_sparse=(32, 256, 128)")
    
    q_shape = (32, 1024, 128)
    k_shape = (32, 256, 128)
    v_shape = (32, 256, 128)
    
    print(f"\n{'block_q':<10} {'JAX (ms)':<12} {'Kernel (ms)':<12} {'Speedup':<10}")
    print("-" * 50)
    
    for block_q in [128, 256, 512, 1024]:
        try:
            speedup, ref_time, kernel_time = benchmark_config(
                q_shape, k_shape, v_shape, block_q, warmup=3, runs=10
            )
            print(f"{block_q:<10} {ref_time:<12.3f} {kernel_time:<12.3f} {speedup:<10.3f}×")
        except Exception as e:
            print(f"{block_q:<10} ERROR: {str(e)[:40]}")
    
    print("\n" + "="*80)
    print("HYPOTHESIS 2: Input size sensitivity")
    print("="*80)
    print("\nTesting different input sizes with block_q=256:")
    
    print(f"\n{'Q shape':<25} {'K shape':<20} {'Speedup':<10}")
    print("-" * 60)
    
    configs = [
        # (num_heads, q_seq_len, sparse_len)
        (8, 512, 128),      # Small
        (16, 512, 128),     # Medium heads
        (32, 512, 256),     # Medium
        (32, 1024, 256),    # Current test
        (32, 2048, 256),    # Large Q
        (32, 1024, 512),    # Large K/V
    ]
    
    for num_heads, q_seq_len, sparse_len in configs:
        head_dim = 128
        q_shape = (num_heads, q_seq_len, head_dim)
        k_shape = (num_heads, sparse_len, head_dim)
        v_shape = (num_heads, sparse_len, head_dim)
        
        try:
            speedup, ref_time, kernel_time = benchmark_config(
                q_shape, k_shape, v_shape, block_q=256, warmup=3, runs=10
            )
            q_str = f"({num_heads}, {q_seq_len}, {head_dim})"
            k_str = f"({num_heads}, {sparse_len}, {head_dim})"
            print(f"{q_str:<25} {k_str:<20} {speedup:<10.3f}×")
        except Exception as e:
            print(f"{q_shape} ERROR: {str(e)[:30]}")
    
    print("\n" + "="*80)
    print("HYPOTHESIS 3: Compare against simple Pallas kernel")
    print("="*80)
    print("\nLet's see if ANY Pallas kernel is slow, or just ours...")
    print("Testing a minimal Pallas matmul vs JAX matmul:")
    
    try:
        import jax.experimental.pallas as pl
        
        def simple_matmul_kernel(x_ref, y_ref, o_ref):
            """Simplest possible Pallas matmul."""
            x = x_ref[...]
            y = y_ref[...]
            o_ref[...] = x @ y
        
        def pallas_matmul(x, y):
            """Pallas matmul wrapper."""
            return pl.pallas_call(
                simple_matmul_kernel,
                out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
            )(x, y)
        
        # Test
        x = jax.random.normal(jax.random.PRNGKey(0), (1024, 128))
        y = jax.random.normal(jax.random.PRNGKey(1), (128, 128))
        
        jax_fn = jax.jit(lambda x, y: x @ y)
        pallas_fn = jax.jit(pallas_matmul)
        
        # Warmup
        for _ in range(5):
            _ = jax_fn(x, y).block_until_ready()
            _ = pallas_fn(x, y).block_until_ready()
        
        # Benchmark
        times_jax = []
        times_pallas = []
        for _ in range(20):
            start = time.perf_counter()
            _ = jax_fn(x, y).block_until_ready()
            times_jax.append(time.perf_counter() - start)
            
            start = time.perf_counter()
            _ = pallas_fn(x, y).block_until_ready()
            times_pallas.append(time.perf_counter() - start)
        
        median_jax = sorted(times_jax)[len(times_jax)//2]
        median_pallas = sorted(times_pallas)[len(times_pallas)//2]
        speedup = median_jax / median_pallas
        
        print(f"\nSimple matmul (1024×128 @ 128×128):")
        print(f"  JAX:    {median_jax*1000:.3f} ms")
        print(f"  Pallas: {median_pallas*1000:.3f} ms")
        print(f"  Speedup: {speedup:.3f}×")
        
        if speedup < 0.9:
            print("\n⚠️  Even simple Pallas is slower! This suggests:")
            print("     - Pallas overhead is significant on this hardware")
            print("     - Or XLA is optimizing JAX operations very well")
            print("     - Or TPU configuration issue")
        else:
            print("\n✅ Simple Pallas is fast, so our kernel has specific issues")
            
    except Exception as e:
        print(f"\nCouldn't test simple Pallas: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*80)
    print("""
Based on results above:

1. If block size changes performance significantly:
   → Our memory access pattern is suboptimal
   → Need to profile memory bandwidth utilization
   
2. If all Pallas kernels are slow:
   → Pallas overhead dominates on this hardware
   → JAX's XLA optimization is just very good
   → Consider: Maybe custom kernels aren't needed?
   
3. If only certain input sizes are slow:
   → Block sizes don't divide evenly
   → Grid configuration issues
   → Cache thrashing at specific sizes
   
4. If simple Pallas is fast but ours is slow:
   → Our online softmax implementation has issues
   → Too many memory operations
   → Instruction scheduling problems

NEXT STEPS:
- Check XLA HLO output: jax.jit(fn).lower(args).as_text()
- Profile with TPU profiler
- Compare FLOP counts: theoretical vs actual
- Check memory bandwidth utilization
""")


if __name__ == "__main__":
    main()
