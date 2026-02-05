#!/usr/bin/env python3
"""
Comprehensive block size tuning to find 2× speedup.
Tests all combinations of block_q, block_kv_sparse, block_kv_compute.
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
import itertools

kascade_attention_forward = kascade_kernel.kascade_attention_forward
KascadeBlockSizes = kascade_kernel.KascadeBlockSizes


def reference_attention_sparse(q, k_sparse, v_sparse):
    """Reference JAX implementation."""
    scores = jnp.einsum('hqd,hkd->hqk', q, k_sparse)
    scores = scores / jnp.sqrt(q.shape[-1])
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum('hqk,hkd->hqd', attn_weights, v_sparse)
    return output


def benchmark_config(q, k_sparse, v_sparse, block_q, block_kv_sparse, block_kv_compute, 
                     ref_fn, warmup=3, runs=10):
    """Benchmark a specific block size configuration."""
    try:
        block_sizes = KascadeBlockSizes(
            block_q=block_q,
            block_kv_sparse=block_kv_sparse,
            block_kv_compute=block_kv_compute
        )
        
        kernel_fn = jax.jit(lambda q, k, v: kascade_attention_forward(
            q, k, v, block_sizes=block_sizes
        ))
        
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
        
    except Exception as e:
        return None, None, None


def main():
    print("="*80)
    print("COMPREHENSIVE BLOCK SIZE TUNING FOR 2× SPEEDUP")
    print("="*80)
    
    # Check device
    devices = jax.devices()
    print(f"\nDevices: {devices}")
    print(f"Backend: {devices[0].platform}")
    
    if devices[0].platform != 'tpu':
        print("\n❌ ERROR: This benchmark requires TPU!")
        print("   Pallas kernels are TPU-specific and won't work on CPU/GPU")
        print("   Please run on a TPU VM")
        sys.exit(1)
    
    # Test configuration (same as test_phase1_standalone.py TEST 2)
    num_heads = 32
    q_seq_len = 1024
    sparse_len = 256
    head_dim = 128
    
    print(f"\nTest configuration:")
    print(f"  Q: ({num_heads}, {q_seq_len}, {head_dim})")
    print(f"  K_sparse: ({num_heads}, {sparse_len}, {head_dim})")
    print(f"  V_sparse: ({num_heads}, {sparse_len}, {head_dim})")
    
    # Generate test data
    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, 4)
    
    q = jax.random.normal(subkeys[0], (num_heads, q_seq_len, head_dim))
    k_sparse = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
    v_sparse = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))
    
    # JIT compile reference once
    ref_fn = jax.jit(reference_attention_sparse)
    _ = ref_fn(q, k_sparse, v_sparse).block_until_ready()
    
    print("\n" + "="*80)
    print("TESTING BLOCK SIZE COMBINATIONS")
    print("="*80)
    
    # Define search space (must be multiples of 128 for TPU)
    block_q_options = [256, 512, 1024, 2048]
    block_kv_sparse_options = [128, 256, 512]  # sparse_len=256, so try around this
    block_kv_compute_options = [64, 128, 256]
    
    results = []
    
    total_configs = len(block_q_options) * len(block_kv_sparse_options) * len(block_kv_compute_options)
    print(f"\nTesting {total_configs} configurations...")
    print(f"(This may take a few minutes)\n")
    
    best_speedup = 0
    best_config = None
    
    for i, (bq, bkv_s, bkv_c) in enumerate(itertools.product(
        block_q_options, block_kv_sparse_options, block_kv_compute_options
    ), 1):
        # Skip invalid configurations
        if bkv_c > bkv_s:
            continue
        if bkv_s % bkv_c != 0:
            continue
            
        print(f"[{i}] Testing block_q={bq}, block_kv_sparse={bkv_s}, block_kv_compute={bkv_c}...", 
              end='', flush=True)
        
        speedup, time_ref, time_kernel = benchmark_config(
            q, k_sparse, v_sparse, bq, bkv_s, bkv_c, ref_fn
        )
        
        if speedup is None:
            print(" FAILED")
            continue
        
        print(f" {speedup:.3f}×")
        
        results.append({
            'block_q': bq,
            'block_kv_sparse': bkv_s,
            'block_kv_compute': bkv_c,
            'speedup': speedup,
            'time_ref_ms': time_ref,
            'time_kernel_ms': time_kernel
        })
        
        if speedup > best_speedup:
            best_speedup = speedup
            best_config = (bq, bkv_s, bkv_c)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Sort by speedup
    results.sort(key=lambda x: x['speedup'], reverse=True)
    
    print("\nTop 10 configurations:")
    print(f"{'Rank':<6} {'block_q':<10} {'block_kv_s':<12} {'block_kv_c':<12} {'Speedup':<10} {'Ref(ms)':<10} {'Kernel(ms)'}")
    print("-" * 80)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:<6} {r['block_q']:<10} {r['block_kv_sparse']:<12} {r['block_kv_compute']:<12} "
              f"{r['speedup']:<10.3f} {r['time_ref_ms']:<10.3f} {r['time_kernel_ms']:.3f}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if best_speedup >= 2.0:
        print(f"\n🎉 SUCCESS! Found {best_speedup:.3f}× speedup")
        print(f"   Best config: block_q={best_config[0]}, block_kv_sparse={best_config[1]}, "
              f"block_kv_compute={best_config[2]}")
    elif best_speedup >= 1.2:
        print(f"\n✅ GOOD! Found {best_speedup:.3f}× speedup")
        print(f"   Best config: block_q={best_config[0]}, block_kv_sparse={best_config[1]}, "
              f"block_kv_compute={best_config[2]}")
        print(f"   This beats the 1.2× threshold!")
    elif best_speedup >= 1.0:
        print(f"\n⚠️  CLOSE! Found {best_speedup:.3f}× speedup")
        print(f"   Best config: block_q={best_config[0]}, block_kv_sparse={best_config[1]}, "
              f"block_kv_compute={best_config[2]}")
        print(f"   Still slightly faster than JAX, but below 1.2× target")
    else:
        print(f"\n❌ SLOWER! Best is only {best_speedup:.3f}×")
        print(f"   All configurations slower than JAX")
        print(f"   May need algorithmic changes, not just block size tuning")
    
    # Analyze patterns
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    # Group by block_q
    print("\nEffect of block_q:")
    for bq in block_q_options:
        configs = [r for r in results if r['block_q'] == bq]
        if configs:
            avg_speedup = sum(r['speedup'] for r in configs) / len(configs)
            best = max(configs, key=lambda x: x['speedup'])
            print(f"  block_q={bq:4d}: avg={avg_speedup:.3f}×, best={best['speedup']:.3f}×")
    
    # Group by block_kv_sparse
    print("\nEffect of block_kv_sparse:")
    for bkv_s in block_kv_sparse_options:
        configs = [r for r in results if r['block_kv_sparse'] == bkv_s]
        if configs:
            avg_speedup = sum(r['speedup'] for r in configs) / len(configs)
            best = max(configs, key=lambda x: x['speedup'])
            print(f"  block_kv_sparse={bkv_s:3d}: avg={avg_speedup:.3f}×, best={best['speedup']:.3f}×")
    
    # Group by block_kv_compute
    print("\nEffect of block_kv_compute:")
    for bkv_c in block_kv_compute_options:
        configs = [r for r in results if r['block_kv_compute'] == bkv_c]
        if configs:
            avg_speedup = sum(r['speedup'] for r in configs) / len(configs)
            best = max(configs, key=lambda x: x['speedup'])
            print(f"  block_kv_compute={bkv_c:3d}: avg={avg_speedup:.3f}×, best={best['speedup']:.3f}×")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if best_speedup < 1.2:
        print("""
Block size tuning alone won't reach 2× speedup. Consider:

1. ALGORITHM CHANGES:
   - Use Flash Attention style (fused kernel, no materialized attention matrix)
   - Implement online normalization differently
   - Reduce memory operations

2. COMPILER OPTIMIZATIONS:
   - Add @jax.jit with donate_argnums
   - Use static_argnums for constants
   - Try different XLA flags

3. MEMORY LAYOUT:
   - Change tensor layouts (e.g., transpose K)
   - Optimize scratch buffer access patterns
   - Use shared memory more efficiently

4. PROFILING:
   - Use JAX profiler to identify bottlenecks
   - Check memory bandwidth utilization
   - Compare FLOPs: theoretical vs actual

5. HYBRID APPROACH:
   - Keep JAX for small problems
   - Use custom kernel only where it helps
""")
    else:
        print(f"""
Update kascade_kernel.py with best configuration:

    block_q={best_config[0]}
    block_kv_sparse={best_config[1]}
    block_kv_compute={best_config[2]}

Then re-run test_phase1_standalone.py to validate!
""")


if __name__ == "__main__":
    main()
