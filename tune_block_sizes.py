#!/usr/bin/env python3
"""
Phase 2: Block Size Tuning
Find optimal block sizes for TPU performance.

Usage:
  python tune_block_sizes.py
"""

import sys
import os
import importlib.util
import time

# Load kascade_kernel directly
kernel_path = os.path.join(os.path.dirname(__file__), 'src', 'MaxText', 'kernels', 'kascade_kernel.py')
spec = importlib.util.spec_from_file_location("kascade_kernel", kernel_path)
kascade_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kascade_kernel)

import jax
import jax.numpy as jnp

kascade_attention_forward = kascade_kernel.kascade_attention_forward
KascadeBlockSizes = kascade_kernel.KascadeBlockSizes


def reference_attention_sparse(q, k_sparse, v_sparse):
    """Reference JAX implementation."""
    scores = jnp.einsum('hqd,hkd->hqk', q, k_sparse)
    scores = scores / jnp.sqrt(q.shape[-1])
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum('hqk,hkd->hqd', attn_weights, v_sparse)
    return output


def benchmark_block_config(q, k_sparse, v_sparse, block_sizes, num_runs=10):
    """Benchmark a specific block size configuration."""
    
    @jax.jit
    def kernel_fn(q, k, v):
        return kascade_attention_forward(q, k, v, block_sizes)
    
    # Warmup
    for _ in range(3):
        _ = kernel_fn(q, k_sparse, v_sparse).block_until_ready()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = kernel_fn(q, k_sparse, v_sparse).block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return sum(times) / len(times)


def test_correctness(q, k_sparse, v_sparse, block_sizes):
    """Quick correctness check."""
    out_ref = reference_attention_sparse(q, k_sparse, v_sparse)
    out_kernel = kascade_attention_forward(q, k_sparse, v_sparse, block_sizes)
    max_diff = jnp.abs(out_ref - out_kernel).max()
    return max_diff < 5e-3


def main():
    print("\n" + "="*70)
    print("PHASE 2: BLOCK SIZE TUNING")
    print("="*70)
    
    # Check device
    backend = jax.default_backend()
    print(f"\nBackend: {backend}")
    
    if backend != "tpu":
        print(f"⚠️  WARNING: Running on {backend}, not TPU!")
    
    # Test configuration
    num_heads = 32
    q_seq_len = 1024
    sparse_len = 256
    head_dim = 128
    
    key = jax.random.PRNGKey(456)
    key, *subkeys = jax.random.split(key, 4)
    
    q = jax.random.normal(subkeys[0], (num_heads, q_seq_len, head_dim))
    k_sparse = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
    v_sparse = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))
    
    print(f"\nTest configuration:")
    print(f"  Q: {q.shape}")
    print(f"  K_sparse: {k_sparse.shape}")
    print(f"  V_sparse: {v_sparse.shape}")
    print(f"  Sparsity: {100 * (1 - sparse_len / q_seq_len):.1f}%")
    
    # Benchmark reference
    print("\nBenchmarking reference...")
    ref_fn = jax.jit(reference_attention_sparse)
    for _ in range(3):
        _ = ref_fn(q, k_sparse, v_sparse).block_until_ready()
    
    times_ref = []
    for _ in range(10):
        start = time.perf_counter()
        _ = ref_fn(q, k_sparse, v_sparse).block_until_ready()
        times_ref.append(time.perf_counter() - start)
    
    ref_time = sum(times_ref) / len(times_ref)
    print(f"Reference time: {ref_time*1000:.3f} ms")
    
    # Block size configurations to test
    # Format: (block_q, block_kv_sparse, block_kv_compute)
    configs = [
        # Current (Phase 1)
        (512, 256, 128),
        
        # Smaller blocks (better cache utilization)
        (256, 256, 128),
        (256, 128, 128),
        (512, 128, 128),
        
        # Larger blocks (more parallelism)
        (1024, 256, 128),
        (512, 256, 256),
        
        # Balanced
        (384, 256, 128),
        (512, 192, 128),
    ]
    
    print("\n" + "="*70)
    print("TESTING BLOCK SIZE CONFIGURATIONS")
    print("="*70)
    print(f"\n{'Config':<30} {'Time (ms)':<12} {'Speedup':<10} {'Status'}")
    print("-" * 70)
    
    results = []
    
    for block_q, block_kv_sparse, block_kv_compute in configs:
        config_name = f"({block_q}, {block_kv_sparse}, {block_kv_compute})"
        
        try:
            block_sizes = KascadeBlockSizes(
                block_q=block_q,
                block_kv_sparse=block_kv_sparse,
                block_kv_compute=block_kv_compute
            )
            
            # Quick correctness check
            if not test_correctness(q, k_sparse, v_sparse, block_sizes):
                print(f"{config_name:<30} {'N/A':<12} {'N/A':<10} ❌ INCORRECT")
                continue
            
            # Benchmark
            kernel_time = benchmark_block_config(q, k_sparse, v_sparse, block_sizes)
            speedup = ref_time / kernel_time
            
            status = "✅" if speedup >= 1.0 else "⚠️"
            print(f"{config_name:<30} {kernel_time*1000:<12.3f} {speedup:<10.3f}× {status}")
            
            results.append({
                'config': (block_q, block_kv_sparse, block_kv_compute),
                'time': kernel_time,
                'speedup': speedup
            })
            
        except Exception as e:
            print(f"{config_name:<30} {'ERROR':<12} {'N/A':<10} ❌ {str(e)[:30]}")
    
    # Find best configuration
    if results:
        best = max(results, key=lambda x: x['speedup'])
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nReference time:     {ref_time*1000:.3f} ms")
        print(f"\nBest configuration: {best['config']}")
        print(f"  Time:             {best['time']*1000:.3f} ms")
        print(f"  Speedup:          {best['speedup']:.3f}×")
        
        # Calculate improvement over Phase 1
        phase1_speedup = 0.785  # From previous test
        improvement = (best['speedup'] / phase1_speedup - 1) * 100
        
        print(f"\n  vs Phase 1:       {improvement:+.1f}% improvement")
        
        if best['speedup'] >= 1.0:
            print(f"\n🎉 SUCCESS! Kernel is now faster than JAX baseline!")
            print(f"   Ready for Phase 3 (prefetching) to reach 2-3× target")
        elif best['speedup'] >= 0.9:
            print(f"\n✅ CLOSE! Almost at baseline parity")
            print(f"   Phase 3 (prefetching) should get us well past 1.0×")
        else:
            print(f"\n⚠️  Still slower, but improved")
            print(f"   Continue tuning or move to Phase 3 for bigger gains")
        
        # Show top 3
        top3 = sorted(results, key=lambda x: x['speedup'], reverse=True)[:3]
        print(f"\nTop 3 configurations:")
        for i, result in enumerate(top3, 1):
            print(f"  {i}. {result['config']}: {result['speedup']:.3f}×")
        
        # Recommendation
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        
        if best['speedup'] >= phase1_speedup * 1.1:
            print(f"\n✅ Update kascade_kernel.py with best config: {best['config']}")
            print(f"   Then proceed to Phase 3 (prefetching)")
        else:
            print(f"\n⚠️  Block tuning gave minimal improvement")
            print(f"   Recommend proceeding directly to Phase 3 (prefetching)")
            print(f"   That's where the big performance gains will come from")
    
    print("="*70)


if __name__ == "__main__":
    main()
