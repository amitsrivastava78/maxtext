#!/usr/bin/env python3
"""
Focused optimization to reach 1.2× speedup target.
Tests exact configuration from test_phase1_standalone.py TEST 2.
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
KascadeBlockSizes = kascade_kernel.KascadeBlockSizes


def reference_attention_sparse(q, k_sparse, v_sparse):
    """Reference JAX implementation."""
    scores = jnp.einsum('hqd,hkd->hqk', q, k_sparse)
    scores = scores / jnp.sqrt(q.shape[-1])
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum('hqk,hkd->hqd', attn_weights, v_sparse)
    return output


def benchmark(fn, q, k, v, warmup=5, runs=20):
    """Benchmark a function."""
    for _ in range(warmup):
        _ = fn(q, k, v).block_until_ready()
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = fn(q, k, v).block_until_ready()
        times.append(time.perf_counter() - start)
    
    times.sort()
    median_ms = times[len(times)//2] * 1000
    return median_ms


def main():
    print("="*80)
    print("FOCUSED OPTIMIZATION TO REACH 1.2× SPEEDUP")
    print("="*80)
    
    # Check device
    devices = jax.devices()
    if devices[0].platform != 'tpu':
        print("\n⚠️  Not on TPU! Results may not match.")
    print(f"\nDevices: {devices}")
    
    # EXACT configuration from test_phase1_standalone.py TEST 2
    num_heads = 32
    q_seq_len = 1024
    sparse_len = 256
    head_dim = 128
    
    print(f"\nTest configuration (matching validation):")
    print(f"  Q: ({num_heads}, {q_seq_len}, {head_dim})")
    print(f"  K_sparse: ({num_heads}, {sparse_len}, {head_dim})")
    print(f"  V_sparse: ({num_heads}, {sparse_len}, {head_dim})")
    
    # Generate data
    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, 4)
    
    q = jax.random.normal(subkeys[0], (num_heads, q_seq_len, head_dim))
    k = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
    v = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))
    
    # Benchmark reference
    print("\nBenchmarking JAX reference...")
    ref_fn = jax.jit(reference_attention_sparse)
    ref_time = benchmark(ref_fn, q, k, v)
    print(f"Reference: {ref_time:.3f} ms")
    
    # Current default
    print("\n" + "="*80)
    print("CURRENT DEFAULT (block_kv_compute=64)")
    print("="*80)
    
    default_sizes = KascadeBlockSizes(block_q=1024, block_kv_sparse=256, block_kv_compute=64)
    default_fn = jax.jit(lambda q, k, v: kascade_attention_forward(q, k, v, block_sizes=default_sizes))
    default_time = benchmark(default_fn, q, k, v)
    default_speedup = ref_time / default_time
    print(f"Time: {default_time:.3f} ms")
    print(f"Speedup: {default_speedup:.3f}×")
    
    # Try more aggressive configurations
    print("\n" + "="*80)
    print("TESTING ALTERNATIVE CONFIGURATIONS")
    print("="*80)
    
    configs = [
        # Vary all three parameters
        (512, 128, 32),
        (512, 128, 64),
        (512, 256, 32),
        (512, 256, 64),
        (1024, 128, 32),
        (1024, 128, 64),
        (1024, 256, 32),
        (1024, 256, 64),
        (2048, 256, 64),
        (2048, 256, 128),
        (2048, 512, 64),
        (2048, 512, 128),
        # Very small compute blocks
        (1024, 256, 16),
        (512, 128, 16),
        # All small
        (256, 128, 32),
        (256, 128, 64),
    ]
    
    results = []
    target_reached = False
    
    for block_q, block_kv_sparse, block_kv_compute in configs:
        # Skip invalid configs
        if block_kv_compute > block_kv_sparse:
            continue
        if block_kv_sparse > sparse_len:
            continue
        if block_kv_sparse % block_kv_compute != 0:
            continue
            
        try:
            sizes = KascadeBlockSizes(
                block_q=block_q,
                block_kv_sparse=block_kv_sparse,
                block_kv_compute=block_kv_compute
            )
            
            fn = jax.jit(lambda q, k, v: kascade_attention_forward(q, k, v, block_sizes=sizes))
            t = benchmark(fn, q, k, v, warmup=3, runs=10)
            speedup = ref_time / t
            
            results.append({
                'block_q': block_q,
                'block_kv_sparse': block_kv_sparse,
                'block_kv_compute': block_kv_compute,
                'time': t,
                'speedup': speedup
            })
            
            status = ""
            if speedup >= 1.2:
                status = " ✅ TARGET!"
                target_reached = True
            elif speedup >= 1.0:
                status = " ✓ Faster"
                
            print(f"  [{block_q:4d}, {block_kv_sparse:3d}, {block_kv_compute:3d}] "
                  f"→ {t:.3f} ms, {speedup:.3f}×{status}")
            
        except Exception as e:
            print(f"  [{block_q:4d}, {block_kv_sparse:3d}, {block_kv_compute:3d}] → FAILED")
    
    # Sort and display results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    results.sort(key=lambda x: x['speedup'], reverse=True)
    
    print("\nTop 10 configurations:")
    print(f"{'Rank':<6} {'block_q':<10} {'block_kv_s':<10} {'block_kv_c':<10} {'Time(ms)':<12} {'Speedup':<10} {'Status'}")
    print("-" * 80)
    
    for i, r in enumerate(results[:10], 1):
        status = "🎯 TARGET!" if r['speedup'] >= 1.2 else ("✅ FASTER" if r['speedup'] >= 1.0 else "")
        print(f"{i:<6} {r['block_q']:<10} {r['block_kv_sparse']:<10} {r['block_kv_compute']:<10} "
              f"{r['time']:<12.3f} {r['speedup']:<10.3f} {status}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if results:
        best = results[0]
        print(f"\nBest configuration:")
        print(f"  block_q={best['block_q']}")
        print(f"  block_kv_sparse={best['block_kv_sparse']}")
        print(f"  block_kv_compute={best['block_kv_compute']}")
        print(f"  Speedup: {best['speedup']:.3f}×")
        print(f"  Time: {best['time']:.3f} ms (ref: {ref_time:.3f} ms)")
        
        if best['speedup'] >= 1.2:
            print("\n🎉 SUCCESS! Achieved 1.2× target!")
            print(f"\nUpdate kascade_kernel.py with these values:")
            print(f"  block_q: int = {best['block_q']}")
            print(f"  block_kv_sparse: int = {best['block_kv_sparse']}")
            print(f"  block_kv_compute: int | None = {best['block_kv_compute']}")
        elif best['speedup'] >= 1.0:
            print(f"\n✅ FASTER than JAX! ({best['speedup']:.1%} speedup)")
            print(f"   Close to target. May reach 1.2× with:")
            print(f"   - Different JAX/libtpu versions")
            print(f"   - Larger batch sizes")
            print(f"   - Algorithm changes (Flash Attention style)")
        else:
            print(f"\n⚠️  Best is {best['speedup']:.3f}×, still below 1.0×")
            print(f"   Block size tuning alone won't reach 1.2×")
            print(f"   Need algorithmic improvements:")
            print(f"   - Flash Attention implementation")
            print(f"   - bfloat16 precision")
            print(f"   - More kernel fusion")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
