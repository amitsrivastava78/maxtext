#!/usr/bin/env python3
"""
Standalone Phase 1 test for TPU - no MaxText dependencies.
Directly imports the kernel module.

Usage:
  python test_phase1_standalone.py
"""

import sys
import os
import importlib.util

# Load kascade_kernel directly without going through MaxText __init__
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


def test_correctness_tpu():
    """Test that Phase 1 maintains correctness on TPU."""
    print("="*70)
    print("PHASE 1 CORRECTNESS TEST (TPU)")
    print("="*70)
    
    # Test configuration
    num_heads = 8
    q_seq_len = 512
    sparse_len = 128
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
    
    # Compute reference
    print("\nComputing reference (JAX)...")
    out_ref = reference_attention_sparse(q, k_sparse, v_sparse)
    out_ref.block_until_ready()
    
    # Compute with Phase 1 optimized kernel
    print("Computing with Phase 1 optimized kernel...")
    block_sizes = KascadeBlockSizes(
        block_q=512,
        block_kv_sparse=128,
        block_kv_compute=128
    )
    out_kernel = kascade_attention_forward(q, k_sparse, v_sparse, block_sizes)
    out_kernel.block_until_ready()
    
    # Compare
    max_diff = jnp.abs(out_ref - out_kernel).max()
    mean_diff = jnp.abs(out_ref - out_kernel).mean()
    
    print(f"\n{'='*70}")
    print("CORRECTNESS RESULTS:")
    print(f"{'='*70}")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    
    # Check
    threshold = 5e-3
    if max_diff < threshold:
        print(f"✅ PASS: Phase 1 maintains correctness on TPU")
        print(f"   Max diff {max_diff:.6e} < {threshold}")
        return True
    else:
        print(f"❌ FAIL: Correctness broken on TPU!")
        print(f"   Max diff {max_diff:.6e} >= {threshold}")
        
        # Debug
        print("\nSample outputs (first 5 values):")
        print(f"  Reference: {out_ref[0, 0, :5]}")
        print(f"  Kernel:    {out_kernel[0, 0, :5]}")
        return False


def benchmark_phase1_tpu(num_warmup=5, num_runs=20):
    """Benchmark Phase 3 performance on TPU."""
    print("\n" + "="*70)
    print("PHASE 3 PERFORMANCE BENCHMARK (TPU)")
    print("="*70)
    print("\nPhase 3 optimizations:")
    print("  - Hoist Q load outside loop (cache in registers)")
    print("  - Load K and V together (memory coalescing)")
    print("  - Fuse exp operations (better instruction scheduling)")
    print("  - Eliminate redundant type conversions")
    
    # Realistic size
    num_heads = 32
    q_seq_len = 1024
    sparse_len = 256
    head_dim = 128
    
    key = jax.random.PRNGKey(456)
    key, *subkeys = jax.random.split(key, 4)
    
    q = jax.random.normal(subkeys[0], (num_heads, q_seq_len, head_dim))
    k_sparse = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
    v_sparse = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))
    
    print(f"\nBenchmark configuration:")
    print(f"  Q: {q.shape}")
    print(f"  K_sparse: {k_sparse.shape}")
    print(f"  V_sparse: {v_sparse.shape}")
    print(f"  Sparsity: {100 * (1 - sparse_len / q_seq_len):.1f}%")
    print(f"  Warmup runs: {num_warmup}")
    print(f"  Benchmark runs: {num_runs}")
    
    # JIT compile both
    ref_fn = jax.jit(reference_attention_sparse)
    
    block_sizes = KascadeBlockSizes(
        block_q=512,
        block_kv_sparse=256,
        block_kv_compute=128
    )
    
    @jax.jit
    def kernel_fn(q, k, v):
        return kascade_attention_forward(q, k, v, block_sizes)
    
    # Warmup
    print(f"\nWarming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        _ = ref_fn(q, k_sparse, v_sparse).block_until_ready()
        _ = kernel_fn(q, k_sparse, v_sparse).block_until_ready()
    
    # Benchmark reference
    print("\nBenchmarking reference (JAX)...")
    times_ref = []
    for i in range(num_runs):
        start = time.perf_counter()
        _ = ref_fn(q, k_sparse, v_sparse).block_until_ready()
        elapsed = time.perf_counter() - start
        times_ref.append(elapsed)
        if (i + 1) % 5 == 0:
            print(f"  Run {i+1}/{num_runs}: {elapsed*1000:.2f} ms")
    
    mean_ref = sum(times_ref) / len(times_ref)
    
    # Benchmark Phase 1 kernel
    print("\nBenchmarking Phase 1 optimized kernel...")
    times_kernel = []
    for i in range(num_runs):
        start = time.perf_counter()
        _ = kernel_fn(q, k_sparse, v_sparse).block_until_ready()
        elapsed = time.perf_counter() - start
        times_kernel.append(elapsed)
        if (i + 1) % 5 == 0:
            print(f"  Run {i+1}/{num_runs}: {elapsed*1000:.2f} ms")
    
    mean_kernel = sum(times_kernel) / len(times_kernel)
    
    # Results
    speedup = mean_ref / mean_kernel
    
    # Calculate improvement over previous phases
    phase1_baseline = 0.785
    phase2_baseline = 0.911
    
    print(f"\n{'='*70}")
    print("PERFORMANCE RESULTS:")
    print(f"{'='*70}")
    print(f"Reference (JAX):        {mean_ref*1000:.3f} ms")
    print(f"Phase 3 Kernel:         {mean_kernel*1000:.3f} ms")
    print(f"Speedup:                {speedup:.3f}×")
    print(f"\nProgress:")
    print(f"  Phase 1 baseline:     {phase1_baseline:.3f}×")
    print(f"  Phase 2 baseline:     {phase2_baseline:.3f}×")
    print(f"  Phase 3 (current):    {speedup:.3f}×")
    print(f"  vs Phase 2:           {((speedup/phase2_baseline - 1)*100):+.1f}%")
    print(f"  vs Original (0.66×):  {((speedup/0.66 - 1)*100):+.1f}%")
    
    # Analysis
    if speedup >= 1.0:
        improvement = (speedup - 1.0) * 100
        print(f"\n🎉 SUCCESS: Kernel is {improvement:.1f}% faster than JAX!")
        if speedup >= 2.0:
            print(f"   EXCEEDED TARGET! (Goal: 2-3×, Got: {speedup:.2f}×)")
        elif speedup >= 1.5:
            print(f"   EXCELLENT! Close to 2× target")
        else:
            print(f"   Good progress, continue optimizing for 2-3× target")
    elif speedup >= 0.95:
        slowdown = (1.0 / speedup - 1.0) * 100
        print(f"\n✅ NEAR PARITY: Only {slowdown:.1f}% slower")
        print(f"   Almost there! Small tweaks needed")
    elif speedup >= 0.8:
        slowdown = (1.0 / speedup - 1.0) * 100
        print(f"\n⚠️  IMPROVED: Kernel is {slowdown:.1f}% slower (was 52% slower)")
        print(f"   Phase 1 helped! Was 0.66×, now {speedup:.2f}×")
        print(f"   Continue to Phase 2 (block tuning)")
    else:
        slowdown = (1.0 / speedup - 1.0) * 100
        print(f"\n❌ WORSE: Kernel is {slowdown:.1f}% slower")
        print(f"   Phase 1 didn't help as expected. Debug needed.")
    
    return speedup


def main():
    print("\n" + "="*70)
    print("PHASE 1 OPTIMIZATION VALIDATION ON TPU")
    print("="*70)
    
    # Check device
    devices = jax.devices()
    backend = jax.default_backend()
    
    print(f"\nDevices: {devices}")
    print(f"Backend: {backend}")
    
    if backend != "tpu":
        print(f"\n⚠️  WARNING: Running on {backend}, not TPU!")
        print("Results will not reflect TPU performance.")
    
    # Run tests
    print("\n" + "="*70)
    print("TEST 1: CORRECTNESS")
    print("="*70)
    correctness_pass = test_correctness_tpu()
    
    if not correctness_pass:
        print("\n❌ STOPPING: Correctness test failed!")
        print("Fix correctness issues before benchmarking performance.")
        return
    
    # Benchmark
    print("\n" + "="*70)
    print("TEST 2: PERFORMANCE")
    print("="*70)
    speedup = benchmark_phase1_tpu()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Correctness: {'✅ PASS' if correctness_pass else '❌ FAIL'}")
    print(f"Speedup:     {speedup:.3f}×")
    
    # Recommendations
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    
    if speedup >= 2.0:
        print("🎉 TARGET ACHIEVED! Kernel is 2× faster!")
        print("   → Mission accomplished!")
        print("   → Consider: Profile for potential 3× push")
    elif speedup >= 1.5:
        print("✅ Excellent progress - halfway to 2-3× target")
        print("   → Consider: Advanced optimizations")
        print("   → Tune: Block sizes for different input sizes")
        print("   → Profile: TPU utilization and memory bandwidth")
    elif speedup >= 1.0:
        print("✅ Faster than baseline! Great milestone")
        print("   → Need: Additional 50-100% for 2× target")
        print("   → Try: Operation fusion, better block sizes")
        print("   → Profile: Find remaining bottlenecks")
    elif speedup >= 0.95:
        print("⚠️  So close to breaking even!")
        print("   → Small tweaks should get past 1.0×")
        print("   → Then focus on reaching 2× target")
    else:
        print("⚠️  Phase 3 didn't provide expected gains")
        print("   → Debug: TPU utilization, memory patterns")
        print("   → Profile: Instruction-level analysis")
        print("   → Consider: Revisit algorithm approach")
    
    print("="*70)


if __name__ == "__main__":
    main()
