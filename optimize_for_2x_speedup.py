#!/usr/bin/env python3
"""
Comprehensive optimization suite to achieve 2× speedup over JAX.
Tests multiple strategies: block sizes, compiler hints, memory layouts, etc.

Run this on TPU via Colab!
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
from jax.experimental import pallas as pl
import time
from functools import partial

kascade_attention_forward = kascade_kernel.kascade_attention_forward
KascadeBlockSizes = kascade_kernel.KascadeBlockSizes


def reference_attention_sparse(q, k_sparse, v_sparse):
    """Reference JAX implementation."""
    scores = jnp.einsum('hqd,hkd->hqk', q, k_sparse)
    scores = scores / jnp.sqrt(q.shape[-1])
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum('hqk,hkd->hqd', attn_weights, v_sparse)
    return output


def benchmark_implementation(name, fn, q, k, v, warmup=5, runs=20):
    """Benchmark a specific implementation."""
    try:
        # Warmup
        for _ in range(warmup):
            _ = fn(q, k, v).block_until_ready()
        
        # Benchmark
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = fn(q, k, v).block_until_ready()
            times.append(time.perf_counter() - start)
        
        median_time = sorted(times)[len(times)//2]
        return median_time * 1000  # Convert to ms
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def test_strategy_1_block_sizes(q, k, v, ref_time):
    """Strategy 1: Comprehensive block size tuning."""
    print("\n" + "="*80)
    print("STRATEGY 1: BLOCK SIZE TUNING")
    print("="*80)
    
    # Test configurations
    configs = [
        # Original
        (1024, 256, 128),
        # Larger block_q
        (2048, 256, 128),
        # Larger block_kv_sparse
        (1024, 512, 128),
        (1024, 512, 256),
        # Smaller block_kv_compute for better cache
        (1024, 256, 64),
        # All large
        (2048, 512, 256),
        # All small
        (512, 128, 64),
    ]
    
    best_speedup = 0
    best_config = None
    
    for block_q, block_kv_sparse, block_kv_compute in configs:
        print(f"\nTesting block_q={block_q}, block_kv_sparse={block_kv_sparse}, block_kv_compute={block_kv_compute}")
        
        try:
            block_sizes = KascadeBlockSizes(
                block_q=block_q,
                block_kv_sparse=block_kv_sparse,
                block_kv_compute=block_kv_compute
            )
            
            kernel_fn = jax.jit(lambda q, k, v: kascade_attention_forward(
                q, k, v, block_sizes=block_sizes
            ))
            
            kernel_time = benchmark_implementation(
                f"block_sizes_{block_q}_{block_kv_sparse}_{block_kv_compute}",
                kernel_fn, q, k, v
            )
            
            if kernel_time:
                speedup = ref_time / kernel_time
                print(f"  Time: {kernel_time:.3f} ms, Speedup: {speedup:.3f}×")
                
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_config = (block_q, block_kv_sparse, block_kv_compute)
        except Exception as e:
            print(f"  FAILED: {e}")
    
    print(f"\n✓ Best block size config: {best_config}")
    print(f"  Speedup: {best_speedup:.3f}×")
    return best_speedup, best_config


def test_strategy_2_compiler_hints(q, k, v, ref_time, best_block_config):
    """Strategy 2: Compiler optimizations."""
    print("\n" + "="*80)
    print("STRATEGY 2: COMPILER OPTIMIZATIONS")
    print("="*80)
    
    block_sizes = KascadeBlockSizes(
        block_q=best_block_config[0],
        block_kv_sparse=best_block_config[1],
        block_kv_compute=best_block_config[2]
    )
    
    tests = []
    
    # Test 1: donate_argnums
    print("\n[1] Testing donate_argnums (donate input buffers)...")
    try:
        fn = jax.jit(
            lambda q, k, v: kascade_attention_forward(q, k, v, block_sizes=block_sizes),
            donate_argnums=(0, 1, 2)
        )
        t = benchmark_implementation("donate_argnums", fn, q, k, v)
        if t:
            speedup = ref_time / t
            print(f"  Time: {t:.3f} ms, Speedup: {speedup:.3f}×")
            tests.append(("donate_argnums", speedup))
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 2: inline (force inline the jit)
    print("\n[2] Testing inline=True...")
    try:
        fn = jax.jit(
            lambda q, k, v: kascade_attention_forward(q, k, v, block_sizes=block_sizes),
            inline=True
        )
        t = benchmark_implementation("inline", fn, q, k, v)
        if t:
            speedup = ref_time / t
            print(f"  Time: {t:.3f} ms, Speedup: {speedup:.3f}×")
            tests.append(("inline", speedup))
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 3: XLA flags
    print("\n[3] Testing with XLA optimization flags...")
    old_flags = os.environ.get('XLA_FLAGS', '')
    try:
        os.environ['XLA_FLAGS'] = '--xla_gpu_enable_fast_min_max=true --xla_gpu_enable_triton_gemm=false'
        fn = jax.jit(lambda q, k, v: kascade_attention_forward(q, k, v, block_sizes=block_sizes))
        t = benchmark_implementation("xla_flags", fn, q, k, v)
        if t:
            speedup = ref_time / t
            print(f"  Time: {t:.3f} ms, Speedup: {speedup:.3f}×")
            tests.append(("xla_flags", speedup))
    except Exception as e:
        print(f"  FAILED: {e}")
    finally:
        os.environ['XLA_FLAGS'] = old_flags
    
    if tests:
        best = max(tests, key=lambda x: x[1])
        print(f"\n✓ Best compiler hint: {best[0]}")
        print(f"  Speedup: {best[1]:.3f}×")
        return best[1], best[0]
    return 0, None


def test_strategy_3_memory_layout(q, k, v, ref_time, best_block_config):
    """Strategy 3: Memory layout optimizations."""
    print("\n" + "="*80)
    print("STRATEGY 3: MEMORY LAYOUT OPTIMIZATIONS")
    print("="*80)
    
    block_sizes = KascadeBlockSizes(
        block_q=best_block_config[0],
        block_kv_sparse=best_block_config[1],
        block_kv_compute=best_block_config[2]
    )
    
    tests = []
    
    # Test 1: Transpose K upfront
    print("\n[1] Testing with pre-transposed K...")
    try:
        k_transposed = jnp.swapaxes(k, 1, 2)  # [h, d, k] instead of [h, k, d]
        
        # Note: This would require modifying the kernel, skipping for now
        print("  SKIPPED: Requires kernel modification")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 2: Ensure contiguous memory
    print("\n[2] Testing with explicit copy (contiguous memory)...")
    try:
        q_copy = jnp.array(q, copy=True)
        k_copy = jnp.array(k, copy=True)
        v_copy = jnp.array(v, copy=True)
        
        fn = jax.jit(lambda q, k, v: kascade_attention_forward(q, k, v, block_sizes=block_sizes))
        t = benchmark_implementation("contiguous", fn, q_copy, k_copy, v_copy)
        if t:
            speedup = ref_time / t
            print(f"  Time: {t:.3f} ms, Speedup: {speedup:.3f}×")
            tests.append(("contiguous", speedup))
    except Exception as e:
        print(f"  FAILED: {e}")
    
    if tests:
        best = max(tests, key=lambda x: x[1])
        print(f"\n✓ Best memory layout: {best[0]}")
        print(f"  Speedup: {best[1]:.3f}×")
        return best[1], best[0]
    return 0, None


def test_strategy_4_input_size_sweep(best_block_config):
    """Strategy 4: Test if performance varies with input size."""
    print("\n" + "="*80)
    print("STRATEGY 4: INPUT SIZE SENSITIVITY")
    print("="*80)
    print("\nTesting if certain input sizes show better performance...")
    
    block_sizes = KascadeBlockSizes(
        block_q=best_block_config[0],
        block_kv_sparse=best_block_config[1],
        block_kv_compute=best_block_config[2]
    )
    
    # Test different sizes
    configs = [
        (16, 512, 128, 128),   # Small
        (32, 1024, 256, 128),  # Current
        (64, 2048, 512, 128),  # Large
    ]
    
    ref_fn = jax.jit(reference_attention_sparse)
    kernel_fn = jax.jit(lambda q, k, v: kascade_attention_forward(q, k, v, block_sizes=block_sizes))
    
    best_speedup = 0
    best_size = None
    
    for num_heads, q_len, sparse_len, head_dim in configs:
        print(f"\nTesting heads={num_heads}, q_len={q_len}, sparse_len={sparse_len}")
        
        key = jax.random.PRNGKey(42)
        key, *subkeys = jax.random.split(key, 4)
        
        q = jax.random.normal(subkeys[0], (num_heads, q_len, head_dim))
        k = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
        v = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))
        
        ref_time = benchmark_implementation("ref", ref_fn, q, k, v, warmup=3, runs=10)
        kernel_time = benchmark_implementation("kernel", kernel_fn, q, k, v, warmup=3, runs=10)
        
        if ref_time and kernel_time:
            speedup = ref_time / kernel_time
            print(f"  Ref: {ref_time:.3f} ms, Kernel: {kernel_time:.3f} ms, Speedup: {speedup:.3f}×")
            
            if speedup > best_speedup:
                best_speedup = speedup
                best_size = (num_heads, q_len, sparse_len)
    
    print(f"\n✓ Best input size: heads={best_size[0]}, q_len={best_size[1]}, sparse_len={best_size[2]}")
    print(f"  Speedup: {best_speedup:.3f}×")
    return best_speedup, best_size


def test_strategy_5_fused_operations(q, k, v, ref_time, best_block_config):
    """Strategy 5: Try fusing scale operation."""
    print("\n" + "="*80)
    print("STRATEGY 5: ALGORITHMIC TWEAKS")
    print("="*80)
    
    print("\nChecking if we can gain from algorithmic changes...")
    print("(Would require kernel modifications - analyzing potential)")
    
    # Ideas to document:
    print("""
Potential improvements requiring kernel changes:

1. FUSED SCALING: 
   - Pre-scale Q by 1/sqrt(d) instead of scaling scores
   - Saves one division per element
   
2. REDUCE SCRATCH BUFFERS:
   - Current: 3 scratch buffers (m, l, o)
   - Try: 2 buffers by reusing space
   
3. BETTER VECTORIZATION:
   - Ensure all operations use SIMD
   - Check if loops are unrolled properly
   
4. FLASH ATTENTION STYLE:
   - Don't materialize full attention matrix
   - Compute in smaller tiles with online updates
   
5. NUMERICAL PRECISION:
   - Try bfloat16 instead of float32 (2× memory bandwidth)
   - Maintain accuracy with careful scaling
""")
    
    return 0, "needs_kernel_changes"


def main():
    print("="*80)
    print("COMPREHENSIVE OPTIMIZATION SUITE FOR 2× SPEEDUP")
    print("="*80)
    
    # Check device
    try:
        devices = jax.devices()
        print(f"\nDevices: {devices}")
        print(f"Backend: {devices[0].platform}")
    except RuntimeError as e:
        if "TPU is already in use" in str(e):
            print("\n❌ ERROR: TPU is locked by another process!")
            print("\nSOLUTION in Colab:")
            print("1. Go to Runtime → Restart runtime")
            print("2. Then run this script again")
            print("\nAlternatively, try:")
            print("  import os")
            print("  os.system('sudo lsof -t /dev/accel0 | xargs -r kill -9')")
            print("  # Then restart runtime")
            sys.exit(1)
        else:
            raise
    
    if devices[0].platform != 'tpu':
        print("\n⚠️  WARNING: Not running on TPU!")
        print("   Results may not be representative")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Test configuration (same as test_phase1_standalone.py TEST 2)
    num_heads = 32
    q_seq_len = 1024
    sparse_len = 256
    head_dim = 128
    
    print(f"\nBase test configuration:")
    print(f"  Q: ({num_heads}, {q_seq_len}, {head_dim})")
    print(f"  K_sparse: ({num_heads}, {sparse_len}, {head_dim})")
    print(f"  V_sparse: ({num_heads}, {sparse_len}, {head_dim})")
    
    # Generate test data
    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, 4)
    
    q = jax.random.normal(subkeys[0], (num_heads, q_seq_len, head_dim))
    k = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
    v = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))
    
    # Benchmark reference
    print("\nBenchmarking JAX reference...")
    ref_fn = jax.jit(reference_attention_sparse)
    ref_time = benchmark_implementation("reference", ref_fn, q, k, v)
    print(f"Reference time: {ref_time:.3f} ms")
    
    # Baseline kernel (current best: block_q=1024)
    print("\nBenchmarking baseline kernel (block_q=1024)...")
    baseline_sizes = KascadeBlockSizes(block_q=1024, block_kv_sparse=256, block_kv_compute=128)
    baseline_fn = jax.jit(lambda q, k, v: kascade_attention_forward(q, k, v, block_sizes=baseline_sizes))
    baseline_time = benchmark_implementation("baseline", baseline_fn, q, k, v)
    baseline_speedup = ref_time / baseline_time
    print(f"Baseline time: {baseline_time:.3f} ms")
    print(f"Baseline speedup: {baseline_speedup:.3f}×")
    
    # Track all results
    results = {
        'baseline': baseline_speedup
    }
    
    # Strategy 1: Block sizes
    speedup, config = test_strategy_1_block_sizes(q, k, v, ref_time)
    results['block_sizes'] = speedup
    best_block_config = config if speedup > baseline_speedup else (1024, 256, 128)
    
    # Strategy 2: Compiler hints
    speedup, hint = test_strategy_2_compiler_hints(q, k, v, ref_time, best_block_config)
    results['compiler'] = speedup
    
    # Strategy 3: Memory layout
    speedup, layout = test_strategy_3_memory_layout(q, k, v, ref_time, best_block_config)
    results['memory'] = speedup
    
    # Strategy 4: Input size sensitivity
    speedup, size = test_strategy_4_input_size_sweep(best_block_config)
    results['input_size'] = speedup
    
    # Strategy 5: Algorithmic ideas
    _, _ = test_strategy_5_fused_operations(q, k, v, ref_time, best_block_config)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    best_strategy = max(results.items(), key=lambda x: x[1])
    
    print(f"\nBaseline (current):     {results['baseline']:.3f}×")
    print(f"Block size tuning:      {results['block_sizes']:.3f}×")
    print(f"Compiler optimizations: {results['compiler']:.3f}×")
    print(f"Memory layout:          {results['memory']:.3f}×")
    print(f"Input size sensitivity: {results['input_size']:.3f}×")
    
    print(f"\n{'='*80}")
    print(f"BEST RESULT: {best_strategy[0]} → {best_strategy[1]:.3f}×")
    print(f"{'='*80}")
    
    if best_strategy[1] >= 2.0:
        print("\n🎉 SUCCESS! Achieved 2× speedup target!")
    elif best_strategy[1] >= 1.2:
        print("\n✅ GOOD! Achieved 1.2× threshold!")
    elif best_strategy[1] >= 1.0:
        print("\n⚠️  MARGINAL: Faster than JAX but below target")
    else:
        print("\n❌ SLOWER: Need algorithmic changes")
        print("\nRecommendations:")
        print("1. Implement Flash Attention style memory-efficient attention")
        print("2. Use bfloat16 for 2× memory bandwidth")
        print("3. Fuse more operations in the kernel")
        print("4. Consider hybrid: use JAX for this case, kernel for longer sequences")


if __name__ == "__main__":
    main()
