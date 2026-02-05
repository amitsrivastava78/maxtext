"""
Test Kascade Custom Kernel
===========================
Simple tests to verify correctness and performance of the custom TPU kernel.
"""

import jax
import jax.numpy as jnp
import time
import sys
from pathlib import Path

# Direct imports without going through __init__.py
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import directly to avoid MaxText __init__.py dependencies
import importlib.util

kernel_path = Path(__file__).parent / "src" / "MaxText" / "kernels" / "kascade_kernel.py"
spec = importlib.util.spec_from_file_location("kascade_kernel", kernel_path)
kascade_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kascade_kernel)

kascade_attention_forward = kascade_kernel.kascade_attention_forward
kascade_attention_reference = kascade_kernel.kascade_attention_reference
make_kascade_kernel = kascade_kernel.make_kascade_kernel
KascadeBlockSizes = kascade_kernel.KascadeBlockSizes


def test_correctness_small():
    """Test kernel correctness with small inputs."""
    print("\n" + "="*60)
    print("Test 1: Correctness (Small - 128 seq_len)")
    print("="*60)
    
    # Test case (head_dim MUST be multiple of 128 for TPU)
    num_heads = 4
    q_seq_len = 256
    sparse_len = 128  # 50% sparsity
    head_dim = 128  # Must be multiple of 128 for TPU Pallas!
    
    # Random inputs
    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, 4)
    
    q = jax.random.normal(subkeys[0], (num_heads, q_seq_len, head_dim))
    k_sparse = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
    v_sparse = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))
    
    print(f"Q shape: {q.shape}")
    print(f"K_sparse shape: {k_sparse.shape}")
    print(f"V_sparse shape: {v_sparse.shape}")
    
    # Reference implementation
    print("\nComputing reference (JAX)...")
    out_ref = kascade_attention_reference(q, k_sparse, v_sparse)
    print(f"Reference output shape: {out_ref.shape}")
    print(f"Reference output range: [{out_ref.min():.4f}, {out_ref.max():.4f}]")
    
    # Custom kernel
    print("\nComputing with custom kernel...")
    block_sizes = KascadeBlockSizes(
        block_q=256,
        block_kv_sparse=128,
        block_kv_compute=128,
    )
    out_kernel = kascade_attention_forward(q, k_sparse, v_sparse, block_sizes)
    print(f"Kernel output shape: {out_kernel.shape}")
    print(f"Kernel output range: [{out_kernel.min():.4f}, {out_kernel.max():.4f}]")
    
    # Compare
    max_diff = jnp.abs(out_ref - out_kernel).max()
    mean_diff = jnp.abs(out_ref - out_kernel).mean()
    rel_error = max_diff / (jnp.abs(out_ref).max() + 1e-6)
    
    print(f"\nMax absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Relative error: {rel_error:.6e}")
    
    # Pass/fail
    threshold = 1e-3  # Tolerance for float32
    if max_diff < threshold:
        print(f"✅ PASS: Max diff {max_diff:.6e} < {threshold}")
        return True
    else:
        print(f"❌ FAIL: Max diff {max_diff:.6e} >= {threshold}")
        return False


def test_correctness_medium():
    """Test kernel correctness with medium inputs."""
    print("\n" + "="*60)
    print("Test 2: Correctness (Medium - 512 seq_len)")
    print("="*60)
    
    num_heads = 8
    q_seq_len = 512
    sparse_len = 128  # 25% sparsity
    head_dim = 128  # Must be multiple of 128 for TPU!
    
    key = jax.random.PRNGKey(123)
    key, *subkeys = jax.random.split(key, 4)
    
    q = jax.random.normal(subkeys[0], (num_heads, q_seq_len, head_dim))
    k_sparse = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
    v_sparse = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))
    
    print(f"Q shape: {q.shape}")
    print(f"K_sparse shape: {k_sparse.shape}")
    print(f"V_sparse shape: {v_sparse.shape}")
    
    # Reference
    out_ref = kascade_attention_reference(q, k_sparse, v_sparse)
    
    # Custom kernel
    block_sizes = KascadeBlockSizes(
        block_q=512,
        block_kv_sparse=128,
        block_kv_compute=128,
    )
    out_kernel = kascade_attention_forward(q, k_sparse, v_sparse, block_sizes)
    
    # Compare
    max_diff = jnp.abs(out_ref - out_kernel).max()
    mean_diff = jnp.abs(out_ref - out_kernel).mean()
    
    print(f"\nMax absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    
    threshold = 1e-3
    if max_diff < threshold:
        print(f"✅ PASS: Max diff {max_diff:.6e} < {threshold}")
        return True
    else:
        print(f"❌ FAIL: Max diff {max_diff:.6e} >= {threshold}")
        return False


def benchmark_kernel(num_warmup=3, num_runs=10):
    """Benchmark kernel vs reference implementation."""
    print("\n" + "="*60)
    print("Test 3: Performance Benchmark")
    print("="*60)
    
    # Realistic size
    num_heads = 32
    q_seq_len = 1024
    sparse_len = 256  # 25% sparsity
    head_dim = 128
    
    key = jax.random.PRNGKey(456)
    key, *subkeys = jax.random.split(key, 4)
    
    q = jax.random.normal(subkeys[0], (num_heads, q_seq_len, head_dim))
    k_sparse = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
    v_sparse = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))
    
    print(f"Input shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K_sparse: {k_sparse.shape}")
    print(f"  V_sparse: {v_sparse.shape}")
    print(f"Sparsity: {sparse_len / q_seq_len * 100:.1f}%")
    
    # Create JIT-compiled versions
    ref_fn = jax.jit(kascade_attention_reference)
    kernel_fn = make_kascade_kernel(
        KascadeBlockSizes(
            block_q=512,
            block_kv_sparse=256,
            block_kv_compute=128,
        )
    )
    
    # Warmup
    print(f"\nWarmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        _ = ref_fn(q, k_sparse, v_sparse).block_until_ready()
        _ = kernel_fn(q, k_sparse, v_sparse).block_until_ready()
    
    # Benchmark reference
    print(f"\nBenchmarking reference (JAX) - {num_runs} runs...")
    times_ref = []
    for i in range(num_runs):
        start = time.perf_counter()
        out_ref = ref_fn(q, k_sparse, v_sparse).block_until_ready()
        end = time.perf_counter()
        times_ref.append(end - start)
        if i % 2 == 0:
            print(f"  Run {i+1}/{num_runs}: {times_ref[-1]*1000:.2f} ms")
    
    # Benchmark kernel
    print(f"\nBenchmarking custom kernel - {num_runs} runs...")
    times_kernel = []
    for i in range(num_runs):
        start = time.perf_counter()
        out_kernel = kernel_fn(q, k_sparse, v_sparse).block_until_ready()
        end = time.perf_counter()
        times_kernel.append(end - start)
        if i % 2 == 0:
            print(f"  Run {i+1}/{num_runs}: {times_kernel[-1]*1000:.2f} ms")
    
    # Results
    mean_ref = jnp.mean(jnp.array(times_ref))
    std_ref = jnp.std(jnp.array(times_ref))
    mean_kernel = jnp.mean(jnp.array(times_kernel))
    std_kernel = jnp.std(jnp.array(times_kernel))
    speedup = mean_ref / mean_kernel
    
    print("\n" + "-"*60)
    print("RESULTS:")
    print(f"  Reference (JAX):  {mean_ref*1000:.2f} ± {std_ref*1000:.2f} ms")
    print(f"  Custom Kernel:    {mean_kernel*1000:.2f} ± {std_kernel*1000:.2f} ms")
    print(f"  Speedup:          {speedup:.2f}×")
    print("-"*60)
    
    # Verify correctness
    max_diff = jnp.abs(out_ref - out_kernel).max()
    print(f"Max difference: {max_diff:.6e}")
    
    if speedup > 1.0:
        print(f"✅ Kernel is {speedup:.2f}× faster!")
    else:
        print(f"⚠️  Kernel is {1/speedup:.2f}× slower (needs optimization)")
    
    return speedup


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" KASCADE CUSTOM KERNEL TEST SUITE")
    print("="*70)
    
    # Check device
    devices = jax.devices()
    print(f"\nDevices: {devices}")
    print(f"Default backend: {jax.default_backend()}")
    
    if jax.default_backend() != 'tpu':
        print("\n⚠️  WARNING: Not running on TPU! Kernel is optimized for TPU.")
        print("Results on CPU/GPU will not show expected speedups.")
    
    # Run tests
    results = []
    
    try:
        results.append(("Correctness (Small)", test_correctness_small()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Correctness (Small)", False))
    
    try:
        results.append(("Correctness (Medium)", test_correctness_medium()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Correctness (Medium)", False))
    
    try:
        speedup = benchmark_kernel()
        results.append(("Performance", speedup > 1.0))
    except Exception as e:
        print(f"❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Performance", False))
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} {name}")
    
    all_passed = all(result for _, result in results)
    print("="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
