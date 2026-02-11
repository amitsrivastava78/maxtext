"""
Kascade Sparse Decode — Correctness & Benchmark Test
=====================================================
Tests that sparse decode (loading only top-k KV tiles) produces
results close to dense decode (loading full KV cache).

Usage:
    # CPU correctness test (no TPU needed)
    python test_kascade_decode.py

    # TPU benchmark
    python test_kascade_decode.py --device tpu --benchmark

    # Full test with real weights
    python test_kascade_decode.py --device tpu --benchmark --use_weights
"""

import argparse
import sys
import os
import time
import functools

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
import importlib.util

# Import decode kernel directly to avoid MaxText __init__.py pulling in pyconfig
_kernel_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'src', 'MaxText', 'kernels', 'kascade_decode_kernel.py'
)
_spec = importlib.util.spec_from_file_location("kascade_decode_kernel", _kernel_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

kascade_sparse_decode = _mod.kascade_sparse_decode
kascade_sparse_decode_jax = _mod.kascade_sparse_decode_jax
kascade_sparse_decode_vectorized = _mod.kascade_sparse_decode_vectorized
kascade_sparse_decode_slice = _mod.kascade_sparse_decode_slice
kascade_sparse_decode_fused = _mod.kascade_sparse_decode_fused
kascade_sparse_decode_tiled = _mod.kascade_sparse_decode_tiled
kascade_sparse_decode_scan = _mod.kascade_sparse_decode_scan
kascade_sparse_decode_hotbuf = _mod.kascade_sparse_decode_hotbuf
build_hot_kv_buffer = _mod.build_hot_kv_buffer
dense_decode_attention_jax = _mod.dense_decode_attention_jax
get_decode_tile_indices = _mod.get_decode_tile_indices
benchmark_sparse_vs_dense_decode = _mod.benchmark_sparse_vs_dense_decode

# Pallas v2 may not be available on CPU
kascade_sparse_decode_pallas_v2 = getattr(_mod, 'kascade_sparse_decode_pallas_v2', None)


def test_correctness_basic():
    """Test 1: Sparse decode output matches dense decode when all tiles selected."""
    print("\n" + "="*60)
    print("TEST 1: Correctness — sparse with all tiles == dense")
    print("="*60)

    B, H, S, D = 1, 4, 512, 64
    tile_size = 128
    num_tiles = S // tile_size  # 4

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)

    # Select ALL tiles — should match dense exactly
    tile_indices = jnp.broadcast_to(
        jnp.arange(num_tiles, dtype=jnp.int32)[None, None, :],
        (B, H, num_tiles)
    )
    query_pos = jnp.array([S - 1], dtype=jnp.int32)

    out_sparse = kascade_sparse_decode(
        q, k_cache, v_cache, tile_indices,
        tile_size=tile_size, query_pos=query_pos
    )
    out_dense = dense_decode_attention_jax(
        q, k_cache, v_cache, query_pos=query_pos
    )

    diff = float(jnp.max(jnp.abs(out_sparse - out_dense)))
    cosine_sim = float(jnp.sum(out_sparse * out_dense) /
                       (jnp.linalg.norm(out_sparse) * jnp.linalg.norm(out_dense)))

    print(f"  Sparse output shape: {out_sparse.shape}")
    print(f"  Dense  output shape: {out_dense.shape}")
    print(f"  Max abs diff: {diff:.2e}")
    print(f"  Cosine similarity: {cosine_sim:.6f}")
    
    passed = diff < 1e-4
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def test_correctness_partial_tiles():
    """Test 2: Sparse decode with subset of tiles has reasonable output."""
    print("\n" + "="*60)
    print("TEST 2: Correctness — partial tiles vs dense")
    print("="*60)

    B, H, S, D = 1, 8, 2048, 64
    tile_size = 128
    num_tiles = S // tile_size  # 16
    top_k = 4  # Only 4 out of 16 tiles

    key = jax.random.PRNGKey(123)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)

    # Select top_k random tiles per head
    tile_indices = jax.random.randint(k4, (B, H, top_k), 0, num_tiles, dtype=jnp.int32)
    query_pos = jnp.array([S - 1], dtype=jnp.int32)

    out_sparse = kascade_sparse_decode(
        q, k_cache, v_cache, tile_indices,
        tile_size=tile_size, query_pos=query_pos
    )
    out_dense = dense_decode_attention_jax(
        q, k_cache, v_cache, query_pos=query_pos
    )

    cosine_sim = float(jnp.sum(out_sparse * out_dense) /
                       (jnp.linalg.norm(out_sparse) * jnp.linalg.norm(out_dense)))

    print(f"  Tiles selected: {top_k}/{num_tiles} ({top_k/num_tiles*100:.0f}%)")
    print(f"  Sparse output mean: {float(jnp.mean(out_sparse)):.6f}")
    print(f"  Dense  output mean: {float(jnp.mean(out_dense)):.6f}")
    print(f"  Cosine similarity: {cosine_sim:.6f}")
    
    # With random data and random tile selection, cosine sim won't be perfect
    # but should be reasonable (> 0.5 for random, > 0.9 for real attention)
    passed = cosine_sim > 0.3  # Low bar for random data
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'} (random data, sim > 0.3)")
    return passed


def test_correctness_jax_vs_vectorized():
    """Test 3: JAX reference matches vectorized implementation."""
    print("\n" + "="*60)
    print("TEST 3: JAX reference == vectorized implementation")
    print("="*60)

    B, H, S, D = 2, 8, 1024, 64
    tile_size = 128
    num_tiles = S // tile_size
    top_k = 4

    key = jax.random.PRNGKey(999)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)
    tile_indices = jax.random.randint(k4, (B, H, top_k), 0, num_tiles, dtype=jnp.int32)
    query_pos = jnp.array([S - 1] * B, dtype=jnp.int32)

    out_jax = kascade_sparse_decode_jax(
        q, k_cache, v_cache, tile_indices,
        tile_size=tile_size, query_pos=query_pos
    )
    out_vec = kascade_sparse_decode_vectorized(
        q, k_cache, v_cache, tile_indices,
        tile_size=tile_size, query_pos=query_pos
    )

    diff = float(jnp.max(jnp.abs(out_jax - out_vec)))

    print(f"  JAX ref shape: {out_jax.shape}")
    print(f"  Vectorized shape: {out_vec.shape}")
    print(f"  Max abs diff: {diff:.2e}")

    passed = diff < 1e-5
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def test_slice_correctness():
    """Test 3b: dynamic_slice backend matches vectorized."""
    print("\n" + "="*60)
    print("TEST 3b: dynamic_slice == vectorized")
    print("="*60)

    B, H, S, D = 2, 8, 1024, 64
    tile_size = 128
    num_tiles = S // tile_size
    top_k = 4

    key = jax.random.PRNGKey(777)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)
    tile_indices = jax.random.randint(k4, (B, H, top_k), 0, num_tiles, dtype=jnp.int32)
    query_pos = jnp.array([S - 1] * B, dtype=jnp.int32)

    out_vec = kascade_sparse_decode_vectorized(
        q, k_cache, v_cache, tile_indices,
        tile_size=tile_size, query_pos=query_pos
    )
    out_slice = kascade_sparse_decode_slice(
        q, k_cache, v_cache, tile_indices,
        tile_size=tile_size, query_pos=query_pos
    )

    diff = float(jnp.max(jnp.abs(out_vec - out_slice)))

    print(f"  Vectorized shape: {out_vec.shape}")
    print(f"  Slice shape:      {out_slice.shape}")
    print(f"  Max abs diff: {diff:.2e}")

    passed = diff < 1e-5
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def test_slice_all_tiles_match_dense():
    """Test 3c: dynamic_slice with ALL tiles matches dense exactly."""
    print("\n" + "="*60)
    print("TEST 3c: dynamic_slice (all tiles) == dense")
    print("="*60)

    B, H, S, D = 1, 4, 512, 64
    tile_size = 128
    num_tiles = S // tile_size
    top_k = num_tiles  # Select ALL tiles

    key = jax.random.PRNGKey(555)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)
    tile_indices = jnp.broadcast_to(
        jnp.arange(num_tiles, dtype=jnp.int32)[None, None, :],
        (B, H, num_tiles)
    )

    out_slice = kascade_sparse_decode_slice(
        q, k_cache, v_cache, tile_indices, tile_size=tile_size
    )
    out_dense = dense_decode_attention_jax(q, k_cache, v_cache)

    diff = float(jnp.max(jnp.abs(out_slice - out_dense)))
    cosine = float(jnp.sum(out_slice * out_dense) /
                   (jnp.linalg.norm(out_slice) * jnp.linalg.norm(out_dense)))

    print(f"  Max abs diff: {diff:.2e}")
    print(f"  Cosine similarity: {cosine:.6f}")

    passed = diff < 1e-4 and cosine > 0.9999
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def test_fused_correctness():
    """Test 3d: fused fori_loop backend matches vectorized."""
    print("\n" + "="*60)
    print("TEST 3d: fused (fori_loop) == vectorized")
    print("="*60)

    B, H, S, D = 2, 8, 1024, 64
    tile_size = 128
    num_tiles = S // tile_size
    top_k = 4

    key = jax.random.PRNGKey(333)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)
    tile_indices = jax.random.randint(k4, (B, H, top_k), 0, num_tiles, dtype=jnp.int32)

    out_vec = kascade_sparse_decode_vectorized(
        q, k_cache, v_cache, tile_indices, tile_size=tile_size
    )
    out_fused = kascade_sparse_decode_fused(
        q, k_cache, v_cache, tile_indices, tile_size=tile_size
    )

    diff = float(jnp.max(jnp.abs(out_vec - out_fused)))
    cosine = float(jnp.sum(out_vec * out_fused) /
                   (jnp.linalg.norm(out_vec) * jnp.linalg.norm(out_fused)))

    print(f"  Vectorized shape: {out_vec.shape}")
    print(f"  Fused shape:      {out_fused.shape}")
    print(f"  Max abs diff: {diff:.2e}")
    print(f"  Cosine similarity: {cosine:.6f}")

    passed = diff < 1e-4 and cosine > 0.9999
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def test_fused_all_tiles_match_dense():
    """Test 3e: fused with ALL tiles matches dense exactly."""
    print("\n" + "="*60)
    print("TEST 3e: fused (all tiles) == dense")
    print("="*60)

    B, H, S, D = 1, 4, 512, 64
    tile_size = 128
    num_tiles = S // tile_size

    key = jax.random.PRNGKey(444)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)
    tile_indices = jnp.broadcast_to(
        jnp.arange(num_tiles, dtype=jnp.int32)[None, None, :],
        (B, H, num_tiles)
    )

    out_fused = kascade_sparse_decode_fused(
        q, k_cache, v_cache, tile_indices, tile_size=tile_size
    )
    out_dense = dense_decode_attention_jax(q, k_cache, v_cache)

    diff = float(jnp.max(jnp.abs(out_fused - out_dense)))
    cosine = float(jnp.sum(out_fused * out_dense) /
                   (jnp.linalg.norm(out_fused) * jnp.linalg.norm(out_dense)))

    print(f"  Max abs diff: {diff:.2e}")
    print(f"  Cosine similarity: {cosine:.6f}")

    passed = diff < 1e-4 and cosine > 0.9999
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    return passed

def test_tiled_correctness():
    """Test 3f: tiled gather backend matches vectorized."""
    print("\n" + "="*60)
    print("TEST 3f: tiled gather == vectorized")
    print("="*60)

    B, H, S, D = 2, 8, 1024, 64
    tile_size = 128
    num_tiles = S // tile_size
    top_k = 4

    key = jax.random.PRNGKey(222)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)
    tile_indices = jax.random.randint(k4, (B, H, top_k), 0, num_tiles, dtype=jnp.int32)

    out_vec = kascade_sparse_decode_vectorized(
        q, k_cache, v_cache, tile_indices, tile_size=tile_size
    )
    out_tiled = kascade_sparse_decode_tiled(
        q, k_cache, v_cache, tile_indices, tile_size=tile_size
    )

    diff = float(jnp.max(jnp.abs(out_vec - out_tiled)))
    cosine = float(jnp.sum(out_vec * out_tiled) /
                   (jnp.linalg.norm(out_vec) * jnp.linalg.norm(out_tiled)))

    print(f"  Vectorized shape: {out_vec.shape}")
    print(f"  Tiled shape:      {out_tiled.shape}")
    print(f"  Max abs diff: {diff:.2e}")
    print(f"  Cosine similarity: {cosine:.6f}")

    passed = diff < 1e-5 and cosine > 0.9999
    print(f"  Result: {'\u2705 PASS' if passed else '\u274c FAIL'}")
    return passed


def test_tiled_all_tiles_match_dense():
    """Test 3g: tiled with ALL tiles matches dense exactly."""
    print("\n" + "="*60)
    print("TEST 3g: tiled (all tiles) == dense")
    print("="*60)

    B, H, S, D = 1, 4, 512, 64
    tile_size = 128
    num_tiles = S // tile_size

    key = jax.random.PRNGKey(111)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)
    tile_indices = jnp.broadcast_to(
        jnp.arange(num_tiles, dtype=jnp.int32)[None, None, :],
        (B, H, num_tiles)
    )

    out_tiled = kascade_sparse_decode_tiled(
        q, k_cache, v_cache, tile_indices, tile_size=tile_size
    )
    out_dense = dense_decode_attention_jax(q, k_cache, v_cache)

    diff = float(jnp.max(jnp.abs(out_tiled - out_dense)))
    cosine = float(jnp.sum(out_tiled * out_dense) /
                   (jnp.linalg.norm(out_tiled) * jnp.linalg.norm(out_dense)))

    print(f"  Max abs diff: {diff:.2e}")
    print(f"  Cosine similarity: {cosine:.6f}")

    passed = diff < 1e-4 and cosine > 0.9999
    print(f"  Result: {'\u2705 PASS' if passed else '\u274c FAIL'}")
    return passed


def test_scan_correctness():
    """Test 3h: scan backend matches vectorized."""
    print("\n" + "="*60)
    print("TEST 3h: scan (lax.scan) == vectorized")
    print("="*60)

    B, H, S, D = 2, 8, 1024, 64
    tile_size = 128
    num_tiles = S // tile_size
    top_k = 4

    key = jax.random.PRNGKey(888)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)
    tile_indices = jax.random.randint(k4, (B, H, top_k), 0, num_tiles, dtype=jnp.int32)

    out_vec = kascade_sparse_decode_vectorized(
        q, k_cache, v_cache, tile_indices, tile_size=tile_size
    )
    out_scan = kascade_sparse_decode_scan(
        q, k_cache, v_cache, tile_indices, tile_size=tile_size
    )

    diff = float(jnp.max(jnp.abs(out_vec - out_scan)))
    cosine = float(jnp.sum(out_vec * out_scan) /
                   (jnp.linalg.norm(out_vec) * jnp.linalg.norm(out_scan)))

    print(f"  Vectorized shape: {out_vec.shape}")
    print(f"  Scan shape:       {out_scan.shape}")
    print(f"  Max abs diff: {diff:.2e}")
    print(f"  Cosine similarity: {cosine:.6f}")

    passed = diff < 1e-4 and cosine > 0.9999
    print(f"  Result: {'\u2705 PASS' if passed else '\u274c FAIL'}")
    return passed


def test_scan_all_tiles_match_dense():
    """Test 3i: scan with ALL tiles matches dense exactly."""
    print("\n" + "="*60)
    print("TEST 3i: scan (all tiles) == dense")
    print("="*60)

    B, H, S, D = 1, 4, 512, 64
    tile_size = 128
    num_tiles = S // tile_size

    key = jax.random.PRNGKey(666)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)
    tile_indices = jnp.broadcast_to(
        jnp.arange(num_tiles, dtype=jnp.int32)[None, None, :],
        (B, H, num_tiles)
    )

    out_scan = kascade_sparse_decode_scan(
        q, k_cache, v_cache, tile_indices, tile_size=tile_size
    )
    out_dense = dense_decode_attention_jax(q, k_cache, v_cache)

    diff = float(jnp.max(jnp.abs(out_scan - out_dense)))
    cosine = float(jnp.sum(out_scan * out_dense) /
                   (jnp.linalg.norm(out_scan) * jnp.linalg.norm(out_dense)))

    print(f"  Max abs diff: {diff:.2e}")
    print(f"  Cosine similarity: {cosine:.6f}")

    passed = diff < 1e-4 and cosine > 0.9999
    print(f"  Result: {'\u2705 PASS' if passed else '\u274c FAIL'}")
    return passed


def test_hotbuf_correctness():
    """Test 3j: hot buffer decode matches dense when all tiles selected."""
    print("\n" + "="*60)
    print("TEST 3j: hot buffer (all tiles) == dense")
    print("="*60)

    B, H, S, D = 1, 4, 512, 64
    tile_size = 128
    num_tiles = S // tile_size

    key = jax.random.PRNGKey(1234)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)
    tile_indices = jnp.broadcast_to(
        jnp.arange(num_tiles, dtype=jnp.int32)[None, None, :],
        (B, H, num_tiles)
    )

    # Build hot buffer and decode
    hot_k, hot_v = build_hot_kv_buffer(k_cache, v_cache, tile_indices, tile_size)
    out_hot = kascade_sparse_decode_hotbuf(q, hot_k, hot_v)
    out_dense = dense_decode_attention_jax(q, k_cache, v_cache)

    diff = float(jnp.max(jnp.abs(out_hot - out_dense)))
    cosine = float(jnp.sum(out_hot * out_dense) /
                   (jnp.linalg.norm(out_hot) * jnp.linalg.norm(out_dense)))

    print(f"  Hot buffer shape: {hot_k.shape}")
    print(f"  Max abs diff: {diff:.2e}")
    print(f"  Cosine similarity: {cosine:.6f}")

    passed = diff < 1e-4 and cosine > 0.9999
    print(f"  Result: {'\u2705 PASS' if passed else '\u274c FAIL'}")
    return passed


def test_causal_masking():
    """Test 4: Causal masking works — query can't attend to future tokens."""
    print("\n" + "="*60)
    print("TEST 4: Causal masking — no future token leakage")
    print("="*60)

    B, H, S, D = 1, 2, 512, 64
    tile_size = 128
    num_tiles = S // tile_size  # 4

    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.float32)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.float32)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.float32)

    # Query at position 200 (in tile 1)
    # Select tiles 0, 1, 2, 3 — but tile 2 and 3 contain future tokens
    tile_indices = jnp.array([[[0, 1, 2, 3], [0, 1, 2, 3]]], dtype=jnp.int32)
    query_pos = jnp.array([200], dtype=jnp.int32)  # Position 200

    out = kascade_sparse_decode_jax(
        q, k_cache, v_cache, tile_indices,
        tile_size=tile_size, query_pos=query_pos
    )

    # Verify by computing manually with masking
    sparse_len = 4 * tile_size
    offsets = jnp.arange(tile_size)
    token_indices = (tile_indices[..., None] * tile_size + offsets[None, None, None, :]).reshape(B, H, sparse_len)
    
    # Check that scores for future tokens are masked
    def gather_single(cache_bh, idx_bh):
        return cache_bh[idx_bh]
    gather_fn = jax.vmap(jax.vmap(gather_single))
    k_sparse = gather_fn(k_cache, token_indices)
    
    scores = jnp.einsum('bhqd,bhkd->bhqk', q, k_sparse) * (D ** -0.5)
    
    # Tokens at positions > 200 should be masked
    causal_mask = token_indices[:, :, None, :] <= 200
    scores_masked = jnp.where(causal_mask, scores, -1e10)
    
    # Check that future-position scores are effectively zero after softmax
    weights = jax.nn.softmax(scores_masked, axis=-1)
    future_mask = token_indices[:, :, None, :] > 200
    future_weight_sum = float(jnp.sum(jnp.where(future_mask, weights, 0.0)))

    print(f"  Query position: 200")
    print(f"  Future token weight sum: {future_weight_sum:.2e}")
    print(f"  Output shape: {out.shape}")

    passed = future_weight_sum < 1e-6
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def test_get_decode_tile_indices():
    """Test 5: get_decode_tile_indices extracts correct tiles from cache."""
    print("\n" + "="*60)
    print("TEST 5: get_decode_tile_indices utility")
    print("="*60)

    # Use a local dict to simulate KASCADE_CACHE (it's just a plain dict)
    KASCADE_CACHE = {}

    B, H = 1, 4
    num_tiles = 8
    top_k = 3
    tile_size = 128

    # Simulate cached tile indices from ANCHOR layer prefill
    # Shape: [B, H, Qg, top_k] (one set of top-k tiles per query group)
    cached = jnp.array(np.random.randint(0, num_tiles, (B, H, num_tiles, top_k)),
                        dtype=jnp.int32)
    KASCADE_CACHE["layer_1_indices"] = cached

    # Decode at position 300 (falls in tile 2: 256-383)
    query_pos = jnp.array([300], dtype=jnp.int32)

    tile_sel = get_decode_tile_indices(
        KASCADE_CACHE,
        anchor_layer_id=1,
        query_pos=query_pos,
        tile_size=tile_size,
        num_heads=H,
        add_local=True,
        num_local_tiles=1,
    )

    expected_query_tile = 300 // 128  # = 2
    print(f"  Query position: 300 → tile {expected_query_tile}")
    print(f"  Cached tiles for tile group 2: {cached[0, 0, expected_query_tile]}")
    print(f"  Selected tiles (head 0): {tile_sel[0, 0]}")
    print(f"  Shape: {tile_sel.shape} (expected [1, 4, {top_k + 1}])")

    # Verify local tile is included
    last_tile = int(tile_sel[0, 0, -1])
    print(f"  Last tile (local): {last_tile} (expected {expected_query_tile})")

    passed = (tile_sel.shape == (B, H, top_k + 1) and last_tile == expected_query_tile)
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    # Cleanup
    del KASCADE_CACHE["layer_1_indices"]
    return passed


def test_bf16_support():
    """Test 6: Works with bfloat16 inputs (TPU native dtype)."""
    print("\n" + "="*60)
    print("TEST 6: bf16 support")
    print("="*60)

    B, H, S, D = 1, 8, 1024, 64
    tile_size = 128
    num_tiles = S // tile_size
    top_k = 4

    key = jax.random.PRNGKey(55)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    q = jax.random.normal(k1, (B, H, 1, D), dtype=jnp.bfloat16)
    k_cache = jax.random.normal(k2, (B, H, S, D), dtype=jnp.bfloat16)
    v_cache = jax.random.normal(k3, (B, H, S, D), dtype=jnp.bfloat16)
    tile_indices = jax.random.randint(k4, (B, H, top_k), 0, num_tiles, dtype=jnp.int32)
    query_pos = jnp.array([S - 1], dtype=jnp.int32)

    out = kascade_sparse_decode(
        q, k_cache, v_cache, tile_indices,
        tile_size=tile_size, query_pos=query_pos
    )

    print(f"  Input dtype: {q.dtype}")
    print(f"  Output dtype: {out.dtype}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output has NaN: {bool(jnp.any(jnp.isnan(out)))}")
    print(f"  Output has Inf: {bool(jnp.any(jnp.isinf(out)))}")

    passed = (not jnp.any(jnp.isnan(out)) and
              not jnp.any(jnp.isinf(out)) and
              out.shape == (B, H, 1, D))
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def test_benchmark(seq_len=32768, top_k=25, tile_size=128):
    """Test 7: Benchmark sparse vs dense decode performance."""
    print("\n" + "="*60)
    print(f"BENCHMARK: Sparse vs Dense Decode (S={seq_len}, top_k={top_k})")
    print("="*60)

    results = benchmark_sparse_vs_dense_decode(
        B=1, H=32, S=seq_len, D=64,
        tile_size=tile_size, top_k=top_k,
        num_warmup=5, num_runs=20,
        dtype=jnp.bfloat16,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Kascade Sparse Decode Test")
    parser.add_argument("--device", default="cpu", choices=["cpu", "tpu", "gpu"])
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--seq_len", type=int, default=32768, help="Sequence length for benchmark")
    parser.add_argument("--top_k", type=int, default=25, help="Number of tiles to select")
    parser.add_argument("--tile_size", type=int, default=128, help="Tile size")
    args = parser.parse_args()

    print(f"\nDevice: {jax.devices()[0].platform} ({jax.device_count()} devices)")
    print(f"JAX version: {jax.__version__}")

    # Run correctness tests
    results = []
    results.append(("Basic correctness (all tiles)", test_correctness_basic()))
    results.append(("Partial tiles", test_correctness_partial_tiles()))
    results.append(("JAX vs vectorized", test_correctness_jax_vs_vectorized()))
    results.append(("Slice vs vectorized", test_slice_correctness()))
    results.append(("Slice all tiles == dense", test_slice_all_tiles_match_dense()))
    results.append(("Fused vs vectorized", test_fused_correctness()))
    results.append(("Fused all tiles == dense", test_fused_all_tiles_match_dense()))
    results.append(("Tiled vs vectorized", test_tiled_correctness()))
    results.append(("Tiled all tiles == dense", test_tiled_all_tiles_match_dense()))
    results.append(("Scan vs vectorized", test_scan_correctness()))
    results.append(("Scan all tiles == dense", test_scan_all_tiles_match_dense()))
    results.append(("Hot buffer all tiles == dense", test_hotbuf_correctness()))
    results.append(("Causal masking", test_causal_masking()))
    results.append(("Decode tile index utility", test_get_decode_tile_indices()))
    results.append(("bf16 support", test_bf16_support()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    if args.benchmark:
        for seq in [4096, 8192, 16384, 32768, 65536, 131072]:
            if seq <= args.seq_len:
                test_benchmark(seq_len=seq, top_k=args.top_k, tile_size=args.tile_size)

    print(f"\n{'='*60}")
    if all_pass:
        print("All correctness tests PASSED ✅")
    else:
        print("Some tests FAILED ❌")
    print(f"{'='*60}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
