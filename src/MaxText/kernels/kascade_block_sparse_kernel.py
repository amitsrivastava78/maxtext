"""
Kascade Block-Sparse Attention Kernel (Long-Sequence Edition)
===============================================================
Provides two sparse attention backends:

1. block_sparse_attention_jax: Masked 6D einsum (CPU/GPU fallback, correct but no speedup)
2. splash_sparse_attention: Tokamax SplashAttention with dynamic grid (TPU)
   - Actual block skipping via active_rows/active_cols grid scheduling
   - TILE_SIZE=128 matches TPU hardware blocks
   - Union mask across heads (Tokamax limitation: single 2D mask)
   - Expected >10% total model speedup at seq_len >= 32K

Usage:
    from kascade_block_sparse_kernel import splash_sparse_attention
    # block_mask: [B, H, Qg, Kg] boolean from anchor layer
    output = splash_sparse_attention(q, k, v, block_mask, tile_size=128, num_heads=32)
"""

import functools
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


# ============================================================
# Tokamax SplashAttention Import (TPU only)
# ============================================================

TOKAMAX_SPLASH_AVAILABLE = False
_tokamax_make_dynamic_splash_mha = None
_tokamax_SplashConfig = None

try:
    from tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_kernel import (
        make_dynamic_splash_mha as _tokamax_make_dynamic_splash_mha,
        SplashConfig as _tokamax_SplashConfig,
    )
    TOKAMAX_SPLASH_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# Block Mask Creation (shared by all backends)
# ============================================================

def create_block_mask_from_tile_indices(
    tile_indices,   # [B, H, num_q_tiles, top_k]
    num_kv_tiles,
    causal=True,
):
    """Convert tile indices to a dense block mask.
    
    Args:
        tile_indices: [B, H, Qg, top_k]
        num_kv_tiles: Total number of KV tiles
        causal: If True, mask future KV tiles
    Returns:
        block_mask: [B, H, Qg, num_kv_tiles] boolean
    """
    B, H, Qg, top_k = tile_indices.shape
    mask = jnp.zeros((B, H, Qg, num_kv_tiles), dtype=jnp.bool_)
    b_idx = jnp.arange(B)[:, None, None, None]
    h_idx = jnp.arange(H)[None, :, None, None]
    q_idx = jnp.arange(Qg)[None, None, :, None]
    safe_indices = jnp.clip(tile_indices, 0, num_kv_tiles - 1)
    mask = mask.at[b_idx, h_idx, q_idx, safe_indices].set(True)
    if causal:
        causal_mask = jnp.arange(num_kv_tiles)[None, None, None, :] <= jnp.arange(Qg)[None, None, :, None]
        mask = mask & causal_mask
    return mask


# ============================================================
# Backend 1: JAX Block-Sparse (CPU/GPU fallback)
# ============================================================

def block_sparse_attention_jax(q, k, v, block_mask, tile_size=128):
    """Block-sparse attention using JAX ops.
    
    Computes Q@K^T for ALL blocks, applies block_mask, softmax over selected.
    No actual compute savings -- used as correctness reference and CPU fallback.
    """
    B, H, S, D = q.shape
    num_tiles = S // tile_size
    q_tiles = q.reshape(B, H, num_tiles, tile_size, D)
    k_tiles = k.reshape(B, H, num_tiles, tile_size, D)
    v_tiles = v.reshape(B, H, num_tiles, tile_size, D)
    logits = jnp.einsum('bhqid,bhkjd->bhqikj', q_tiles, k_tiles) / jnp.sqrt(D)
    block_mask_expanded = block_mask[:, :, :, None, :, None]
    logits = jnp.where(block_mask_expanded, logits, DEFAULT_MASK_VALUE)
    q_pos = (jnp.arange(num_tiles)[:, None] * tile_size + jnp.arange(tile_size)[None, :])
    k_pos = (jnp.arange(num_tiles)[:, None] * tile_size + jnp.arange(tile_size)[None, :])
    causal = q_pos[:, :, None, None] >= k_pos[None, None, :, :]
    logits = jnp.where(causal[None, None], logits, DEFAULT_MASK_VALUE)
    logits_flat = logits.reshape(B, H, num_tiles, tile_size, num_tiles * tile_size)
    weights_flat = jax.nn.softmax(logits_flat, axis=-1)
    all_masked = jnp.all(logits_flat <= DEFAULT_MASK_VALUE + 1, axis=-1, keepdims=True)
    weights_flat = jnp.where(all_masked, 0.0, weights_flat)
    weights = weights_flat.reshape(B, H, num_tiles, tile_size, num_tiles, tile_size)
    output = jnp.einsum('bhqikj,bhkjd->bhqid', weights, v_tiles)
    output = output.reshape(B, H, S, D)
    return output


# ============================================================
# Backend 2: Tokamax SplashAttention (TPU — real block skipping)
# ============================================================

def _block_mask_to_2d_union(block_mask, tile_size):
    """Convert per-head block mask to a single 2D token-level mask.
    
    Tokamax SplashAttention takes a single [q_seq, kv_seq] mask shared
    across all heads. We create a union (OR) of all per-head block masks,
    then expand from block-level to token-level.
    
    Args:
        block_mask: [B, H, Qg, Kg] boolean
        tile_size: block size (Kascade tile size, typically 128)
    Returns:
        mask_2d: [q_seq, kv_seq] boolean (jax.Array)
    """
    # Union across heads and batch: if ANY head in ANY batch needs the block, keep it
    union_mask = jnp.any(block_mask, axis=(0, 1))  # [Qg, Kg]
    
    # Apply causal constraint at block level first (cheap: [Qg, Kg])
    Qg, Kg = union_mask.shape
    block_causal = jnp.tril(jnp.ones((Qg, Kg), dtype=jnp.bool_))
    union_mask = union_mask & block_causal
    
    # Construct token-level mask on CPU to avoid TPU HBM pressure.
    # PROBLEM: Fancy indexing on TPU creates int32[S,S] gather indices = 4GB OOM.
    # SOLUTION: Use NumPy on host CPU (12+ GB RAM), transfer final 1GB bool mask.
    seq_len = Qg * tile_size
    union_np = np.asarray(union_mask)  # [Qg, Kg] → CPU
    
    # Expand blocks to tokens: repeat each block entry tile_size times per axis
    mask_np = np.repeat(np.repeat(union_np, tile_size, axis=0), tile_size, axis=1)
    # Apply token-level causal (tril zeros upper triangle, in-place for memory)
    mask_np = np.tril(mask_np).astype(np.bool_)
    
    # Transfer to TPU (1GB for 32K seq_len)
    mask_2d = jax.device_put(jnp.array(mask_np))
    del mask_np  # Free CPU memory immediately
    
    return mask_2d


# ============================================================
# Cache for compiled splash kernels (avoid per-layer recompilation)
# ============================================================
_FULL_CAUSAL_SPLASH_CACHE = {}
_SPARSE_SPLASH_CACHE = {}


def full_causal_splash_attention(q, k, v):
    """Dense causal attention via Tokamax SplashAttention (flash attention on TPU).
    
    Replacement for jax.nn.dot_product_attention which is broken on TPU in
    JAX 0.9.0.1 (produces random outputs regardless of implementation param).
    Uses a full lower-triangular causal mask (all blocks active = standard attention).
    
    The kernel is cached by seq_len so compilation happens only once.
    
    Args:
        q, k, v: [B, H, S, D]
    Returns:
        output: [B, H, S, D]
    """
    if not TOKAMAX_SPLASH_AVAILABLE:
        raise RuntimeError("Tokamax not available for full_causal_splash_attention")
    
    B, H, S, D = q.shape
    
    # Choose splash block size
    splash_block = 256
    if S < splash_block:
        splash_block = max(64, S)
    if S % splash_block != 0:
        for sb in [256, 128, 64]:
            if S % sb == 0:
                splash_block = sb
                break
    
    cache_key = ('full_causal', S, splash_block)
    if cache_key not in _FULL_CAUSAL_SPLASH_CACHE:
        # Build full causal mask on CPU to avoid TPU HBM pressure
        num_blocks = S // splash_block
        block_mask_np = np.tril(np.ones((num_blocks, num_blocks), dtype=np.bool_))
        mask_np = np.repeat(np.repeat(block_mask_np, splash_block, axis=0),
                            splash_block, axis=1)
        mask_np = np.tril(mask_np)  # Token-level causal within diagonal blocks
        mask_2d = jax.device_put(jnp.array(mask_np))
        del mask_np, block_mask_np
        
        config = _tokamax_SplashConfig(
            block_q=splash_block,
            block_kv=splash_block,
            block_kv_compute=min(splash_block, 128),
        )
        kernel = _tokamax_make_dynamic_splash_mha(mask=mask_2d, config=config)
        _FULL_CAUSAL_SPLASH_CACHE[cache_key] = kernel
        del mask_2d
    
    kernel = _FULL_CAUSAL_SPLASH_CACHE[cache_key]
    
    orig_dtype = q.dtype
    # SplashAttention computes raw Q@K^T (no 1/sqrt(D) scaling internally).
    # Pre-scale Q so logits = (Q/sqrt(D)) @ K^T = Q@K^T / sqrt(D).
    scale = jnp.float32(D) ** -0.5
    q = (q * scale).astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    
    outputs = []
    for b in range(B):
        out_b = kernel(q[b], k[b], v[b])
        outputs.append(out_b)
    
    output = jnp.stack(outputs, axis=0).astype(orig_dtype)
    return output


def splash_sparse_attention(q, k, v, block_mask, tile_size=128, num_heads=32):
    """Sparse attention via Tokamax SplashAttention with dynamic grid.
    
    On TPU: Uses make_dynamic_splash_mha for actual block skipping.
    On CPU: Falls back to block_sparse_attention_jax.
    
    Args:
        q, k, v: [B, H, S, D] — already RoPE'd
        block_mask: [B, H, Qg, Kg] boolean — from anchor's cached block mask
        tile_size: Must be 128 (TPU hardware block size)
        num_heads: Number of attention heads
    Returns:
        output: [B, H, S, D]
    """
    B, H, S, D = q.shape
    platform = jax.devices()[0].platform
    
    # Only use Tokamax on TPU when available
    if platform != 'tpu' or not TOKAMAX_SPLASH_AVAILABLE:
        return block_sparse_attention_jax(q, k, v, block_mask, tile_size)
    
    # --- Tokamax SplashAttention path ---
    
    # Cast to bf16: reduces HBM for Q/K/V (~768MB → ~384MB at 32K)
    orig_dtype = q.dtype
    # SplashAttention computes raw Q@K^T (no 1/sqrt(D) scaling internally).
    # Pre-scale Q so logits = (Q/sqrt(D)) @ K^T = Q@K^T / sqrt(D).
    scale = jnp.float32(D) ** -0.5
    q = (q * scale).astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    
    # 1. Check cache: hash the union mask to avoid recompiling the kernel
    union_mask = jnp.any(block_mask, axis=(0, 1))  # [Qg, Kg]
    union_np = np.asarray(union_mask)
    cache_key = ('sparse', S, tile_size, hash(union_np.tobytes()))
    
    if cache_key in _SPARSE_SPLASH_CACHE:
        splash_kernel = _SPARSE_SPLASH_CACHE[cache_key]
    else:
        # Build token-level mask on CPU (only on cache miss)
        mask_2d = _block_mask_to_2d_union(block_mask, tile_size)  # [S, S] bool
        
        # Choose SplashAttention block sizes to fit in TPU SMEM (1MB limit)
        num_tiles_sq = (S // tile_size) ** 2
        smem_estimate = 21 * num_tiles_sq
        if smem_estimate > 900_000:
            splash_block = tile_size * 2  # 128 → 256
        else:
            splash_block = tile_size
        
        config = _tokamax_SplashConfig(
            block_q=splash_block,
            block_kv=splash_block,
            block_kv_compute=min(splash_block, 128),
        )
        
        splash_kernel = _tokamax_make_dynamic_splash_mha(
            mask=mask_2d.astype(jnp.bool_),
            config=config,
        )
        _SPARSE_SPLASH_CACHE[cache_key] = splash_kernel
        del mask_2d
    
    # 3. Call kernel for each batch element
    # SplashAttention expects: q=[num_heads, seq_len, head_dim], k/v same
    # Process batch elements sequentially (batch=1 for inference)
    outputs = []
    for b in range(B):
        q_b = q[b]  # [H, S, D]
        k_b = k[b]  # [H, S, D]
        v_b = v[b]  # [H, S, D]
        out_b = splash_kernel(q_b, k_b, v_b)
        outputs.append(out_b)
    
    output = jnp.stack(outputs, axis=0)  # [B, H, S, D]
    output = output.astype(orig_dtype)  # Cast back to original dtype
    return output


# ============================================================
# Pre-warm Kernels (eliminate compilation during timing)
# ============================================================

def prewarm_sparse_kernels(kascade_cache, schedule, tile_size=128, num_heads=32):
    """Pre-build Tokamax kernels for each REUSE layer's block_mask.
    
    Replicates the exact mask transformation that KascadeReuseAttention applies:
    head permutation + diagonal local_mask + union across heads + hash.
    This ensures cache_key matches at runtime → zero recompilation during timing.
    
    Args:
        kascade_cache: The KASCADE_CACHE dict with block_masks from anchor layers
        schedule: Layer schedule dict with head_map for REUSE layers
        tile_size: Block tile size (128 for 32K)
        num_heads: Number of attention heads (32 for LLaMA 3.2-1B)
    """
    if not TOKAMAX_SPLASH_AVAILABLE:
        print("   Prewarm skipped: Tokamax not available")
        return
    
    prewarmed = 0
    already_cached = 0
    for layer_id in sorted(schedule.keys()):
        plan = schedule[layer_id]
        if plan["type"] != "REUSE":
            continue
        
        anchor_id = plan["anchor_id"]
        block_mask = kascade_cache.get(f"layer_{anchor_id}_block_mask")
        if block_mask is None:
            print(f"   Prewarm L{layer_id}: no block_mask for anchor L{anchor_id}")
            continue
        
        # Replicate KascadeReuseAttention's mask transformation exactly
        head_map = plan.get("head_map", {})
        perm_list = [head_map.get(h, h) for h in range(num_heads)]
        perm_indices = jnp.array(perm_list, dtype=jnp.int32)
        bm = block_mask[:, perm_indices, :, :]
        
        num_tiles = block_mask.shape[2]
        local_mask = jnp.eye(num_tiles, dtype=jnp.bool_)
        bm = bm | local_mask[None, None, :, :]
        
        # Compute union + hash (same as splash_sparse_attention)
        union_mask = jnp.any(bm, axis=(0, 1))
        union_np = np.asarray(union_mask)
        S = num_tiles * tile_size
        cache_key = ('sparse', S, tile_size, hash(union_np.tobytes()))
        
        if cache_key in _SPARSE_SPLASH_CACHE:
            already_cached += 1
            continue
        
        # Build token-level mask and compile kernel
        mask_2d = _block_mask_to_2d_union(bm, tile_size)
        
        num_tiles_sq = num_tiles ** 2
        smem_estimate = 21 * num_tiles_sq
        splash_block = tile_size * 2 if smem_estimate > 900_000 else tile_size
        
        config = _tokamax_SplashConfig(
            block_q=splash_block,
            block_kv=splash_block,
            block_kv_compute=min(splash_block, 128),
        )
        
        kernel = _tokamax_make_dynamic_splash_mha(
            mask=mask_2d.astype(jnp.bool_), config=config)
        _SPARSE_SPLASH_CACHE[cache_key] = kernel
        del mask_2d
        prewarmed += 1
        
        # Report density
        causal_mask_blk = jnp.tril(jnp.ones_like(union_mask, dtype=jnp.bool_))
        causal_blocks = int(jnp.sum(causal_mask_blk))
        active = int(jnp.sum(union_mask & causal_mask_blk))
        skipped_pct = (1 - active / causal_blocks) * 100 if causal_blocks > 0 else 0
        print(f"   L{layer_id} (anchor={anchor_id}): {skipped_pct:.0f}% blocks skipped, block={splash_block}")
    
    print(f"   Pre-warmed {prewarmed} new + {already_cached} cached REUSE kernels")


# ============================================================
# Main Entry Point
# ============================================================

def block_sparse_attention(q, k, v, block_mask, tile_size=128, use_pallas=True):
    """Block-sparse attention — main entry point.
    
    Dispatches to Tokamax SplashAttention on TPU, JAX fallback on CPU/GPU.
    """
    if use_pallas and TOKAMAX_SPLASH_AVAILABLE and jax.devices()[0].platform == 'tpu':
        num_heads = q.shape[1]
        return splash_sparse_attention(q, k, v, block_mask, tile_size, num_heads)
    else:
        return block_sparse_attention_jax(q, k, v, block_mask, tile_size)


__all__ = [
    'block_sparse_attention',
    'block_sparse_attention_jax',
    'splash_sparse_attention',
    'full_causal_splash_attention',
    'create_block_mask_from_tile_indices',
    'prewarm_sparse_kernels',
    'TOKAMAX_SPLASH_AVAILABLE',
]
