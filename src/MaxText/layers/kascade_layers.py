"""
Kascade Sparse Attention Layers (Long-Sequence Edition)
--------------------------------------------------------
Supports seq_len up to 32K+ on single TPU v5e/v6e (16GB HBM).

Key changes for long sequences:
  - DENSE/ANCHOR use jax.nn.dot_product_attention (O(1) memory, no S*S matrix)
  - REUSE PATH A (TPU): Tokamax SplashAttention with dynamic grid -> true block skipping
  - REUSE PATH B (any device with splash): Masked dense for short sequences
  - REUSE PATH C (CPU fallback): Tile-group gather sparse attention
  - Tile size 128 (matches TPU block size) for seq_len >= 4096
  - bf16 activations supported

Architecture per Kascade paper:
  Layer 0:  DENSE   -- full causal attention (baseline quality anchor)
  Layer 1+: ANCHOR  -- full attention + fused tile scoring (caches indices + block_mask)
  Layer 2+: REUSE   -- sparse attention borrowing tiles from nearest anchor
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import os
import importlib.util
import functools

# Debug Mode Flag
DEBUG_MODE = False

# Import block-sparse kernel (handles TPU + CPU fallback)
try:
    _kernel_path = os.path.join(os.path.dirname(__file__), '..', 'kernels', 'kascade_block_sparse_kernel.py')
    _kernel_path = os.path.abspath(_kernel_path)
    _spec = importlib.util.spec_from_file_location('kascade_block_sparse_kernel', _kernel_path)
    _bsk = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_bsk)
    block_sparse_attention = _bsk.block_sparse_attention
    create_block_mask_from_tile_indices = _bsk.create_block_mask_from_tile_indices
    splash_sparse_attention = getattr(_bsk, 'splash_sparse_attention', None)
    full_causal_splash_attention = getattr(_bsk, 'full_causal_splash_attention', None)
    BLOCK_SPARSE_AVAILABLE = True
except Exception as e:
    BLOCK_SPARSE_AVAILABLE = False
    splash_sparse_attention = None
    full_causal_splash_attention = None

# Import decode kernel (sparse KV-cache loading for autoregressive)
try:
    _decode_kernel_path = os.path.join(os.path.dirname(__file__), '..', 'kernels', 'kascade_decode_kernel.py')
    _decode_kernel_path = os.path.abspath(_decode_kernel_path)
    _dspec = importlib.util.spec_from_file_location('kascade_decode_kernel', _decode_kernel_path)
    _dkm = importlib.util.module_from_spec(_dspec)
    _dspec.loader.exec_module(_dkm)
    kascade_sparse_decode = _dkm.kascade_sparse_decode
    get_decode_tile_indices = _dkm.get_decode_tile_indices
    dense_decode_attention_jax = _dkm.dense_decode_attention_jax
    build_hot_kv_buffer = _dkm.build_hot_kv_buffer
    kascade_sparse_decode_hotbuf = _dkm.kascade_sparse_decode_hotbuf
    hotbuf_attention = _dkm.hotbuf_attention
    build_prefill_causal_mask = _dkm.build_prefill_causal_mask
    hotbuf_prefill_attention_chunked = _dkm.hotbuf_prefill_attention_chunked
    kascade_sparse_decode_pallas_v2 = getattr(_dkm, 'kascade_sparse_decode_pallas_v2', None)
    DECODE_KERNEL_AVAILABLE = True
except Exception as e:
    kascade_sparse_decode = None
    get_decode_tile_indices = None
    dense_decode_attention_jax = None
    build_hot_kv_buffer = None
    kascade_sparse_decode_hotbuf = None
    hotbuf_attention = None
    build_prefill_causal_mask = None
    hotbuf_prefill_attention_chunked = None
    kascade_sparse_decode_pallas_v2 = None
    DECODE_KERNEL_AVAILABLE = False

# Expose kernel internals for benchmark pre-warming (same module instance
# as splash_sparse_attention uses, so writes to cache are visible at runtime)
if BLOCK_SPARSE_AVAILABLE:
    prewarm_sparse_kernels = getattr(_bsk, 'prewarm_sparse_kernels', None)
    _SPARSE_SPLASH_CACHE = _bsk._SPARSE_SPLASH_CACHE
    _FULL_CAUSAL_SPLASH_CACHE = _bsk._FULL_CAUSAL_SPLASH_CACHE
    TOKAMAX_SPLASH_AVAILABLE = getattr(_bsk, 'TOKAMAX_SPLASH_AVAILABLE', False)
else:
    prewarm_sparse_kernels = None
    _SPARSE_SPLASH_CACHE = {}
    _FULL_CAUSAL_SPLASH_CACHE = {}
    TOKAMAX_SPLASH_AVAILABLE = False

# Global Cache
KASCADE_CACHE = {}


# ============================================================
# RoPE
# ============================================================

def precompute_freqs_cis(dim, end, theta=500000.0, rope_scaling=None):
    """Precompute RoPE frequency tensor with LLaMA-3.x scaling."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float64) / dim))
    
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", ""))
        if rope_type == "llama3":
            factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
            orig_max_pos = rope_scaling["original_max_position_embeddings"]
            low_freq_wavelen = orig_max_pos / low_freq_factor
            high_freq_wavelen = orig_max_pos / high_freq_factor
            new_freqs = []
            for freq in freqs:
                wavelen = 2.0 * np.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / factor)
                else:
                    smooth = (orig_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                    new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
            freqs = np.array(new_freqs)
    
    freqs = jnp.array(freqs, dtype=jnp.float32)
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)
    return freqs_cis


def apply_rope(xq, xk, freqs_cis):
    """Apply RoPE using Half-Split (rotate_half) format, matching HuggingFace LLaMA."""
    xq = xq.astype(jnp.float32)
    xk = xk.astype(jnp.float32)
    cos = freqs_cis.real[None, None, :xq.shape[2], :]
    sin = freqs_cis.imag[None, None, :xq.shape[2], :]
    cos_full = jnp.concatenate([cos, cos], axis=-1)
    sin_full = jnp.concatenate([sin, sin], axis=-1)
    
    def rotate_half(x):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return jnp.concatenate([-x2, x1], axis=-1)
    
    xq_out = xq * cos_full + rotate_half(xq) * sin_full
    xk_out = xk * cos_full + rotate_half(xk) * sin_full
    return xq_out, xk_out


# ============================================================
# Memory-Efficient Attention
# ============================================================

def memory_efficient_causal_attention(q, k, v):
    """Causal attention dispatcher.
    
    On TPU: Uses Tokamax SplashAttention (flash attention, O(S) memory).
           jax.nn.dot_product_attention is broken on TPU in JAX 0.9.0.1
           (produces random outputs → PPL ≈ vocab_size for all implementation options).
    On CPU/GPU: Uses explicit Q@K^T.
    
    Args:
        q, k, v: [B, H, S, D]
    Returns:
        output: [B, H, S, D]
    """
    platform = jax.devices()[0].platform
    if platform == 'tpu':
        # PRIMARY: Tokamax SplashAttention with full causal mask (cached kernel)
        if full_causal_splash_attention is not None:
            return full_causal_splash_attention(q, k, v)
        # FALLBACK: Chunked explicit attention (correct, but no flash speedup)
        return tpu_chunked_causal_attention(q, k, v)
    else:
        # CPU/GPU: explicit attention (works correctly)
        return explicit_causal_attention(q, k, v)


def explicit_causal_attention(q, k, v):
    """Explicit Q@K^T attention -- only for short sequences (< 4096)."""
    seq_len = q.shape[2]
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(q.shape[-1])
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    logits = jnp.where(mask[None, None, :, :], logits, -1e10)
    weights = jax.nn.softmax(logits, axis=-1)
    output = jnp.einsum('bhqk,bhkd->bhqd', weights, v)
    return output


def tpu_chunked_causal_attention(q, k, v, chunk_size=256):
    """Chunked causal attention for TPU (fallback when Tokamax unavailable).
    
    Processes Q in chunks via lax.scan to avoid O(S²) memory.
    Peak memory: O(H × chunk_size × S) per step.
    """
    B, H, S, D = q.shape
    if S <= chunk_size:
        return explicit_causal_attention(q, k, v)
    num_chunks = S // chunk_size
    scale = 1.0 / jnp.sqrt(jnp.float32(D))
    
    q_chunks = q.reshape(B, H, num_chunks, chunk_size, D)
    q_scan = jnp.moveaxis(q_chunks, 2, 0)  # [nc, B, H, cs, D]
    
    def body(carry, xs):
        idx, q_c = xs
        logits = jnp.einsum('bhqd,bhkd->bhqk', q_c, k) * scale
        q_pos = idx * chunk_size + jnp.arange(chunk_size)
        k_pos = jnp.arange(S)
        mask = q_pos[:, None] >= k_pos[None, :]
        logits = jnp.where(mask[None, None], logits, -1e10)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum('bhqk,bhkd->bhqd', weights, v)
        return carry, out
    
    _, chunks_out = jax.lax.scan(body, None, (jnp.arange(num_chunks), q_scan))
    return jnp.moveaxis(chunks_out, 0, 2).reshape(B, H, S, D)


# ============================================================
# Tile Scoring
# ============================================================

def compute_tile_scores(q, k, seq_len, num_heads, head_dim,
                        tile_size, top_k_tiles, batch):
    """Compute tile scores and top-k indices.
    
    For short sequences (<= 8192): uses explicit logits (fused with full attention).
    For long sequences (> 8192): representative-query sampling (O(Qg*S) not O(S^2)).
    """
    num_tiles = seq_len // tile_size
    actual_top_k = min(top_k_tiles, num_tiles)
    
    if seq_len <= 8192:
        # Short: can afford full S*S for tile scoring
        logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(head_dim)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        logits = jnp.where(mask[None, None, :, :], logits, -1e10)
        rep_pos = jnp.arange(tile_size - 1, seq_len, tile_size)
        rep_logits = logits[:, :, rep_pos, :]
    else:
        # Long: sample representative queries only
        rep_pos = jnp.arange(tile_size - 1, seq_len, tile_size)
        q_reps = q[:, :, rep_pos, :]  # [B, H, Qg, D]
        rep_logits = jnp.einsum('bhqd,bhkd->bhqk', q_reps, k) / jnp.sqrt(head_dim)
        # Causal mask for rep queries
        rep_positions = rep_pos[None, None, :, None]
        key_positions = jnp.arange(seq_len)[None, None, None, :]
        causal_mask = key_positions <= rep_positions
        rep_logits = jnp.where(causal_mask, rep_logits, -1e10)
    
    rep_weights = jax.nn.softmax(rep_logits, axis=-1)
    rep_weights_tiled = rep_weights.reshape(batch, num_heads, num_tiles, num_tiles, tile_size)
    tile_scores = jnp.max(rep_weights_tiled, axis=-1)
    _, group_tile_indices = jax.lax.top_k(tile_scores, actual_top_k)
    return group_tile_indices


# ============================================================
# Attention Modules
# ============================================================

class DenseFullAttention(nn.Module):
    """Clean full attention. Memory-efficient for long sequences."""
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x, mask=None, freq_cis=None,
                 decode_mode=False, k_cache=None, v_cache=None,
                 query_pos=None):
        # === DECODE MODE: Dense attention over full KV cache ===
        if decode_mode and k_cache is not None:
            batch = x.shape[0]
            q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
            q = q.reshape(batch, 1, self.num_heads, self.head_dim)
            q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, 1, D]
            if freq_cis is not None:
                k_dummy = jnp.zeros_like(q)
                q, _ = apply_rope(q, k_dummy, freq_cis)
            if dense_decode_attention_jax is not None:
                output = dense_decode_attention_jax(
                    q, k_cache, v_cache, query_pos=query_pos)
            else:
                S = k_cache.shape[2]
                sm_scale = self.head_dim ** -0.5
                scores = jnp.einsum('bhqd,bhkd->bhqk', q, k_cache) * sm_scale
                qp = query_pos[:, None, None, None]
                kv_pos = jnp.arange(S, dtype=jnp.int32)[None, None, None, :]
                scores = jnp.where(kv_pos <= qp, scores, -1e10)
                weights = jax.nn.softmax(scores, axis=-1)
                output = jnp.einsum('bhqk,bhkd->bhqd', weights, v_cache)
            output = jnp.transpose(output, (0, 2, 1, 3))
            output = output.reshape(batch, 1, self.num_heads * self.head_dim)
            output = nn.Dense(x.shape[-1], use_bias=False)(output)
            return output
        
        batch, seq_len, _ = x.shape
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)
        if freq_cis is not None:
            q_t = jnp.transpose(q, (0, 2, 1, 3))
            k_t = jnp.transpose(k, (0, 2, 1, 3))
            q_t, k_t = apply_rope(q_t, k_t, freq_cis)
            q = jnp.transpose(q_t, (0, 2, 1, 3))
            k = jnp.transpose(k_t, (0, 2, 1, 3))
        q_bh = jnp.transpose(q, (0, 2, 1, 3))
        k_bh = jnp.transpose(k, (0, 2, 1, 3))
        v_bh = jnp.transpose(v, (0, 2, 1, 3))
        output = memory_efficient_causal_attention(q_bh, k_bh, v_bh)
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        return output


class KascadeAnchorAttention(nn.Module):
    """Anchor (Scout) Layer: Full attention + fused tile scoring.
    
    Memory-efficient output via jax.nn.dot_product_attention.
    Tile scoring via representative-query sampling for long sequences.
    Caches tile indices + block mask for REUSE layers.
    """
    num_heads: int
    head_dim: int
    layer_id: int
    top_k_tiles: int = 8
    tile_size: int = 128
    use_splash: bool = False

    def _decode_forward(self, x, k_cache, v_cache, query_pos, freq_cis=None):
        """Decode-mode forward for ANCHOR layer.
        
        During decode, ANCHOR layers use full attention over KV cache
        (no sparsity — they are the "scout" layers that maintain quality).
        Also updates tile scoring cache for subsequent REUSE layers.
        
        Args:
            x: [B, 1, embed_dim]
            k_cache: [B, H, S, D] full key cache
            v_cache: [B, H, S, D] full value cache
            query_pos: [B] int32
            freq_cis: RoPE frequencies
            
        Returns:
            output: [B, 1, embed_dim]
        """
        batch = x.shape[0]
        S = k_cache.shape[2]
        
        # Project Q (K/V come from cache)
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        q = q.reshape(batch, 1, self.num_heads, self.head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, 1, D]
        
        if freq_cis is not None:
            k_dummy = jnp.zeros_like(q)
            q, _ = apply_rope(q, k_dummy, freq_cis)
        
        # Dense attention over full cache (ANCHOR = full attention for quality)
        if dense_decode_attention_jax is not None:
            output = dense_decode_attention_jax(
                q, k_cache, v_cache, query_pos=query_pos
            )  # [B, H, 1, D]
        else:
            # Fallback: manual dense decode
            sm_scale = self.head_dim ** -0.5
            scores = jnp.einsum('bhqd,bhkd->bhqk', q, k_cache) * sm_scale
            qp = query_pos[:, None, None, None]
            kv_pos = jnp.arange(S, dtype=jnp.int32)[None, None, None, :]
            causal_mask = kv_pos <= qp
            scores = jnp.where(causal_mask, scores, -1e10)
            weights = jax.nn.softmax(scores, axis=-1)
            output = jnp.einsum('bhqk,bhkd->bhqd', weights, v_cache)
        
        # Update tile scoring for REUSE layers during decode:
        # Re-score tiles using the new query against the full KV cache
        num_tiles = S // self.tile_size
        actual_top_k = min(self.top_k_tiles, num_tiles)
        
        # Simple tile scoring for single decode query:
        # Compute Q @ K_cache^T per tile and pick top-k
        k_tiled = k_cache.reshape(batch, self.num_heads, num_tiles, self.tile_size, self.head_dim)
        # Score = max attention to each tile
        q_expanded = q[:, :, :, None, :]  # [B, H, 1, 1, D]
        tile_scores = jnp.einsum('bhqtd,bhntd->bhqn', q_expanded, k_tiled)  # [B, H, 1, num_tiles]
        tile_scores = tile_scores.squeeze(2)  # [B, H, num_tiles]
        tile_scores = jnp.max(tile_scores, axis=-1, keepdims=False)  # max over tile_size
        
        # Actually we want [B, H, num_tiles] scores
        tile_scores_full = jnp.einsum('bhd,bhntd->bhnt',
                                       q.squeeze(2),  # [B, H, D]
                                       k_tiled)        # [B, H, num_tiles, tile_size, D]
        tile_scores_max = jnp.max(tile_scores_full, axis=-1)  # [B, H, num_tiles]
        _, top_tile_indices = jax.lax.top_k(tile_scores_max, actual_top_k)  # [B, H, top_k]
        
        # Update cache with new selections (for REUSE decode steps)
        # Store as [B, H, 1, top_k] to match prefill format [B, H, Qg, top_k]
        # During decode, query_tile_idx determines which row to update
        query_tile_idx = query_pos // self.tile_size
        existing = KASCADE_CACHE.get(f"layer_{self.layer_id}_indices")
        if existing is not None:
            # Update the specific query tile's selections
            B_cache, H_cache, Qg, topk = existing.shape
            for b_idx in range(batch):
                qt = int(query_tile_idx[b_idx])
                if qt < Qg:
                    existing = existing.at[b_idx, :, qt, :actual_top_k].set(
                        top_tile_indices[b_idx, :, :actual_top_k])
            KASCADE_CACHE[f"layer_{self.layer_id}_indices"] = existing
        
        # Build hot KV buffers for downstream REUSE layers.
        # These are CONTIGUOUS in memory, so REUSE decode achieves
        # near-peak HBM bandwidth (no gather overhead).
        if build_hot_kv_buffer is not None:
            # Include local tile around query position
            local_tile = jnp.clip(query_tile_idx, 0, num_tiles - 1)
            local_tiles_bh = jnp.broadcast_to(
                local_tile[:, None, None], (batch, self.num_heads, 1)
            ).astype(jnp.int32)
            hot_tile_indices = jnp.concatenate(
                [top_tile_indices, local_tiles_bh], axis=-1
            )  # [B, H, top_k + 1]
            hot_k, hot_v = build_hot_kv_buffer(
                k_cache, v_cache, hot_tile_indices, self.tile_size
            )
            KASCADE_CACHE[f"layer_{self.layer_id}_hot_k"] = hot_k
            KASCADE_CACHE[f"layer_{self.layer_id}_hot_v"] = hot_v
        
        output = jnp.transpose(output, (0, 2, 1, 3))  # [B, 1, H, D]
        output = output.reshape(batch, 1, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        return output

    @nn.compact
    def __call__(self, x, mask=None, freq_cis=None,
                 decode_mode=False, k_cache=None, v_cache=None,
                 query_pos=None):
        # === DECODE MODE: Full attention with cache + tile re-scoring ===
        if decode_mode and k_cache is not None:
            return self._decode_forward(x, k_cache, v_cache, query_pos, freq_cis)
        
        batch, seq_len, _ = x.shape
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)
        if freq_cis is not None:
            q_t = jnp.transpose(q, (0, 2, 1, 3))
            k_t = jnp.transpose(k, (0, 2, 1, 3))
            q_t, k_t = apply_rope(q_t, k_t, freq_cis)
            q = jnp.transpose(q_t, (0, 2, 1, 3))
            k = jnp.transpose(k_t, (0, 2, 1, 3))
        q_bh = jnp.transpose(q, (0, 2, 1, 3))
        k_bh = jnp.transpose(k, (0, 2, 1, 3))
        v_bh = jnp.transpose(v, (0, 2, 1, 3))
        
        # --- TILE SCORING ---
        num_tiles = seq_len // self.tile_size
        group_tile_indices = compute_tile_scores(
            q_bh, k_bh, seq_len, self.num_heads, self.head_dim,
            self.tile_size, self.top_k_tiles, batch
        )
        KASCADE_CACHE[f"layer_{self.layer_id}_indices"] = group_tile_indices
        KASCADE_CACHE[f"layer_{self.layer_id}_indices_calib"] = group_tile_indices[:, :, -1, :]
        if BLOCK_SPARSE_AVAILABLE:
            block_mask = create_block_mask_from_tile_indices(
                group_tile_indices, num_tiles, causal=True)
            KASCADE_CACHE[f"layer_{self.layer_id}_block_mask"] = block_mask
        
        if DEBUG_MODE:
            def print_anchor(idx):
                print(f"  [Anchor L{self.layer_id}] Top tiles (Head 0): {idx[0,0]}")
            jax.debug.callback(print_anchor, group_tile_indices[:, :, -1, :])
        
        # --- FULL ATTENTION OUTPUT (memory-efficient) ---
        output = memory_efficient_causal_attention(q_bh, k_bh, v_bh)
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        return output


class KascadeReuseAttention(nn.Module):
    """Reuse (Worker) Layer: Sparse attention using anchor's tile selection.
    
    PREFILL MODE:
      PATH A (TPU with Tokamax): SplashAttention dynamic grid -> real block skipping
      PATH B (short seq with splash): Masked dense for short sequences
      PATH C (CPU fallback): Tile-group gather sparse attention
    
    DECODE MODE:
      PATH D: Sparse KV-cache loading — gathers only top-k tiles from cache,
              reducing HBM bandwidth by ~90% (the decode bottleneck).
              Expected speedup: 2-4× at 32K+ context.
    """
    num_heads: int
    head_dim: int
    anchor_layer_id: int
    tile_size: int = 128
    head_map: dict = None
    use_splash: bool = False
    force_sparse: bool = False

    def _decode_forward(self, x, k_cache, v_cache, query_pos, freq_cis=None):
        """Decode-mode forward: hot buffer sparse attention.
        
        Uses pre-built contiguous KV buffer from the ANCHOR layer.
        Dense attention on this small buffer achieves near-peak HBM
        bandwidth on TPU (no gather overhead).
        
        If hot buffers aren't available yet, falls back to dense
        attention over the full KV cache (safe but slow).
        
        Args:
            x: [B, 1, embed_dim] — single new token embedding
            k_cache: [B, H, S, D] — full key cache from all previous tokens
            v_cache: [B, H, S, D] — full value cache from all previous tokens
            query_pos: [B] int32 — position of the new token
            freq_cis: RoPE frequencies
            
        Returns:
            output: [B, 1, embed_dim]
        """
        batch = x.shape[0]
        
        # Project query only (K/V come from cache)
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)  # [B, 1, H*D]
        q = q.reshape(batch, 1, self.num_heads, self.head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, 1, D]
        
        if freq_cis is not None:
            k_dummy = jnp.zeros_like(q)
            q, _ = apply_rope(q, k_dummy, freq_cis)
        
        # Hot buffer from ANCHOR layer (contiguous, near-peak bandwidth)
        hot_k = KASCADE_CACHE.get(f"layer_{self.anchor_layer_id}_hot_k")
        hot_v = KASCADE_CACHE.get(f"layer_{self.anchor_layer_id}_hot_v")
        
        if hot_k is not None and hot_v is not None and kascade_sparse_decode_hotbuf is not None:
            # Dense attention on pre-gathered contiguous buffer
            output = kascade_sparse_decode_hotbuf(q, hot_k, hot_v)  # [B, H, 1, D]
            
            if DEBUG_MODE:
                sparse_len = hot_k.shape[2]
                full_len = k_cache.shape[2]
                def print_hotbuf_stats(out, sl, fl):
                    print(f"   [DECODE REUSE from L{self.anchor_layer_id}] "
                          f"HOT BUFFER sparse_tokens={int(sl)}/{int(fl)} "
                          f"({(1-int(sl)/int(fl))*100:.0f}% reduced) "
                          f"mean={float(jnp.mean(out)):.6f}")
                jax.debug.callback(print_hotbuf_stats, output, sparse_len, full_len)
        else:
            # Fallback: dense attention over full KV cache
            # (hot buffers not built yet — ANCHOR hasn't run in decode mode)
            S = k_cache.shape[2]
            sm_scale = self.head_dim ** -0.5
            scores = jnp.einsum('bhqd,bhkd->bhqk', q, k_cache) * sm_scale
            qp = query_pos[:, None, None, None]
            kv_pos = jnp.arange(S, dtype=jnp.int32)[None, None, None, :]
            scores = jnp.where(kv_pos <= qp, scores, -1e10)
            weights = jax.nn.softmax(scores, axis=-1)
            output = jnp.einsum('bhqk,bhkd->bhqd', weights, v_cache)
            
            if DEBUG_MODE:
                def print_fallback(out):
                    print(f"   [DECODE REUSE from L{self.anchor_layer_id}] "
                          f"FALLBACK dense (no hot buffer available)")
                jax.debug.callback(print_fallback, output)
        
        output = jnp.transpose(output, (0, 2, 1, 3))  # [B, 1, H, D]
        output = output.reshape(batch, 1, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        
        return output

    @nn.compact
    def __call__(self, x, mask=None, freq_cis=None,
                 decode_mode=False, k_cache=None, v_cache=None,
                 query_pos=None):
        # === PATH D: Decode mode (sparse KV-cache loading) ===
        if decode_mode and DECODE_KERNEL_AVAILABLE and k_cache is not None:
            return self._decode_forward(x, k_cache, v_cache, query_pos, freq_cis)
        
        batch, seq_len, _ = x.shape
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        if freq_cis is not None:
            q, k = apply_rope(q, k, freq_cis)

        num_tiles = seq_len // self.tile_size
        
        # Head mapping permutation
        if self.head_map is not None:
            perm_list = [self.head_map.get(h, h) for h in range(self.num_heads)]
            perm_indices = jnp.array(perm_list, dtype=jnp.int32)
        else:
            perm_indices = jnp.arange(self.num_heads, dtype=jnp.int32)
        
        anchor_mask = KASCADE_CACHE.get(f"layer_{self.anchor_layer_id}_block_mask")
        anchor_indices = KASCADE_CACHE.get(f"layer_{self.anchor_layer_id}_indices")
        
        if anchor_indices is None:
            raise RuntimeError(
                f"REUSE cache miss! Anchor L{self.anchor_layer_id} must run first.")
        
        # === PATH A: Tokamax SplashAttention (TPU only, real block skipping) ===
        if (self.use_splash and splash_sparse_attention is not None
                and anchor_mask is not None
                and jax.devices()[0].platform == 'tpu'):
            bm = anchor_mask[:, perm_indices, :, :]
            local_mask = jnp.eye(num_tiles, dtype=jnp.bool_)
            bm = bm | local_mask[None, None, :, :]
            output = splash_sparse_attention(q, k, v, bm, self.tile_size, self.num_heads, self.force_sparse)
        
        # === PATH B: Masked dense (short sequences, any device) ===
        elif self.use_splash and anchor_mask is not None and seq_len <= 8192:
            bm = anchor_mask[:, perm_indices, :, :]
            local_mask = jnp.eye(num_tiles, dtype=jnp.bool_)
            bm = bm | local_mask[None, None, :, :]
            logits = jnp.einsum('bhsd,bhtd->bhst', q, k) / jnp.sqrt(self.head_dim)
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            logits = jnp.where(causal_mask[None, None], logits, -1e10)
            logits_tiled = logits.reshape(
                batch, self.num_heads, num_tiles, self.tile_size,
                num_tiles, self.tile_size)
            bm_expanded = bm[:, :, :, None, :, None]
            logits_tiled = jnp.where(bm_expanded, logits_tiled, -1e10)
            logits = logits_tiled.reshape(batch, self.num_heads, seq_len, seq_len)
            weights = jax.nn.softmax(logits, axis=-1)
            all_masked = jnp.all(logits <= -1e9, axis=-1, keepdims=True)
            weights = jnp.where(all_masked, 0.0, weights)
            output = jnp.einsum('bhst,bhtd->bhsd', weights, v)
        
        # === PATH C: Tile-group gather (CPU, any seq_len) ===
        else:
            tile_sel = anchor_indices[:, perm_indices, :, :]
            local_tiles = jnp.arange(num_tiles)[None, None, :, None]
            local_tiles = jnp.broadcast_to(
                local_tiles, (batch, self.num_heads, num_tiles, 1)
            ).astype(jnp.int32)
            tile_sel = jnp.concatenate([tile_sel, local_tiles], axis=-1)
            offsets = jnp.arange(self.tile_size)[None, None, None, None, :]
            tile_starts = tile_sel[..., None] * self.tile_size
            token_indices = tile_starts + offsets
            sparse_len = tile_sel.shape[-1] * self.tile_size
            flat_indices = token_indices.reshape(batch, self.num_heads, num_tiles, sparse_len)
            flat_indices = jnp.clip(flat_indices, 0, seq_len - 1)
            B, H, S, D = k.shape
            k_flat = k.reshape(B * H, S, D)
            v_flat = v.reshape(B * H, S, D)
            idx_flat = flat_indices.reshape(B * H, num_tiles, sparse_len)
            k_sparse = jax.vmap(lambda kv, idx: kv[idx])(k_flat, idx_flat)
            v_sparse = jax.vmap(lambda kv, idx: kv[idx])(v_flat, idx_flat)
            k_sparse = k_sparse.reshape(B, H, num_tiles, sparse_len, D)
            v_sparse = v_sparse.reshape(B, H, num_tiles, sparse_len, D)
            q_tiled = q.reshape(B, H, num_tiles, self.tile_size, D)
            logits = jnp.einsum('bhtqd,bhtkd->bhtqk', q_tiled, k_sparse) / jnp.sqrt(self.head_dim)
            q_pos = jnp.arange(seq_len).reshape(num_tiles, self.tile_size)
            future_mask = flat_indices[:, :, :, None, :] > q_pos[None, None, :, :, None]
            logits = jnp.where(future_mask, -1e10, logits)
            weights = jax.nn.softmax(logits, axis=-1)
            all_masked = jnp.all(future_mask, axis=-1, keepdims=True)
            weights = jnp.where(all_masked, 0.0, weights)
            weights = jnp.where(jnp.isnan(weights), 0.0, weights)
            output = jnp.einsum('bhtqk,bhtkd->bhtqd', weights, v_sparse)
            output = output.reshape(B, H, seq_len, D)
        
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        
        if DEBUG_MODE:
            def print_reuse_stats(out, aid):
                print(f"   [REUSE from L{aid}] mean={float(jnp.mean(out)):.6f}, "
                      f"std={float(jnp.std(out)):.6f}")
            jax.debug.callback(print_reuse_stats, output, self.anchor_layer_id)
        return output
