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
    BLOCK_SPARSE_AVAILABLE = True
except Exception as e:
    BLOCK_SPARSE_AVAILABLE = False
    splash_sparse_attention = None

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
    
    On TPU: Uses jax.nn.dot_product_attention (O(1) memory, no S*S matrix).
    On CPU/GPU: Uses explicit Q@K^T (jax.nn.dot_product_attention is buggy on CPU in JAX 0.9.0).
    
    For CPU with seq_len > ~8192, this will OOM. Use chunked/tiled approach instead.
    
    Args:
        q, k, v: [B, H, S, D]
    Returns:
        output: [B, H, S, D]
    """
    platform = jax.devices()[0].platform
    if platform == 'tpu':
        # Do NOT specify implementation='xla' -- broken in JAX 0.9.0.1
        # (returns unblended attention, PPL ≈ vocab_size = random)
        # Let JAX auto-detect the best flash attention implementation.
        output = jax.nn.dot_product_attention(
            q, k, v,
            is_causal=True,
        )
        return output
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
    def __call__(self, x, mask=None, freq_cis=None):
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

    @nn.compact
    def __call__(self, x, mask=None, freq_cis=None):
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
    
    PATH A (TPU with Tokamax): SplashAttention dynamic grid -> real block skipping
    PATH B (short seq with splash): Masked dense via explicit S*S
    PATH C (CPU fallback): Tile-group gather sparse attention
    """
    num_heads: int
    head_dim: int
    anchor_layer_id: int
    tile_size: int = 128
    head_map: dict = None
    use_splash: bool = False

    @nn.compact
    def __call__(self, x, mask=None, freq_cis=None):
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
            output = splash_sparse_attention(q, k, v, bm, self.tile_size, self.num_heads)
        
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
