"""
Kascade Sparse Attention Layers
--------------------------------
Implements the Anchor (Scout) and Reuse (Worker) attention patterns
with tile pooling and Top-K extraction for sparse attention.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import os
import importlib.util

# Debug Mode Flag - Set to False for production benchmarks
DEBUG_MODE = False  # Disable for clean benchmark output

# Import block-sparse kernel (handles TPU + CPU fallback)
try:
    _kernel_path = os.path.join(os.path.dirname(__file__), '..', 'kernels', 'kascade_block_sparse_kernel.py')
    _kernel_path = os.path.abspath(_kernel_path)
    _spec = importlib.util.spec_from_file_location('kascade_block_sparse_kernel', _kernel_path)
    _bsk = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_bsk)
    block_sparse_attention = _bsk.block_sparse_attention
    create_block_mask_from_tile_indices = _bsk.create_block_mask_from_tile_indices
    BLOCK_SPARSE_AVAILABLE = True
except Exception:
    BLOCK_SPARSE_AVAILABLE = False

# Global Cache to pass data between layers during Calibration & Inference
# Format: { "layer_0_indices": array([Batch, Heads, TopK_Tiles]) }
KASCADE_CACHE = {}

# RoPE Helper Functions for LLaMA-3.x
def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0,
                         rope_scaling: dict = None):
    """
    Precompute the frequency tensor for RoPE (Rotary Position Embedding).
    Supports LLaMA-3.x 'llama3' rope_scaling with frequency-dependent scaling.
    
    Args:
        dim: Head dimension (e.g. 64 for LLaMA-3.2-1B, 128 for LLaMA-3.1-8B)
        end: Maximum sequence length
        theta: Base for frequency computation (500000 for LLaMA-3.x)
        rope_scaling: Optional dict with keys:
            - rope_type: "llama3"
            - factor: scaling factor (e.g. 32.0)
            - low_freq_factor: (e.g. 1.0)
            - high_freq_factor: (e.g. 4.0)
            - original_max_position_embeddings: (e.g. 8192)
    
    Returns:
        Complex tensor of shape [end, dim//2] for RoPE rotation
    """
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float64) / dim))
    
    # Apply LLaMA-3 rope scaling if configured
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
                    # High frequency region: keep original
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    # Low frequency region: scale down by factor
                    new_freqs.append(freq / factor)
                else:
                    # Medium frequency: smooth interpolation
                    smooth = (orig_max_pos / wavelen - low_freq_factor) / (
                        high_freq_factor - low_freq_factor)
                    new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
            freqs = np.array(new_freqs)
    
    freqs = jnp.array(freqs, dtype=jnp.float32)
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)  # Complex exponential
    return freqs_cis

def apply_rope(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray):
    """
    Apply RoPE using Half-Split (rotate_half) format, matching HuggingFace LLaMA.
    Pairs dimension d with d + dim//2 (NOT adjacent pairs d, d+1).
    
    This matches HF's rotate_half convention:
      x_rotated = x * cos + rotate_half(x) * sin
    where rotate_half splits x into (x1, x2) at the midpoint and returns (-x2, x1).
    
    Args:
        xq: Query tensor [batch, heads, seq, dim]
        xk: Key tensor [batch, heads, seq, dim]
        freqs_cis: Precomputed frequency tensor [seq, dim//2] (complex)
    
    Returns:
        Rotated query and key tensors in float32
    """
    # xq, xk shape: [Batch, Heads, Seq, Dim]
    # freqs_cis shape: [Seq, Dim//2] (Complex)
    
    # Cast to float32
    xq = xq.astype(jnp.float32)
    xk = xk.astype(jnp.float32)
    
    # Extract cos and sin from complex frequencies
    cos = freqs_cis.real[None, None, :xq.shape[2], :]  # [1, 1, S, D//2]
    sin = freqs_cis.imag[None, None, :xq.shape[2], :]  # [1, 1, S, D//2]
    
    # Broadcast to full dimension: [1, 1, S, D]
    cos_full = jnp.concatenate([cos, cos], axis=-1)
    sin_full = jnp.concatenate([sin, sin], axis=-1)
    
    def rotate_half(x):
        """Split x into two halves and rotate: (x1, x2) -> (-x2, x1)."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return jnp.concatenate([-x2, x1], axis=-1)
    
    xq_out = xq * cos_full + rotate_half(xq) * sin_full
    xk_out = xk * cos_full + rotate_half(xk) * sin_full
    
    return xq_out, xk_out

class DenseFullAttention(nn.Module):
    """
    Clean full attention module for DENSE baseline.
    No tile selection, no caching — just standard causal self-attention.
    Used for fair baseline timing comparison.
    """
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x, mask=None, freq_cis=None):
        batch, seq_len, _ = x.shape

        # 1. Q, K, V projections
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)

        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)

        # 2. Apply RoPE
        if freq_cis is not None:
            q_t = jnp.transpose(q, (0, 2, 1, 3))
            k_t = jnp.transpose(k, (0, 2, 1, 3))
            q_t, k_t = apply_rope(q_t, k_t, freq_cis)
            q = jnp.transpose(q_t, (0, 2, 1, 3))
            k = jnp.transpose(k_t, (0, 2, 1, 3))

        # 3. Full causal attention
        logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)
        if mask is None:
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        logits = jnp.where(mask[None, None, :, :], logits, -1e10)
        weights = jax.nn.softmax(logits, axis=-1)

        # 4. Output
        output = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        return output


class KascadeAnchorAttention(nn.Module):
    """
    The 'Scout' Layer.
    Implements Requirement #3: Tile Pooling & Top-K Extraction.
    """
    num_heads: int
    head_dim: int
    layer_id: int
    
    # --- Configurable Parameters (Defaults) ---
    top_k_tiles: int = 8   # How many tiles to keep (Sparsity Budget: 8/32 = 75% savings)
    tile_size: int = 16    # Size of each memory block (Hardware friendly)
    use_splash: bool = False  # Use SplashAttention kernel if available

    @nn.compact
    def __call__(self, x, mask=None, freq_cis=None):
        batch, seq_len, _ = x.shape
        
        # DEBUG: Check input dtype
        if DEBUG_MODE:
            print(f"   ANCHOR input dtype: {x.dtype}")
        
        # 1. Standard Dense Projections (Q, K, V)
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        
        # DEBUG: Check projection output dtypes
        if DEBUG_MODE:
            print(f"   ANCHOR after Dense: q={q.dtype}, k={k.dtype}, v={v.dtype}")
        
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE if provided (for LLaMA-3.1 compatibility)
        if freq_cis is not None:
            # Transpose to [Batch, Heads, Seq, Dim] for RoPE
            q_t = jnp.transpose(q, (0, 2, 1, 3))
            k_t = jnp.transpose(k, (0, 2, 1, 3))
            q_t, k_t = apply_rope(q_t, k_t, freq_cis)
            # Transpose back to [Batch, Seq, Heads, Dim]
            q = jnp.transpose(q_t, (0, 2, 1, 3))
            k = jnp.transpose(k_t, (0, 2, 1, 3))
        
        # ANCHOR layers ALWAYS compute full attention (paper requirement).
        # The use_splash flag only affects REUSE layers (block-sparse kernel).
        # ANCHOR computes full attention output + caches tile indices for REUSE.
        
        # ============================================================
        # ANCHOR = FULL ATTENTION OUTPUT + TILE INDEX CACHING
        # ============================================================
        # Per Kascade paper: ANCHOR layers compute FULL attention (same quality
        # as dense) and additionally determine which tiles are important.
        # The tile indices are cached for REUSE layers to borrow.
        # Only REUSE layers do sparse attention (that's where speedup comes from).
        #
        # Memory strategy: compute tile scores FIRST with rep-query method (128MB),
        # then compute full attention for output (2GB). They don't coexist in memory.
        # This avoids the 8GB peak from reshaping the full [B,H,S,S] weights tensor.
        
        num_tiles = seq_len // self.tile_size
        actual_top_k = min(self.top_k_tiles, num_tiles)
        
        # Transpose to [B, H, S, D] for attention
        q_bh = jnp.transpose(q, (0, 2, 1, 3))
        k_bh = jnp.transpose(k, (0, 2, 1, 3))
        v_bh = jnp.transpose(v, (0, 2, 1, 3))
        
        # --- STEP 1: TILE SCORING with representative queries (128MB) ---
        # Done BEFORE full attention so they don't compete for memory.
        rep_pos = jnp.arange(self.tile_size - 1, seq_len, self.tile_size)  # [Qg]
        q_reps = q_bh[:, :, rep_pos, :]  # [B, H, Qg, D]
        
        rep_logits = jnp.einsum('bhgd,bhsd->bhgs', q_reps, k_bh) / jnp.sqrt(self.head_dim)
        
        # Causal mask for reps
        rep_positions = rep_pos[None, None, :, None]
        all_positions = jnp.arange(seq_len)[None, None, None, :]
        rep_logits = jnp.where(all_positions <= rep_positions, rep_logits, -1e10)
        
        rep_weights = jax.nn.softmax(rep_logits, axis=-1)  # [B, H, Qg, S]
        
        # Max-pool over tile_size to get tile scores [B, H, Qg, num_tiles]
        rep_weights_tiled = rep_weights.reshape(
            batch, self.num_heads, num_tiles, num_tiles, self.tile_size)
        tile_scores = jnp.max(rep_weights_tiled, axis=-1)  # [B, H, Qg, num_tiles]
        
        _, group_tile_indices = jax.lax.top_k(tile_scores, actual_top_k)
        # [B, H, Qg, top_k]
        
        # Cache indices for REUSE layers
        top_tile_indices = jnp.repeat(group_tile_indices, self.tile_size, axis=2)
        last_token_indices = group_tile_indices[:, :, -1, :]  # [B, H, top_k]
        cache_key = f"layer_{self.layer_id}_indices"
        KASCADE_CACHE[cache_key] = top_tile_indices  # [B, H, Q, top_k]
        KASCADE_CACHE[f"layer_{self.layer_id}_indices_calib"] = last_token_indices
        
        if DEBUG_MODE:
            def print_anchor(idx):
                print(f"  [Anchor L{self.layer_id}] Top-{actual_top_k} Tiles (Head 0): {idx[0,0]}")
            jax.debug.callback(print_anchor, last_token_indices)
        
        # --- STEP 2: FULL ATTENTION OUTPUT (same as DenseFullAttention) ---
        # rep_weights is small (128MB) and will be freed by XLA.
        # Full attention uses the same 2GB as the dense baseline.
        logits = jnp.einsum('bhqd,bhkd->bhqk', q_bh, k_bh) / jnp.sqrt(self.head_dim)
        if mask is None:
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        logits = jnp.where(mask[None, None, :, :], logits, -1e10)
        weights = jax.nn.softmax(logits, axis=-1)
        output = jnp.einsum('bhqk,bhkd->bhqd', weights, v_bh)
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        
        # Debug: track output stats
        if DEBUG_MODE:
            def print_anchor_stats(out, lid):
                print(f"   [ANCHOR L{lid}] output: mean={float(jnp.mean(out)):.6f}, std={float(jnp.std(out)):.6f}, "
                      f"norm={float(jnp.linalg.norm(out)):.4f}, has_nan={bool(jnp.any(jnp.isnan(out)))}")
            jax.debug.callback(print_anchor_stats, output, self.layer_id)
        
        # --- DEBUG: Check for NaN ---
        if DEBUG_MODE:
            def check_nan(x, layer_id):
                if jnp.any(jnp.isnan(x)):
                    print(f"  ⚠️  NaN detected in Anchor Layer {layer_id} output!")
            jax.debug.callback(check_nan, output, self.layer_id)
        
        return output

class KascadeReuseAttention(nn.Module):
    """
    Production-Grade Reuse Layer.
    Uses per-query tile indices from the anchor layer for correct sparse attention.
    Adds local tile per query for causal context.
    """
    num_heads: int
    head_dim: int
    anchor_layer_id: int
    tile_size: int = 16
    head_map: dict = None
    use_splash: bool = False

    @nn.compact
    def __call__(self, x, mask=None, freq_cis=None):
        batch, seq_len, _ = x.shape
        
        # 1. Projections
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for gathering
        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, S, D]
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Apply RoPE
        if freq_cis is not None:
            q, k = apply_rope(q, k, freq_cis)

        # 2. Retrieve Per-Query Anchor Indices
        cache_key = f"layer_{self.anchor_layer_id}_indices"
        anchor_indices = KASCADE_CACHE.get(cache_key, None)
        
        if anchor_indices is None:
            raise RuntimeError(
                f"REUSE Layer cache miss! "
                f"Anchor layer {self.anchor_layer_id} must run before REUSE layer."
            )
        
        # anchor_indices shape: [B, H, Q, top_k] (per-query tile selections)
        
        # 3. Apply Head Mapping (per-query)
        if self.head_map is not None:
            perm_list = [self.head_map.get(h, h) for h in range(self.num_heads)]
            perm_indices = jnp.array(perm_list, dtype=jnp.int32)
            my_tile_indices = anchor_indices[:, perm_indices, :, :]  # [B, H, Q, top_k]
        else:
            my_tile_indices = anchor_indices
            
        # 4. Add PER-QUERY LOCAL TILE for causal context
        num_tiles = seq_len // self.tile_size
        local_tiles = jnp.arange(seq_len) // self.tile_size  # [Q]
        local_tile_indices = local_tiles[None, None, :, None]  # [1, 1, Q, 1]
        local_tile_indices = jnp.broadcast_to(
            local_tile_indices, (batch, self.num_heads, seq_len, 1)
        ).astype(jnp.int32)
        
        # Concatenate: [sparse tiles from anchor] + [local tile per query]
        # Shape: [B, H, Q, top_k + 1]
        my_tile_indices = jnp.concatenate([my_tile_indices, local_tile_indices], axis=-1)
            
        # 5. BLOCK-SPARSE ATTENTION
        #    Two paths:
        #    A) use_splash=True → Masked dense attention (TPU MXU-friendly, no gather)
        #    B) use_splash=False → Tile-group gather (CPU fallback)
        
        num_q_tiles = seq_len // self.tile_size
        num_kv_tiles = num_q_tiles  # same as num_tiles since Q and K have same seq_len
        
        if self.use_splash and BLOCK_SPARSE_AVAILABLE:
            # === PATH A: MASKED DENSE ATTENTION (TPU-optimized) ===
            # Key insight: On TPU, full Q@K^T via MXU is faster than gather-based
            # sparse attention. So we compute full attention scores, then MASK out
            # non-selected tile blocks before softmax. This gives:
            #   - MXU-optimal matmuls (same speed as dense Q@K^T and @V)
            #   - Tile-level sparsity in attention weights (same quality as gather)
            #   - Zero gather overhead
            #
            # The tile mask is applied via a reshape trick (no data copy):
            #   logits [B,H,S,S] → [B,H,Qg,ts,Kg,ts] → apply block_mask → reshape back
            
            # Use last query per tile-group as representative for tile selection
            tile_repr = jnp.arange(self.tile_size - 1, seq_len, self.tile_size)
            tile_sel = my_tile_indices[:, :, tile_repr, :]  # [B, H, Qg, top_k+1]
            
            # Create block mask: [B, H, Qg, Kg] boolean
            bm = create_block_mask_from_tile_indices(
                tile_sel, num_kv_tiles, causal=True
            )
            
            # Full Q@K^T — uses MXU on TPU, same cost as dense
            logits = jnp.einsum('bhsd,bhtd->bhst', q, k) / jnp.sqrt(self.head_dim)
            
            # Causal mask (token-level)
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            logits = jnp.where(causal_mask[None, None], logits, -1e10)
            
            # Apply tile mask via reshape (zero-copy view change)
            # [B, H, S, S] → [B, H, Qg, ts, Kg, ts]
            logits_tiled = logits.reshape(
                batch, self.num_heads, num_q_tiles, self.tile_size,
                num_kv_tiles, self.tile_size
            )
            # bm: [B, H, Qg, Kg] → [B, H, Qg, 1, Kg, 1]
            bm_expanded = bm[:, :, :, None, :, None]
            logits_tiled = jnp.where(bm_expanded, logits_tiled, -1e10)
            logits = logits_tiled.reshape(batch, self.num_heads, seq_len, seq_len)
            
            # Softmax + weighted sum — uses MXU on TPU
            weights = jax.nn.softmax(logits, axis=-1)
            # Handle all-masked rows (early tokens)
            all_masked = jnp.all(logits <= -1e9, axis=-1, keepdims=True)
            weights = jnp.where(all_masked, 0.0, weights)
            
            output = jnp.einsum('bhst,bhtd->bhsd', weights, v)
            
            output = jnp.transpose(output, (0, 2, 1, 3))
            output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
            output = nn.Dense(x.shape[-1], use_bias=False)(output)
            
            if DEBUG_MODE:
                def print_reuse_stats(out, aid):
                    print(f"   [REUSE-MASKED from L{aid}] mean={float(jnp.mean(out)):.6f}, "
                          f"std={float(jnp.std(out)):.6f}, has_nan={bool(jnp.any(jnp.isnan(out)))}")
                jax.debug.callback(print_reuse_stats, output, self.anchor_layer_id)
            
            return output
        
        # === PATH B: TILE-GROUP GATHER (fallback) ===
        
        # Reduce to per-tile: use last query in each tile as representative
        # (it sees the most context → best tile selection for the group)
        tile_repr = jnp.arange(self.tile_size - 1, seq_len, self.tile_size)  # [num_q_tiles]
        tile_sel = my_tile_indices[:, :, tile_repr, :]  # [B, H, num_q_tiles, top_k+1]
        
        # Expand to token indices per tile-group
        offsets = jnp.arange(self.tile_size)[None, None, None, None, :]  # [1,1,1,1,ts]
        tile_starts = tile_sel[..., None] * self.tile_size               # [B,H,nqt,sel,1]
        token_indices = tile_starts + offsets                            # [B,H,nqt,sel,ts]
        sparse_len = tile_sel.shape[-1] * self.tile_size
        flat_indices = token_indices.reshape(batch, self.num_heads, num_q_tiles, sparse_len)
        flat_indices = jnp.clip(flat_indices, 0, seq_len - 1)
        
        # Gather K, V per tile-group (16× smaller than per-query)
        B, H, S, D = k.shape
        k_flat = k.reshape(B * H, S, D)
        v_flat = v.reshape(B * H, S, D)
        idx_flat = flat_indices.reshape(B * H, num_q_tiles, sparse_len)
        
        k_sparse = jax.vmap(lambda kv, idx: kv[idx])(k_flat, idx_flat)  # [B*H, nqt, sl, D]
        v_sparse = jax.vmap(lambda kv, idx: kv[idx])(v_flat, idx_flat)
        k_sparse = k_sparse.reshape(B, H, num_q_tiles, sparse_len, D)
        v_sparse = v_sparse.reshape(B, H, num_q_tiles, sparse_len, D)
        
        # Block-sparse attention: Q_tile @ K_sparse^T per tile-group
        q_tiled = q.reshape(B, H, num_q_tiles, self.tile_size, D)
        # [B,H,nqt,ts,D] @ [B,H,nqt,sl,D]^T → [B,H,nqt,ts,sl]
        logits = jnp.einsum('bhtqd,bhtkd->bhtqk', q_tiled, k_sparse) / jnp.sqrt(self.head_dim)
        
        # Causal mask: each query position vs gathered key positions
        q_pos = jnp.arange(seq_len).reshape(num_q_tiles, self.tile_size)  # [nqt, ts]
        # flat_indices: [B, H, nqt, sl] → broadcast to [B, H, nqt, 1, sl]
        future_mask = flat_indices[:, :, :, None, :] > q_pos[None, None, :, :, None]
        logits = jnp.where(future_mask, -1e10, logits)
        
        # Softmax over sparse keys & weighted sum
        weights = jax.nn.softmax(logits, axis=-1)
        all_masked = jnp.all(future_mask, axis=-1, keepdims=True)
        weights = jnp.where(all_masked, 0.0, weights)
        weights = jnp.where(jnp.isnan(weights), 0.0, weights)
        
        # [B,H,nqt,ts,sl] @ [B,H,nqt,sl,D] → [B,H,nqt,ts,D]
        output = jnp.einsum('bhtqk,bhtkd->bhtqd', weights, v_sparse)
        output = output.reshape(B, H, seq_len, D)
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        
        if DEBUG_MODE:
            def print_reuse_stats(out, aid):
                print(f"   [REUSE from L{aid}] mean={float(jnp.mean(out)):.6f}, "
                      f"std={float(jnp.std(out)):.6f}, has_nan={bool(jnp.any(jnp.isnan(out)))}")
            jax.debug.callback(print_reuse_stats, output, self.anchor_layer_id)

        return output
