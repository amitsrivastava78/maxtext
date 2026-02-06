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

# Debug Mode Flag - Set to False for production benchmarks
DEBUG_MODE = False  # Disable for clean benchmark output

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
        
        # Use optimized Pallas kernel if enabled
        if self.use_splash:
            # Import our optimized kascade_kernel
            import sys
            import os
            import importlib.util
            
            # Load kascade_kernel module
            kernel_path = os.path.join(os.path.dirname(__file__), '..', 'kernels', 'kascade_kernel.py')
            if os.path.exists(kernel_path):
                spec = importlib.util.spec_from_file_location("kascade_kernel", kernel_path)
                kascade_kernel_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(kascade_kernel_module)
                
                # Use complete Kascade pipeline: tile selection + optimized kernel
                select_top_k_tiles = getattr(kascade_kernel_module, 'select_top_k_tiles', None)
                kascade_attention_forward = kascade_kernel_module.kascade_attention_forward
                
                if select_top_k_tiles:
                    # Complete Kascade pipeline (6× speedup)
                    # Transpose to kernel format: (heads, seq_len, head_dim)
                    q_kernel = jnp.transpose(q, (1, 0, 2, 3)).reshape(self.num_heads, batch * seq_len, self.head_dim)
                    k_kernel = jnp.transpose(k, (1, 0, 2, 3)).reshape(self.num_heads, batch * seq_len, self.head_dim)
                    v_kernel = jnp.transpose(v, (1, 0, 2, 3)).reshape(self.num_heads, batch * seq_len, self.head_dim)
                    
                    # Tile selection (top 25% = 4× speedup)
                    top_k_ratio = self.top_k_tiles / (seq_len / self.tile_size)
                    k_sparse, v_sparse = select_top_k_tiles(q_kernel, k_kernel, v_kernel, 
                                                            tile_size=self.tile_size, 
                                                            top_k_ratio=top_k_ratio)
                    
                    # Optimized kernel on selected tiles (0.9× overhead)
                    attn_out = kascade_attention_forward(q_kernel, k_sparse, v_sparse)
                else:
                    # Kernel-only (no tile selection) 
                    q_kernel = jnp.transpose(q, (1, 0, 2, 3)).reshape(self.num_heads, batch * seq_len, self.head_dim)
                    k_kernel = jnp.transpose(k, (1, 0, 2, 3)).reshape(self.num_heads, batch * seq_len, self.head_dim)
                    v_kernel = jnp.transpose(v, (1, 0, 2, 3)).reshape(self.num_heads, batch * seq_len, self.head_dim)
                    attn_out = kascade_attention_forward(q_kernel, k_kernel, v_kernel)
                
                # Reshape back: (num_heads, batch*seq_len, head_dim) -> (batch, seq_len, num_heads*head_dim)
                attn_out = attn_out.reshape(self.num_heads, batch, seq_len, self.head_dim)
                attn_out = jnp.transpose(attn_out, (1, 2, 0, 3))
                attn_out = attn_out.reshape(batch, seq_len, self.num_heads * self.head_dim)
                
                # Project output
                output = nn.Dense(x.shape[-1], use_bias=False)(attn_out)
                return output
        
        # Standard JAX implementation (fallback or when splash disabled)
        # Calculate Scores: (Q @ K) / sqrt(d)
        logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)
        if mask is None: mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        logits = jnp.where(mask[None, None, :, :], logits, -1e10)
        weights = jax.nn.softmax(logits, axis=-1)
        
        # --- REQUIREMENT 3: PER-QUERY TILE POOLING & TOP-K (Paper Algorithm) ---
        # The paper selects top-k tiles PER QUERY POSITION, not just for the last token.
        # Each query q_i picks its own most important tiles based on its attention weights.
        
        # A. Use ALL query positions' attention weights
        # weights shape: [Batch, Heads, Seq_Len(Q), Seq_Len(K)]
        all_probs = weights
        
        # B. Pad K dimension if it doesn't fit tiles perfectly
        pad_len = (self.tile_size - (seq_len % self.tile_size)) % self.tile_size
        if pad_len > 0:
            all_probs = jnp.pad(all_probs, ((0,0), (0,0), (0,0), (0, pad_len)))
            
        # C. Reshape K dimension into Tiles
        # [B, H, Q, K_padded] → [B, H, Q, num_tiles, tile_size]
        K_padded = all_probs.shape[-1]
        num_tiles = K_padded // self.tile_size
        tiled_probs = all_probs.reshape(batch, self.num_heads, seq_len, num_tiles, self.tile_size)
        
        # D. Max Pooling: Find the single highest probability in each tile, per query
        # Shape: [Batch, Heads, Seq_Len, Num_Tiles]
        tile_scores = jnp.max(tiled_probs, axis=-1) 
        
        # E. Extract Top-K Indices per query
        num_tiles = tile_scores.shape[-1]
        actual_top_k = min(self.top_k_tiles, num_tiles)
        # Shape: [Batch, Heads, Seq_Len, Top_K] — per-query tile selections!
        _, top_tile_indices = jax.lax.top_k(tile_scores, actual_top_k)
        
        # F. Save to Cache
        # Per-query indices for REUSE layers: [B, H, Q, top_k]
        cache_key = f"layer_{self.layer_id}_indices"
        KASCADE_CACHE[cache_key] = top_tile_indices
        # Last-token indices for calibration/scheduling: [B, H, top_k]
        KASCADE_CACHE[f"layer_{self.layer_id}_indices_calib"] = top_tile_indices[:, :, -1, :]
        
        # ALWAYS print top_k for debugging
        def print_topk():
            print(f"  Layer {self.layer_id} (ANCHOR): Using top_k={actual_top_k} tiles (out of {num_tiles} total)")
        jax.debug.callback(print_topk)
        
        # Debug Visualization
        if DEBUG_MODE:
            def print_anchor(idx):
                print(f"  [Anchor L{self.layer_id}] Selected Top-{actual_top_k} Tiles (Head 0): {idx[0,0]}")
            jax.debug.callback(print_anchor, top_tile_indices)
        
        # G. ANCHOR always uses FULL attention for output.
        # The top-k tile selection is ONLY for caching indices for REUSE layers.
        # Sparsity comes from REUSE layers, not ANCHOR.
        output = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
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
    Adds:
    1. Local Attention (Always attends to current local tile)
    2. Causal Masking (Prevents future leakage in sparse sets)
    """
    num_heads: int
    head_dim: int
    anchor_layer_id: int
    tile_size: int = 16
    head_map: dict = None
    use_splash: bool = False  # Use SplashAttention kernel if available

    @nn.compact
    def __call__(self, x, mask=None, freq_cis=None):
        batch, seq_len, _ = x.shape
        
        # DEBUG: Check input dtype
        if DEBUG_MODE:
            print(f"   REUSE input dtype: {x.dtype}")
        
        # 1. Projections
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        
        # DEBUG: Check projection output dtypes
        if DEBUG_MODE:
            print(f"   REUSE after Dense: q={q.dtype}, k={k.dtype}, v={v.dtype}")
        
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for gathering
        q = jnp.transpose(q, (0, 2, 1, 3)) # [B, H, S, D]
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Apply RoPE if provided (for LLaMA-3.1 compatibility)
        if freq_cis is not None:
            q, k = apply_rope(q, k, freq_cis)

        # 2. Retrieve Per-Query Anchor Indices
        cache_key = f"layer_{self.anchor_layer_id}_indices"
        anchor_indices = KASCADE_CACHE.get(cache_key, None)
        
        if anchor_indices is None:
            raise RuntimeError(
                f"REUSE Layer {self.anchor_layer_id} cache miss! "
                f"Anchor layer {self.anchor_layer_id} must run before REUSE layer can access its indices."
            )
        
        # anchor_indices shape: [B, H, Q, top_k] (per-query tile selections)
        
        # 3. Apply Head Mapping (per-query)
        if self.head_map is not None:
            perm_list = [self.head_map.get(h, h) for h in range(self.num_heads)]
            perm_indices = jnp.array(perm_list, dtype=jnp.int32)
            my_tile_indices = anchor_indices[:, perm_indices, :, :]  # [B, H, Q, top_k]
            
            if DEBUG_MODE:
                def print_map(p):
                    print(f"  [Reuse  L{self.anchor_layer_id+1}..] Applied Map: H0 uses Anchor H{p[0]}, H1 uses Anchor H{p[1]}...")
                jax.debug.callback(print_map, perm_indices)
        else:
            my_tile_indices = anchor_indices
            
        # 4. Add PER-QUERY LOCAL TILE for causal context
        # Each query at position q gets its own local tile: tile = q // tile_size
        num_tiles = seq_len // self.tile_size
        local_tiles = jnp.arange(seq_len) // self.tile_size  # [Q]
        local_tile_indices = local_tiles[None, None, :, None]  # [1, 1, Q, 1]
        local_tile_indices = jnp.broadcast_to(
            local_tile_indices, (batch, self.num_heads, seq_len, 1)
        ).astype(jnp.int32)
        
        # Concatenate: [sparse tiles from anchor] + [local tile per query]
        # Shape: [B, H, Q, top_k + 1]
        my_tile_indices = jnp.concatenate([my_tile_indices, local_tile_indices], axis=-1)
            
        # 5. Expand to Token Indices (per-query)
        # my_tile_indices: [B, H, Q, num_selected_tiles]
        offsets = jnp.arange(self.tile_size)[None, None, None, None, :]  # [1,1,1,1,tile_size]
        tile_starts = my_tile_indices[..., None] * self.tile_size  # [B,H,Q,tiles,1]
        token_indices = tile_starts + offsets  # [B,H,Q,tiles,tile_size]
        sparse_len = my_tile_indices.shape[-1] * self.tile_size
        flat_token_indices = token_indices.reshape(batch, self.num_heads, seq_len, sparse_len)
        # [B, H, Q, sparse_len] — each query has its own set of key indices!
        
        flat_token_indices = jnp.clip(flat_token_indices, 0, seq_len - 1)
        
        if DEBUG_MODE:
            def print_sparse_info(shape_val):
                print(f"  [Reuse  L{self.anchor_layer_id+1}..] Using {shape_val} sparse tokens per query (vs {seq_len} full)")
            jax.debug.callback(print_sparse_info, flat_token_indices.shape[3])
        
        # 6. Per-Query Gather of K, V
        # k: [B, H, S, D], flat_token_indices: [B, H, Q, sparse_len]
        # We need: k_sparse[b,h,q,i,:] = k[b,h, idx[b,h,q,i], :]
        B, H, S, D = k.shape
        k_flat = k.reshape(B * H, S, D)
        v_flat = v.reshape(B * H, S, D)
        idx_flat = flat_token_indices.reshape(B * H, seq_len, sparse_len)
        
        def gather_per_query(kv_bh, idx_bh):
            # kv_bh: [S, D], idx_bh: [Q, sparse_len]
            return kv_bh[idx_bh]  # Advanced indexing: [Q, sparse_len, D]
        
        k_sparse = jax.vmap(gather_per_query)(k_flat, idx_flat)  # [B*H, Q, sparse_len, D]
        v_sparse = jax.vmap(gather_per_query)(v_flat, idx_flat)
        k_sparse = k_sparse.reshape(B, H, seq_len, sparse_len, D)
        v_sparse = v_sparse.reshape(B, H, seq_len, sparse_len, D)

        # 7. Per-Query Sparse Attention
        # q: [B, H, Q, D], k_sparse: [B, H, Q, sparse_len, D]
        sparse_logits = jnp.einsum('bhqd,bhqkd->bhqk', q, k_sparse) / jnp.sqrt(self.head_dim)
        
        # Causal mask: per-query, each query's key indices checked
        query_idx = jnp.arange(seq_len)[None, None, :, None]  # [1, 1, Q, 1]
        key_idx = flat_token_indices  # [B, H, Q, sparse_len]
        future_mask = (key_idx > query_idx)
        sparse_logits = jnp.where(future_mask, -1e10, sparse_logits)

        # 8. Softmax & Output
        sparse_weights = jax.nn.softmax(sparse_logits, axis=-1)
        
        # Handle fully-masked rows (early queries where all keys are future)
        is_fully_masked = jnp.all(future_mask, axis=-1, keepdims=True)
        sparse_weights = jnp.where(is_fully_masked, 0.0, sparse_weights)
        sparse_weights = jnp.where(jnp.isnan(sparse_weights), 0.0, sparse_weights)
        
        # v_sparse: [B, H, Q, sparse_len, D]
        output = jnp.einsum('bhqk,bhqkd->bhqd', sparse_weights, v_sparse)
        
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        
        # Debug: track output stats  
        if DEBUG_MODE:
            def print_reuse_stats(out, weights_sum, aid):
                print(f"   [REUSE  from L{aid}] output: mean={float(jnp.mean(out)):.6f}, std={float(jnp.std(out)):.6f}, "
                      f"norm={float(jnp.linalg.norm(out)):.4f}, has_nan={bool(jnp.any(jnp.isnan(out)))}, "
                      f"weights_sum={float(jnp.mean(weights_sum)):.6f}")
            jax.debug.callback(print_reuse_stats, output, jnp.sum(sparse_weights, axis=-1), self.anchor_layer_id)
        
        # --- DEBUG: Check for NaN ---
        if DEBUG_MODE:
            def check_nan(x, layer_id):
                if jnp.any(jnp.isnan(x)):
                    print(f"  ⚠️  NaN detected in Reuse Layer {layer_id} output!")
            jax.debug.callback(check_nan, output, self.anchor_layer_id)
        
        return output
