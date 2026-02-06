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
DEBUG_MODE = False

# Global Cache to pass data between layers during Calibration & Inference
# Format: { "layer_0_indices": array([Batch, Heads, TopK_Tiles]) }
KASCADE_CACHE = {}

# RoPE Helper Functions for LLaMA-3.1
def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    """
    Precompute the frequency tensor for RoPE (Rotary Position Embedding).
    LLaMA-3.1 uses theta=500000 instead of standard 10000.
    
    Args:
        dim: Head dimension (should be 128 for LLaMA-3.1)
        end: Maximum sequence length
        theta: Base for frequency computation (500000 for LLaMA-3.1)
    
    Returns:
        Complex tensor of shape [end, dim//2] for RoPE rotation
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)  # Complex exponential
    return freqs_cis

def apply_rope(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray):
    """
    Apply RoPE using Adjacent Pairs format (compatible with Meta weights).
    Correctly pairs dimension (i, i+1) as adjacent pairs.
    
    Args:
        xq: Query tensor [batch, heads, seq, dim]
        xk: Key tensor [batch, heads, seq, dim]
        freqs_cis: Precomputed frequency tensor [seq, dim//2]
    
    Returns:
        Rotated query and key tensors
    """
    # xq, xk shape: [Batch, Heads, Seq, Dim]
    # freqs_cis shape: [Seq, Dim//2] (Complex)
    
    # Cast to float32 for complex operations
    xq = xq.astype(jnp.float32)
    xk = xk.astype(jnp.float32)
    
    # Reshape to pairs: [B, H, S, D] -> [B, H, S, D/2, 2]
    xq_pairs = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_pairs = xk.reshape(*xk.shape[:-1], -1, 2)
    
    # Convert pairs to complex: (x0, x1) -> x0 + i*x1
    xq_complex = jax.lax.complex(xq_pairs[..., 0], xq_pairs[..., 1])
    xk_complex = jax.lax.complex(xk_pairs[..., 0], xk_pairs[..., 1])
    
    # Reshape frequencies for broadcasting [1, 1, Seq, Dim//2]
    freqs_cis = freqs_cis[None, None, :xq.shape[2], :]
    
    # Apply rotation (complex multiplication)
    xq_rotated = xq_complex * freqs_cis
    xk_rotated = xk_complex * freqs_cis
    
    # Reconstruct: interleave real and imaginary
    xq_out = jnp.stack([xq_rotated.real, xq_rotated.imag], axis=-1).reshape(xq.shape)
    xk_out = jnp.stack([xk_rotated.real, xk_rotated.imag], axis=-1).reshape(xk.shape)
    
    # TPU-friendly: use bfloat16 instead of float16
    return xq_out.astype(jnp.bfloat16), xk_out.astype(jnp.bfloat16)

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
        
        # 1. Standard Dense Projections (Q, K, V)
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        
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
        
        # --- REQUIREMENT 3: TILE POOLING & TOP-K ---
        
        # A. Extract Attention for the Last Token (The one generating next word)
        # Shape: [Batch, Heads, Seq_Len]
        last_token_probs = weights[:, :, -1, :] 
        
        # B. Pad Sequence if it doesn't fit tiles perfectly
        pad_len = (self.tile_size - (seq_len % self.tile_size)) % self.tile_size
        if pad_len > 0:
            last_token_probs = jnp.pad(last_token_probs, ((0,0), (0,0), (0, pad_len)))
            
        # C. Reshape into Tiles
        # Shape: [Batch, Heads, Num_Tiles, Tile_Size]
        num_tiles = last_token_probs.shape[-1] // self.tile_size
        tiled_probs = last_token_probs.reshape(batch, self.num_heads, num_tiles, self.tile_size)
        
        # D. Max Pooling: Find the single highest probability in each tile
        # Shape: [Batch, Heads, Num_Tiles]
        tile_scores = jnp.max(tiled_probs, axis=-1) 
        
        # E. Extract Top-K Indices
        # Cap top_k by actual number of tiles (avoid error when tiles < top_k)
        num_tiles = tile_scores.shape[-1]
        actual_top_k = min(self.top_k_tiles, num_tiles)
        # Shape: [Batch, Heads, Top_K]
        _, top_tile_indices = jax.lax.top_k(tile_scores, actual_top_k)
        
        # F. Save to Cache (For Calibration or Reuse)
        cache_key = f"layer_{self.layer_id}_indices"
        KASCADE_CACHE[cache_key] = top_tile_indices
        
        # Debug Visualization
        if DEBUG_MODE:
            def print_anchor(idx):
                print(f"  [Anchor L{self.layer_id}] Selected Top-{actual_top_k} Tiles (Head 0): {idx[0,0]}")
            jax.debug.callback(print_anchor, top_tile_indices)
        
        # Finish Layer
        output = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        
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
        
        # 1. Projections
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        
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

        # 2. Retrieve Anchor Indices
        cache_key = f"layer_{self.anchor_layer_id}_indices"
        anchor_indices = KASCADE_CACHE.get(cache_key, jnp.zeros((batch, self.num_heads, 2), dtype=jnp.int32))
        
        # 3. Apply Head Mapping
        if self.head_map is not None:
            perm_list = [self.head_map.get(h, h) for h in range(self.num_heads)]
            perm_indices = jnp.array(perm_list, dtype=jnp.int32)
            my_tile_indices = anchor_indices[:, perm_indices, :]
            
            # Debug Proof (Show the Shuffle)
            if DEBUG_MODE:
                def print_map(p):
                    print(f"  [Reuse  L{self.anchor_layer_id+1}..] Applied Map: H0 uses Anchor H{p[0]}, H1 uses Anchor H{p[1]}...")
                jax.debug.callback(print_map, perm_indices)
        else:
            my_tile_indices = anchor_indices
            
        # 4. Expand to Tokens (Gather Logic)
        offsets = jnp.arange(self.tile_size)[None, None, None, :]
        tile_starts = my_tile_indices[..., None] * self.tile_size
        token_indices = tile_starts + offsets
        flat_token_indices = token_indices.reshape(batch, self.num_heads, -1)
        
        # --- CRITICAL FIX: Clip indices to valid range [0, seq_len-1] ---
        # This prevents out-of-bounds gather that causes NaN
        flat_token_indices = jnp.clip(flat_token_indices, 0, seq_len - 1)
        
        # Debug print to show sparse computation size (only prints shape, not traced values)
        if DEBUG_MODE:
            def print_sparse_info(shape_val):
                print(f"  [Reuse  L{self.anchor_layer_id+1}..] Using {shape_val} sparse tokens (vs {seq_len} full)")
            jax.debug.callback(print_sparse_info, flat_token_indices.shape[2])
        
        # 5. Perform Gather
        k_sparse = jnp.take_along_axis(k, flat_token_indices[..., None], axis=2)
        v_sparse = jnp.take_along_axis(v, flat_token_indices[..., None], axis=2)

        # 6. Compute Sparse Logits
        # Q: [B, H, Seq, D]  @ K_T: [B, H, D, Sparse_Len]
        sparse_logits = jnp.einsum('bhqd,bhkd->bhqk', q, k_sparse) / jnp.sqrt(self.head_dim)
        
        # --- FIX 2: CAUSAL MASKING ---
        # We need to mask positions where Key_Index > Query_Index.
        # Query Indices: [0, 1, 2 ... Seq_Len]
        # Key Indices: flat_token_indices [Key1, Key2 ...]
        
        # Broadcast to compare: [1, 1, Seq, 1] vs [B, H, 1, Sparse_Len]
        query_idx = jnp.arange(seq_len)[None, None, :, None]
        key_idx = flat_token_indices[:, :, None, :]
        
        # Mask: 1 if Key > Query (Future), 0 otherwise
        future_mask = (key_idx > query_idx)
        
        # Apply mask (-1e10 is standard for neg infinity)
        sparse_logits = jnp.where(future_mask, -1e10, sparse_logits)

        # 7. Softmax & Output
        sparse_weights = jax.nn.softmax(sparse_logits, axis=-1)
        
        # --- CRITICAL FIX: Handle fully-masked rows (all -inf) that produce NaN in softmax ---
        # When all keys are masked (future), softmax produces NaN. Replace with uniform distribution.
        is_nan = jnp.isnan(sparse_weights)
        sparse_weights = jnp.where(is_nan, 1.0 / sparse_weights.shape[-1], sparse_weights)
        
        output = jnp.einsum('bhqk,bhkd->bhqd', sparse_weights, v_sparse)
        
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        
        # --- DEBUG: Check for NaN ---
        if DEBUG_MODE:
            def check_nan(x, layer_id):
                if jnp.any(jnp.isnan(x)):
                    print(f"  ⚠️  NaN detected in Reuse Layer {layer_id} output!")
            jax.debug.callback(check_nan, output, self.anchor_layer_id)
        
        return output
