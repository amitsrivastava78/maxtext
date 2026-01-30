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

# Global Cache to pass data between layers during Calibration & Inference
# Format: { "layer_0_indices": array([Batch, Heads, TopK_Tiles]) }
KASCADE_CACHE = {}

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

    @nn.compact
    def __call__(self, x, mask=None):
        batch, seq_len, _ = x.shape
        
        # 1. Standard Dense Attention
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False)(x)
        
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)
        
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
        # Shape: [Batch, Heads, Top_K]
        _, top_tile_indices = jax.lax.top_k(tile_scores, self.top_k_tiles)
        
        # F. Save to Cache (For Calibration or Reuse)
        cache_key = f"layer_{self.layer_id}_indices"
        KASCADE_CACHE[cache_key] = top_tile_indices
        
        # Debug Visualization
        def print_anchor(idx):
            print(f"  [Anchor L{self.layer_id}] Selected Top-{self.top_k_tiles} Tiles (Head 0): {idx[0,0]}")
        jax.debug.callback(print_anchor, top_tile_indices)
        
        # Finish Layer
        output = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
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

    @nn.compact
    def __call__(self, x, mask=None):
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

        # 2. Retrieve Anchor Indices
        cache_key = f"layer_{self.anchor_layer_id}_indices"
        anchor_indices = KASCADE_CACHE.get(cache_key, jnp.zeros((batch, self.num_heads, 2), dtype=jnp.int32))
        
        # 3. Apply Head Mapping
        if self.head_map is not None:
            perm_list = [self.head_map.get(h, h) for h in range(self.num_heads)]
            perm_indices = jnp.array(perm_list, dtype=jnp.int32)
            my_tile_indices = anchor_indices[:, perm_indices, :]
            
            # Debug Proof (Show the Shuffle)
            def print_map(p):
                print(f"  [Reuse  L{self.anchor_layer_id+1}..] Applied Map: H0 uses Anchor H{p[0]}, H1 uses Anchor H{p[1]}...")
            jax.debug.callback(print_map, perm_indices)
        else:
            my_tile_indices = anchor_indices

        # --- FIX 1: FORCE LOCAL ATTENTION ---
        # We assume the "current" tile is the last one in the sequence.
        # Calc current tile index: (seq_len - 1) // tile_size
        current_tile_idx = (seq_len - 1) // self.tile_size
        
        # We append this index to our list of indices to ensure we look at neighbors
        # (For simplicity in JAX static shapes, we replace the LAST fetched tile with the Local Tile)
        # In a real dynamic kernel, we would append +1 size.
        
        # Overwrite the last slot of top-k with the current tile index
        # shape: [Batch, Heads, TopK]
        my_tile_indices = my_tile_indices.at[:, :, -1].set(current_tile_idx)
            
        # 4. Expand to Tokens (Gather Logic)
        offsets = jnp.arange(self.tile_size)[None, None, None, :]
        tile_starts = my_tile_indices[..., None] * self.tile_size
        token_indices = tile_starts + offsets
        flat_token_indices = token_indices.reshape(batch, self.num_heads, -1)
        
        # Debug print to show sparse computation size (only prints shape, not traced values)
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
        output = jnp.einsum('bhqk,bhkd->bhqd', sparse_weights, v_sparse)
        
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = nn.Dense(x.shape[-1], use_bias=False)(output)
        
        return output
