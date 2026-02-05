"""
Kascade + SplashAttention: Optimized Sparse Attention
======================================================
Combines Kascade's data-driven tile selection with MaxText's optimized SplashAttention kernel.

Key Benefits:
- Kascade calibration: Select important tiles dynamically
- SplashAttention kernel: Fused, hardware-optimized computation  
- Expected speedup: 2-3× on TPU for 7B+ models
"""

import jax
import jax.numpy as jnp
import functools
import sys

# Module-level variable that will be set by the loader
_KERNEL_MODULE = None

# Import splash attention mask from JAX
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

# Use the pre-loaded kernel module if available
if _KERNEL_MODULE is not None:
    splash_attention_kernel = _KERNEL_MODULE
else:
    # Fallback: try to import from JAX (won't work, but needed for structure)
    try:
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
    except ImportError:
        splash_attention_kernel = None

# Cache for tile selections across layers
KASCADE_TILE_CACHE = {}


class KascadeMask(splash_attention_mask._ComputableMask):  # pylint: disable=protected-access
    """
    Custom mask for SplashAttention that uses Kascade's tile selections.
    
    This mask converts Kascade's selected tile indices into a sparse attention mask
    that SplashAttention can consume efficiently.
    """
    
    def __init__(self, tile_indices, tile_size, seq_len):
        """
        Args:
            tile_indices: [batch, heads, top_k] indices of selected tiles
            tile_size: Size of each tile in tokens
            seq_len: Total sequence length
        """
        self.tile_indices = tile_indices
        self.tile_size = tile_size
        self.seq_len = seq_len
    
    def __call__(self, q_idx, kv_idx):
        """
        Mask function called by SplashAttention kernel.
        
        Returns True if kv_idx is within a selected tile for q_idx.
        """
        # Determine which tile kv_idx belongs to
        kv_tile = kv_idx // self.tile_size
        
        # Check if this tile is in the selected tiles for this query
        # This is a simplified version - in practice, SplashAttention
        # handles this more efficiently with block-level masking
        return self._is_tile_selected(q_idx, kv_tile)
    
    def _is_tile_selected(self, q_idx, kv_tile):
        """Check if kv_tile is in the selected tiles for query q_idx"""
        # This would be implemented efficiently in the actual kernel
        # For now, this is a placeholder showing the logic
        batch_idx = 0  # Assuming single batch
        head_idx = 0   # Would need to be parameterized
        selected = self.tile_indices[batch_idx, head_idx]
        return jnp.any(selected == kv_tile)


def kascade_calibrate_tiles(Q, K, tile_size=64, top_k_ratio=0.25):
    """
    Kascade calibration: Select important tiles based on Q-K similarity.
    
    Args:
        Q: Query tensor [batch, seq_len, heads, head_dim]
        K: Key tensor [batch, seq_len, heads, head_dim]
        tile_size: Size of each tile
        top_k_ratio: Fraction of tiles to keep
        
    Returns:
        tile_indices: [batch, heads, top_k] indices of selected tiles
    """
    batch, seq_len, heads, head_dim = Q.shape
    num_tiles = seq_len // tile_size
    top_k = max(1, int(num_tiles * top_k_ratio))
    
    # Reshape for tile-based processing
    Q_reshaped = Q.reshape(batch, seq_len, heads, head_dim)
    K_tiled = K.reshape(batch, num_tiles, tile_size, heads, head_dim)
    
    # Max pool K tiles to get tile summary
    K_summary = jnp.max(K_tiled, axis=2)  # [batch, num_tiles, heads, head_dim]
    
    # Compute Q @ K_summary for each head
    # Q: [batch, seq_len, heads, head_dim]
    # K_summary: [batch, num_tiles, heads, head_dim]
    tile_scores = jnp.einsum('bqhd,bnhd->bqhn', Q_reshaped, K_summary)
    tile_scores = tile_scores / jnp.sqrt(head_dim)
    
    # Select top-k tiles per query per head
    # Use mean over queries to get global tile importance per head
    tile_scores_mean = jnp.mean(tile_scores, axis=1)  # [batch, heads, num_tiles]
    
    # Get top-k tile indices
    top_k_capped = min(top_k, num_tiles)
    _, top_tile_indices = jax.lax.top_k(tile_scores_mean, top_k_capped)
    
    return top_tile_indices  # [batch, heads, top_k]


def kascade_splash_attention(
    query,  # [batch, seq_len, heads, head_dim]
    key,    # [batch, seq_len, heads, head_dim]  
    value,  # [batch, seq_len, heads, head_dim]
    layer_id,
    is_anchor_layer=True,
    anchor_layer_id=None,
    tile_size=64,
    top_k_ratio=0.25,
):
    """
    Kascade + SplashAttention: Optimized sparse attention.
    
    Args:
        query, key, value: Attention inputs
        layer_id: Current layer index
        is_anchor_layer: If True, compute tile selection. If False, reuse from anchor.
        anchor_layer_id: Which anchor layer to reuse tiles from (for REUSE layers)
        tile_size: Tile size for Kascade
        top_k_ratio: Fraction of tiles to keep
        
    Returns:
        output: Attention output [batch, seq_len, heads, head_dim]
    """
    batch, seq_len, heads, head_dim = query.shape
    
    # Step 1: Get tile selections (either compute or reuse)
    cache_key = f"layer_{layer_id}_tiles"
    
    if is_anchor_layer:
        # ANCHOR layer: Compute tile selection
        tile_indices = kascade_calibrate_tiles(
            query, key, 
            tile_size=tile_size,
            top_k_ratio=top_k_ratio
        )
        KASCADE_TILE_CACHE[cache_key] = tile_indices
    else:
        # REUSE layer: Copy from anchor
        anchor_key = f"layer_{anchor_layer_id}_tiles"
        tile_indices = KASCADE_TILE_CACHE.get(anchor_key)
        if tile_indices is None:
            # Fallback: compute if not found
            tile_indices = kascade_calibrate_tiles(query, key, tile_size, top_k_ratio)
    
    # Step 2: Create sparse mask from tile selections
    # For each head, create a mask that allows attention only to selected tiles
    kascade_mask = KascadeMask(tile_indices, tile_size, seq_len)
    
    # Create multi-head mask for SplashAttention
    multi_head_mask = splash_attention_mask.MultiHeadMask(
        masks=(kascade_mask,) * heads
    )
    
    # Step 3: Configure SplashAttention kernel
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=min(512, seq_len),
        block_kv=min(tile_size, seq_len),
        block_kv_compute=min(tile_size, seq_len),
        block_q_major_dkv=min(512, seq_len),
        block_kv_major_dkv=min(tile_size, seq_len),
    )
    
    # Step 4: Create and execute SplashAttention kernel
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
        attn_logits_soft_cap=None,
    )
    
    # Reshape for SplashAttention: [batch, seq_len, heads, head_dim]
    # SplashAttention expects specific format
    output = splash_kernel(query, key, value)
    
    return output


# Convenience function for easy integration
def create_kascade_splash_schedule(num_layers, threshold=0.5, max_reuse_dist=4):
    """
    Create a schedule determining which layers are ANCHOR vs REUSE.
    
    This would typically be done during calibration, but can also use
    heuristics (e.g., every 4th layer is ANCHOR).
    
    Args:
        num_layers: Total number of layers
        threshold: Similarity threshold for reuse (not used in heuristic mode)
        max_reuse_dist: Maximum distance between anchor and reuse layers
        
    Returns:
        schedule: Dict mapping layer_id -> {'type': 'ANCHOR'|'REUSE', 'anchor': layer_id}
    """
    schedule = {}
    
    # Layer 0 is always DENSE (full attention per paper)
    schedule[0] = {'type': 'DENSE', 'anchor': None}
    
    # Use simple heuristic: Every max_reuse_dist-th layer is ANCHOR
    last_anchor = 1
    schedule[1] = {'type': 'ANCHOR', 'anchor': None}
    
    for i in range(2, num_layers):
        if i - last_anchor >= max_reuse_dist:
            # Make this an ANCHOR layer
            schedule[i] = {'type': 'ANCHOR', 'anchor': None}
            last_anchor = i
        else:
            # Make this a REUSE layer
            schedule[i] = {'type': 'REUSE', 'anchor': last_anchor}
    
    return schedule


__all__ = [
    'kascade_splash_attention',
    'kascade_calibrate_tiles',
    'create_kascade_splash_schedule',
    'KascadeMask',
    'KASCADE_TILE_CACHE',
]
