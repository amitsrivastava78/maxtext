"""
Kascade Sparse Attention with Custom TPU Kernel
================================================
Implements Kascade sparse attention using a custom Pallas kernel optimized for TPU.

Architecture:
1. Tile Selection: Select important K/V tiles via calibration (ANCHOR) or reuse (REUSE)
2. Sparse Gather: Extract K/V from selected tiles (reduces to sparse_len << seq_len)
3. Custom Kernel: Optimized block-wise computation with online softmax

Custom Kernel Features (adapted from SplashAttention):
- Block-wise tiling for efficient TPU memory access
- Online softmax algorithm for numerical stability
- Fused operations to minimize memory bandwidth
- Optimized for arbitrary sparse patterns (not just causal/local)

Expected speedup: 2-3× on TPU from both sparsity AND kernel-level optimization
"""

import jax
import jax.numpy as jnp
import sys

# Import custom Kascade kernel
try:
    from MaxText.kernels.kascade_kernel import (
        make_kascade_kernel,
        KascadeBlockSizes,
        kascade_attention_reference,
    )
    KASCADE_KERNEL_AVAILABLE = True
except ImportError:
    KASCADE_KERNEL_AVAILABLE = False
    print("⚠️  Kascade custom kernel not available, falling back to JAX implementation")

# Cache for tile selections across layers
KASCADE_TILE_CACHE = {}


def gather_sparse_kv(K, V, tile_indices, tile_size):
    """
    Gather K and V from selected tiles to create sparse tensors.
    
    Args:
        K: [batch, seq_len, heads, head_dim]
        V: [batch, seq_len, heads, head_dim]
        tile_indices: [batch, heads, top_k] indices of selected tiles
        tile_size: Size of each tile
        
    Returns:
        K_sparse: [batch, sparse_len, heads, head_dim]
        V_sparse: [batch, sparse_len, heads, head_dim]
    """
    batch, seq_len, heads, head_dim = K.shape
    _, _, top_k = tile_indices.shape
    
    # Convert tile indices to token indices
    # Each tile index maps to a range [tile_idx * tile_size : (tile_idx+1) * tile_size]
    # We'll gather all tokens from selected tiles
    
    # Expand tile_indices to token indices: [batch, heads, top_k, tile_size]
    tile_idx_expanded = tile_indices[:, :, :, None]  # [B, H, K, 1]
    offsets = jnp.arange(tile_size)[None, None, None, :]  # [1, 1, 1, T]
    token_indices = tile_idx_expanded * tile_size + offsets  # [B, H, K, T]
    
    # Flatten to [batch, heads, sparse_len] where sparse_len = top_k * tile_size
    token_indices = token_indices.reshape(batch, heads, -1)  # [B, H, sparse_len]
    
    # Gather K and V
    # K/V are [batch, seq_len, heads, head_dim]
    # We need to gather along seq_len dimension for each head
    
    # Reshape for gathering: [batch, heads, seq_len, head_dim]
    K_transposed = jnp.transpose(K, (0, 2, 1, 3))
    V_transposed = jnp.transpose(V, (0, 2, 1, 3))
    
    # Gather: [batch, heads, sparse_len, head_dim]
    K_sparse = jnp.take_along_axis(K_transposed, token_indices[..., None], axis=2)
    V_sparse = jnp.take_along_axis(V_transposed, token_indices[..., None], axis=2)
    
    # Transpose back to [batch, sparse_len, heads, head_dim]
    K_sparse = jnp.transpose(K_sparse, (0, 2, 1, 3))
    V_sparse = jnp.transpose(V_sparse, (0, 2, 1, 3))
    
    return K_sparse, V_sparse


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
    
    # Reshape K into tiles: [batch, num_tiles, tile_size, heads, head_dim]
    K_tiled = K.reshape(batch, num_tiles, tile_size, heads, head_dim)
    
    # Max pool K tiles to get tile summary: [batch, num_tiles, heads, head_dim]
    K_summary = jnp.max(K_tiled, axis=2)
    
    # Compute Q @ K_summary for each head
    # Q: [batch, seq_len, heads, head_dim]
    # K_summary: [batch, num_tiles, heads, head_dim]
    tile_scores = jnp.einsum('bqhd,bnhd->bqhn', Q, K_summary)
    tile_scores = tile_scores / jnp.sqrt(head_dim)
    
    # Select top-k tiles per head (average over queries)
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
    Kascade sparse attention with custom TPU kernel.
    
    Strategy:
    1. Select important tiles via Kascade calibration (ANCHOR) or reuse (REUSE)
    2. Gather sparse K/V from those tiles (reduces FLOPs)
    3. Run custom kernel on Q × K_sparse (fast kernel execution)
    
    Args:
        query, key, value: Attention inputs [batch, seq_len, heads, head_dim]
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
    
    # Step 2: Gather sparse K/V from selected tiles
    key_sparse, value_sparse = gather_sparse_kv(key, value, tile_indices, tile_size)
    # key_sparse, value_sparse: [batch, sparse_len, heads, head_dim] where sparse_len << seq_len
    sparse_len = key_sparse.shape[1]
    
    # Step 3: Run custom kernel if available, otherwise use JAX reference
    if KASCADE_KERNEL_AVAILABLE:
        # Reshape for custom kernel: [batch, seq_len, heads, head_dim] -> [batch*heads, seq_len, head_dim]
        q_reshaped = query.reshape(batch * heads, seq_len, head_dim)
        k_sparse_reshaped = key_sparse.reshape(batch * heads, sparse_len, head_dim)
        v_sparse_reshaped = value_sparse.reshape(batch * heads, sparse_len, head_dim)
        
        # Transpose to [num_heads, seq_len, head_dim] format expected by kernel
        q_transposed = jnp.transpose(q_reshaped, (0, 1, 2))  # Already correct order
        k_sparse_transposed = jnp.transpose(k_sparse_reshaped, (0, 1, 2))
        v_sparse_transposed = jnp.transpose(v_sparse_reshaped, (0, 1, 2))
        
        # Configure block sizes
        block_sizes = KascadeBlockSizes(
            block_q=min(512, seq_len),
            block_kv_sparse=min(256, sparse_len),
            block_kv_compute=min(128, sparse_len),
        )
        
        # Create and call kernel
        kernel_fn = make_kascade_kernel(block_sizes)
        output_transposed = kernel_fn(q_transposed, k_sparse_transposed, v_sparse_transposed)
        
        # Reshape back: [batch*heads, seq_len, head_dim] -> [batch, seq_len, heads, head_dim]
        output = output_transposed.reshape(batch, heads, seq_len, head_dim)
        output = jnp.transpose(output, (0, 2, 1, 3))  # -> [batch, seq_len, heads, head_dim]
    else:
        # Fallback: Use JAX reference implementation
        # Transpose to [batch, heads, seq_len, head_dim]
        q_t = jnp.transpose(query, (0, 2, 1, 3))
        k_sparse_t = jnp.transpose(key_sparse, (0, 2, 1, 3))
        v_sparse_t = jnp.transpose(value_sparse, (0, 2, 1, 3))
        
        # Attention: Q @ K^T @ V
        logits = jnp.einsum('bhqd,bhkd->bhqk', q_t, k_sparse_t) / jnp.sqrt(head_dim)
        weights = jax.nn.softmax(logits, axis=-1)
        output_t = jnp.einsum('bhqk,bhkd->bhqd', weights, v_sparse_t)
        
        # Transpose back
        output = jnp.transpose(output_t, (0, 2, 1, 3))  # -> [batch, seq_len, heads, head_dim]
    
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
    'gather_sparse_kv',
    'create_kascade_splash_schedule',
    'KASCADE_TILE_CACHE',
    'KASCADE_KERNEL_AVAILABLE',
]
