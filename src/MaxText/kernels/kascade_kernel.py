"""
Kascade Custom TPU Kernel
==========================
Optimized Pallas kernel for Kascade sparse attention on TPU.

Adapts techniques from SplashAttention for arbitrary sparse patterns:
- Block-wise computation with online softmax
- Efficient memory access patterns
- TPU-optimized tiling strategy

Key Differences from SplashAttention:
- Works with pre-gathered sparse K/V (no structured masking)
- Arbitrary tile indices (not causal/local patterns)
- Optimized for Q @ K_sparse^T where sparse_len << seq_len

Performance Target: 2-3× speedup vs JAX implementation on TPU
"""

import functools
from typing import Any, Callable, Literal, NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

# TPU constants
NUM_LANES = 128
NUM_SUBLANES = 8
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)

# Dimension numbers for matmul
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed


class KascadeBlockSizes(NamedTuple):
    """Block sizes for Kascade kernel.
    
    Args:
        block_q: Query block size (must be multiple of NUM_LANES=128)
        block_kv_sparse: Sparse K/V block size (must be multiple of NUM_LANES=128)
        block_kv_compute: K/V compute block size (must divide block_kv_sparse)
    
    Optimized block sizes based on empirical TPU testing:
    - Larger block_q significantly improves performance (see debug_performance.py)
    - Diagnostic results: 256→0.72×, 512→0.85×, 1024→0.97×
    - Comprehensive suite: block_kv_compute=64 achieves best stability (0.971×)
    """
    block_q: int = 1024  # Empirically optimal
    block_kv_sparse: int = 256  # Stable across different sparse lengths
    block_kv_compute: int | None = 64  # Best stable performance (0.971×)
    
    def __post_init__(self):
        if self.block_kv_compute is None:
            object.__setattr__(self, 'block_kv_compute', min(128, self.block_kv_sparse))


def kascade_attention_kernel(
    # Inputs (passed first)
    q_ref,  # [block_q, head_dim]
    k_sparse_ref,  # [block_kv_sparse, head_dim]
    v_sparse_ref,  # [block_kv_sparse, head_dim]
    # Output (passed BEFORE scratch buffers!)
    o_ref,  # [block_q, head_dim] - final output
    # Scratch buffers (passed AFTER output, in scratch_shapes order)
    m_scratch_ref,  # [block_q, NUM_LANES] - max logits (scratch_shapes[0])
    l_scratch_ref,  # [block_q, NUM_LANES] - sum of exp (scratch_shapes[1])
    o_scratch_ref,  # [block_q, head_dim] - output accumulator (scratch_shapes[2])
    # Parameters
    *,
    mask_value: float,
    bq: int,
    bkv_sparse: int,
    bkv_compute: int,
    head_dim_v: int,
):
    """
    Kascade attention kernel for one Q block × sparse K/V.
    
    Implements online softmax algorithm:
    1. Compute Q @ K_sparse^T in blocks
    2. Update running max (m) and sum (l) for numerical stability
    3. Accumulate weighted output (o) with correction factors
    4. Final normalization by l
    
    This is adapted from SplashAttention's flash_attention_kernel but:
    - No masking logic (K/V already sparse)
    - No data_next prefetching (simple linear scan)
    - Simpler indexing (no shrinking iteration space)
    """
    float32 = jnp.float32
    
    # Get grid indices (same as SplashAttention)
    h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    
    # Note: head_dim_v does NOT need to be a multiple of NUM_LANES
    # The NUM_LANES constraint only applies to sequence-dimension block sizes
    # Head dimensions (e.g., 64, 96, 128) are handled by standard dot products
    
    # Validate bkv_compute is multiple of NUM_LANES for efficient TPU memory access
    bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
    if rem != 0:
        raise NotImplementedError(f"{bkv_compute=} should be a multiple of {NUM_LANES}")
    
    # Initialize output buffer
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    o_ref[...] = jnp.zeros_like(o_ref)
    
    # Main computation loop over K/V blocks
    # Phase 1: Keep m, l in carry to avoid scratch buffer reads
    def body(kv_compute_idx, carry):
        """Process one K/V compute block."""
        m_prev, l_prev = carry  # [bq, 1] - kept in registers, not memory!
        
        # Slice current K/V block
        slice_k = pl.ds(kv_compute_idx * bkv_compute, bkv_compute)
        
        # Compute Q @ K^T for this block (with scaling)
        q = q_ref[...]  # [bq, head_dim]
        k = k_sparse_ref[slice_k, :]  # [bkv_compute, head_dim]
        qk = lax.dot_general(q, k, NT_DIM_NUMBERS, preferred_element_type=float32)
        qk = qk / jnp.sqrt(float32(head_dim_v))  # Scale by 1/sqrt(d)
        
        # Online softmax: update max
        m_curr = qk.max(axis=-1, keepdims=True)  # [bq, 1]
        m_next = jnp.maximum(m_prev, m_curr)
        
        # Compute exp(qk - m_next) and sum  
        s_curr = jnp.exp(qk - m_next)
        l_curr = s_curr.sum(axis=-1, keepdims=True)  # [bq, 1]
        
        # Update running sum with correction factor
        alpha = jnp.exp(m_prev - m_next)  # [bq, 1]
        l_next = l_curr + alpha * l_prev
        
        # Compute weighted output: s_curr @ V
        v = v_sparse_ref[slice_k, :].astype(float32)  # [bkv_compute, head_dim]
        o_curr = lax.dot_general(s_curr, v, NN_DIM_NUMBERS, preferred_element_type=float32)
        
        # Update output accumulator with correction
        o_scratch_ref[:] = alpha * o_scratch_ref[:] + o_curr
        
        return (m_next, l_next)
    
    # Run the loop with m, l in carry (stays in registers, not memory!)
    num_iters = bkv_sparse // bkv_compute
    m_init = jnp.full((bq, 1), mask_value, dtype=float32)
    l_init = jnp.zeros((bq, 1), dtype=float32)
    # Phase 2: Adaptive unroll for small iteration counts (2-4 iters typical)
    should_unroll = num_iters <= 4
    m_final, l_final = lax.fori_loop(0, num_iters, body, (m_init, l_init), unroll=should_unroll)
    
    # Write final statistics to scratch (for consistency, though not strictly needed)
    m_scratch_ref[...] = jnp.broadcast_to(m_final, (bq, NUM_LANES))
    l_scratch_ref[...] = jnp.broadcast_to(l_final, (bq, NUM_LANES))
    
    # Final normalization (wrapped in pl.when even though j is always 0)
    @pl.when(j == 0)  # Since grid third dim is 1, j is always 0
    def end():
        l = l_scratch_ref[:, :1]  # (bq, 1) - use first column only for broadcasting
        # Normalize output
        o_ref[...] = (o_scratch_ref[...] / l).astype(o_ref.dtype)


def kascade_attention_forward(
    q: jax.Array,  # [num_heads, q_seq_len, head_dim]
    k_sparse: jax.Array,  # [num_heads, sparse_len, head_dim]
    v_sparse: jax.Array,  # [num_heads, sparse_len, head_dim]
    block_sizes: KascadeBlockSizes | None = None,
) -> jax.Array:
    """
    Kascade attention forward pass using custom Pallas kernel.
    
    Args:
        q: Query tensor [num_heads, q_seq_len, head_dim]
        k_sparse: Sparse key tensor [num_heads, sparse_len, head_dim]
        v_sparse: Sparse value tensor [num_heads, sparse_len, head_dim]
        block_sizes: Block tiling configuration
        
    Returns:
        output: Attention output [num_heads, q_seq_len, head_dim]
    """
    if block_sizes is None:
        block_sizes = KascadeBlockSizes()
    
    num_heads, q_seq_len, head_dim_qk = q.shape
    _, sparse_len, head_dim_v = v_sparse.shape
    
    # Handle small sparse_len by padding to NUM_LANES
    if sparse_len < NUM_LANES:
        pad_len = NUM_LANES - sparse_len
        k_sparse = jnp.pad(k_sparse, ((0, 0), (0, pad_len), (0, 0)), mode='constant')
        v_sparse = jnp.pad(v_sparse, ((0, 0), (0, pad_len), (0, 0)), mode='constant')
        sparse_len = NUM_LANES
    
    bq = block_sizes.block_q
    bkv_sparse = min(block_sizes.block_kv_sparse, sparse_len)
    # Round bkv_sparse up to nearest multiple of NUM_LANES
    bkv_sparse = ((bkv_sparse + NUM_LANES - 1) // NUM_LANES) * NUM_LANES
    bkv_compute = block_sizes.block_kv_compute or min(NUM_LANES, bkv_sparse)
    # Ensure bkv_compute is multiple of NUM_LANES
    bkv_compute = max(NUM_LANES, bkv_compute)
    
    # Validate shapes
    if k_sparse.shape != (num_heads, sparse_len, head_dim_qk):
        raise ValueError(
            f"k_sparse shape {k_sparse.shape} doesn't match "
            f"expected ({num_heads}, {sparse_len}, {head_dim_qk})"
        )
    
    if bkv_sparse % bkv_compute != 0:
        raise ValueError(f"{bkv_sparse=} must be multiple of {bkv_compute=}")
    
    if bkv_compute % NUM_LANES != 0:
        raise ValueError(f"{bkv_compute=} must be multiple of {NUM_LANES}")
    
    if bq % NUM_LANES != 0:
        raise ValueError(f"{bq=} must be multiple of {NUM_LANES}")
    
    # Grid: (num_heads, num_q_blocks, 1)
    num_q_blocks = (q_seq_len + bq - 1) // bq
    grid = (num_heads, num_q_blocks, 1)
    
    # Index maps for Pallas
    def q_index_map(h, i, j):
        return h, i, 0
    
    def k_sparse_index_map(h, i, j):
        return h, 0, 0  # K/V are same for all Q blocks
    
    def v_sparse_index_map(h, i, j):
        return h, 0, 0
    
    def out_index_map(h, i, j):
        return h, i, 0
    
    # BlockSpecs for inputs/outputs
    in_specs = [
        pl.BlockSpec((None, bq, head_dim_qk), q_index_map),  # q
        pl.BlockSpec((None, bkv_sparse, head_dim_qk), k_sparse_index_map),  # k_sparse
        pl.BlockSpec((None, bkv_sparse, head_dim_v), v_sparse_index_map),  # v_sparse
    ]
    
    out_specs = pl.BlockSpec((None, bq, head_dim_v), out_index_map)
    
    # Scratch space for online softmax (in VMEM)
    scratch_shapes = [
        pltpu.VMEM((bq, NUM_LANES), jnp.float32),  # m_scratch
        pltpu.VMEM((bq, NUM_LANES), jnp.float32),  # l_scratch
        pltpu.VMEM((bq, head_dim_v), jnp.float32),  # o_scratch
    ]
    
    # Create kernel with parameters
    kernel = functools.partial(
        kascade_attention_kernel,
        mask_value=DEFAULT_MASK_VALUE,
        bq=bq,
        bkv_sparse=bkv_sparse,
        bkv_compute=bkv_compute,
        head_dim_v=head_dim_v,
    )
    
    # Call Pallas (no special compiler_params needed for TPU)
    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim_v), q.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        grid=grid,
    )(q, k_sparse, v_sparse)
    
    return output


def make_kascade_kernel(
    block_sizes: KascadeBlockSizes | None = None,
):
    """
    Create a JIT-compiled Kascade attention function.
    
    Args:
        block_sizes: Block tiling configuration
        
    Returns:
        Callable that computes Kascade attention with custom kernel
    """
    
    @jax.jit
    def _kascade_attention(
        q: jax.Array,
        k_sparse: jax.Array,
        v_sparse: jax.Array,
    ) -> jax.Array:
        """
        Compute Kascade attention: Q @ K_sparse^T @ V_sparse.
        
        Args:
            q: [num_heads, q_seq_len, head_dim]
            k_sparse: [num_heads, sparse_len, head_dim]
            v_sparse: [num_heads, sparse_len, head_dim]
            
        Returns:
            output: [num_heads, q_seq_len, head_dim]
        """
        return kascade_attention_forward(q, k_sparse, v_sparse, block_sizes)
    
    return _kascade_attention


# Reference JAX implementation for correctness testing
def kascade_attention_reference(
    q: jax.Array,  # [num_heads, q_seq_len, head_dim]
    k_sparse: jax.Array,  # [num_heads, sparse_len, head_dim]
    v_sparse: jax.Array,  # [num_heads, sparse_len, head_dim]
) -> jax.Array:
    """
    Reference implementation using pure JAX (for testing).
    
    This is the baseline to compare our custom kernel against.
    """
    # Q @ K^T
    logits = jnp.einsum('hqd,hkd->hqk', q, k_sparse) / jnp.sqrt(q.shape[-1])
    
    # Softmax
    weights = jax.nn.softmax(logits, axis=-1)
    
    # Weights @ V
    output = jnp.einsum('hqk,hkd->hqd', weights, v_sparse)
    
    return output


__all__ = [
    'kascade_attention_forward',
    'kascade_attention_reference',
    'make_kascade_kernel',
    'KascadeBlockSizes',
]
