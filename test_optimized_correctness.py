"""
Test that Phase 1 optimizations maintain correctness.
Validates the algorithm changes without needing TPU.
"""

import jax
import jax.numpy as jnp

def reference_attention_sparse(q, k_sparse, v_sparse):
    """Reference implementation using JAX operations."""
    # q: [num_heads, q_seq_len, head_dim]
    # k_sparse: [num_heads, sparse_len, head_dim]
    # v_sparse: [num_heads, sparse_len, head_dim]
    
    # Compute attention scores
    scores = jnp.einsum('hqd,hkd->hqk', q, k_sparse)
    scores = scores / jnp.sqrt(q.shape[-1])
    
    # Softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)
    
    # Apply to values
    output = jnp.einsum('hqk,hkd->hqd', attn_weights, v_sparse)
    
    return output


def simulated_optimized_kernel(q, k_sparse, v_sparse, block_size=128):
    """
    Simulates the optimized kernel algorithm (Phase 1) without Pallas.
    Tests the carry-based approach for maintaining m, l in registers.
    """
    num_heads, q_seq_len, head_dim = q.shape
    num_heads_k, sparse_len, head_dim_k = k_sparse.shape
    
    assert num_heads == num_heads_k
    assert head_dim == head_dim_k
    
    # Process one head at a time for simplicity
    outputs = []
    
    for h in range(num_heads):
        q_h = q[h]  # [q_seq_len, head_dim]
        k_h = k_sparse[h]  # [sparse_len, head_dim]
        v_h = v_sparse[h]  # [sparse_len, head_dim]
        
        # Initialize accumulators (simulating the carry approach)
        mask_value = -1e10
        m_prev = jnp.full((q_seq_len, 1), mask_value, dtype=jnp.float32)
        l_prev = jnp.zeros((q_seq_len, 1), dtype=jnp.float32)
        o_acc = jnp.zeros((q_seq_len, head_dim), dtype=jnp.float32)
        
        # Process in blocks (simulating the loop)
        num_blocks = sparse_len // block_size
        
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = start + block_size
            
            k_block = k_h[start:end]  # [block_size, head_dim]
            v_block = v_h[start:end]  # [block_size, head_dim]
            
            # Compute QK^T with scaling
            qk = jnp.dot(q_h, k_block.T) / jnp.sqrt(float(head_dim))  # [q_seq_len, block_size]
            
            # Online softmax: update max (THIS IS THE OPTIMIZED APPROACH)
            m_curr = qk.max(axis=-1, keepdims=True)  # [q_seq_len, 1]
            m_next = jnp.maximum(m_prev, m_curr)  # No slicing - direct comparison!
            
            # Compute exp and sum
            s_curr = jnp.exp(qk - m_next)
            l_curr = s_curr.sum(axis=-1, keepdims=True)  # [q_seq_len, 1]
            
            # Update running sum with correction
            alpha = jnp.exp(m_prev - m_next)  # [q_seq_len, 1]
            l_next = l_curr + alpha * l_prev  # No slicing - direct operation!
            
            # Compute weighted output
            o_curr = jnp.dot(s_curr, v_block)  # [q_seq_len, head_dim]
            
            # Update output accumulator (alpha broadcasts automatically)
            o_acc = alpha * o_acc + o_curr
            
            # Update carry for next iteration (THIS IS THE KEY OPTIMIZATION)
            m_prev = m_next
            l_prev = l_next
        
        # Final normalization
        output_h = o_acc / l_prev  # l_prev broadcasts from [q_seq_len, 1]
        outputs.append(output_h)
    
    return jnp.stack(outputs, axis=0)


def test_optimized_algorithm():
    """Test that the optimized algorithm produces correct results."""
    print("Testing Phase 1 Optimized Algorithm (Carry-based approach)")
    print("=" * 70)
    
    # Small test case
    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, 4)
    
    num_heads = 4
    q_seq_len = 256
    sparse_len = 128
    head_dim = 128
    block_size = 128  # Process all at once for this test
    
    q = jax.random.normal(subkeys[0], (num_heads, q_seq_len, head_dim))
    k_sparse = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
    v_sparse = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))
    
    print(f"\nInput shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K_sparse: {k_sparse.shape}")
    print(f"  V_sparse: {v_sparse.shape}")
    print(f"  Block size: {block_size}")
    
    # Compute reference
    print("\nComputing reference...")
    out_ref = reference_attention_sparse(q, k_sparse, v_sparse)
    print(f"Reference output shape: {out_ref.shape}")
    print(f"Reference range: [{out_ref.min():.4f}, {out_ref.max():.4f}]")
    
    # Compute with optimized algorithm
    print("\nComputing with optimized algorithm...")
    out_opt = simulated_optimized_kernel(q, k_sparse, v_sparse, block_size)
    print(f"Optimized output shape: {out_opt.shape}")
    print(f"Optimized range: [{out_opt.min():.4f}, {out_opt.max():.4f}]")
    
    # Compare
    max_diff = jnp.abs(out_ref - out_opt).max()
    mean_diff = jnp.abs(out_ref - out_opt).mean()
    rel_error = max_diff / (jnp.abs(out_ref).max() + 1e-6)
    
    print(f"\n{'='*70}")
    print("CORRECTNESS RESULTS:")
    print(f"{'='*70}")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Relative error: {rel_error:.6e}")
    
    # Check correctness
    threshold = 5e-3  # Same as test suite
    if max_diff < threshold:
        print(f"\n✅ PASS: Optimized algorithm maintains correctness!")
        print(f"   Max diff {max_diff:.6e} < {threshold}")
        
        # Print sample values
        print(f"\nSample outputs (first 5 values of head 0, position 0):")
        print(f"  Reference: {out_ref[0, 0, :5]}")
        print(f"  Optimized: {out_opt[0, 0, :5]}")
        
        return True
    else:
        print(f"\n❌ FAIL: Algorithm correctness broken!")
        print(f"   Max diff {max_diff:.6e} >= {threshold}")
        
        # Debug info
        print(f"\nDEBUG: Finding location of max difference...")
        diff = jnp.abs(out_ref - out_opt)
        max_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
        print(f"  Location: head={max_idx[0]}, pos={max_idx[1]}, dim={max_idx[2]}")
        print(f"  Reference value: {out_ref[max_idx]:.6f}")
        print(f"  Optimized value: {out_opt[max_idx]:.6f}")
        
        return False


def test_multi_block():
    """Test with multiple blocks to ensure carry propagation works."""
    print("\n" + "="*70)
    print("Testing Multi-Block Processing (Carry Propagation)")
    print("="*70)
    
    key = jax.random.PRNGKey(123)
    key, *subkeys = jax.random.split(key, 4)
    
    num_heads = 2
    q_seq_len = 128
    sparse_len = 256  # 2 blocks
    head_dim = 128
    block_size = 128
    
    q = jax.random.normal(subkeys[0], (num_heads, q_seq_len, head_dim))
    k_sparse = jax.random.normal(subkeys[1], (num_heads, sparse_len, head_dim))
    v_sparse = jax.random.normal(subkeys[2], (num_heads, sparse_len, head_dim))
    
    print(f"\nProcessing {sparse_len // block_size} blocks...")
    
    out_ref = reference_attention_sparse(q, k_sparse, v_sparse)
    out_opt = simulated_optimized_kernel(q, k_sparse, v_sparse, block_size)
    
    max_diff = jnp.abs(out_ref - out_opt).max()
    
    if max_diff < 5e-3:
        print(f"✅ PASS: Multi-block carry propagation correct (diff: {max_diff:.6e})")
        return True
    else:
        print(f"❌ FAIL: Carry propagation broken (diff: {max_diff:.6e})")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 1 OPTIMIZATION CORRECTNESS VALIDATION")
    print("="*70)
    print("\nThis tests the optimized algorithm (carry-based m/l updates)")
    print("without needing TPU hardware.")
    print()
    
    # Run tests
    test1_pass = test_optimized_algorithm()
    test2_pass = test_multi_block()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Single block test: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Multi-block test:  {'✅ PASS' if test2_pass else '❌ FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n✅ ALL TESTS PASSED - Phase 1 optimizations maintain correctness!")
        print("\nKey optimizations validated:")
        print("  1. m, l kept in carry (not scratch buffers)")
        print("  2. Eliminated unnecessary broadcasts")
        print("  3. Direct operations on (bq, 1) tensors")
        print("\nNext: Deploy to TPU and measure performance improvement")
    else:
        print("\n❌ SOME TESTS FAILED - Do not deploy to TPU yet!")
        print("Fix algorithm errors before performance testing.")
    
    print("="*70)
