# Kascade Kernel Performance Optimization Plan

## Current Status
- **Correctness**: ✅ Achieved (max diff 1.85e-3)
- **Performance**: ❌ 0.66× speedup (1.52× **slower** than JAX reference)
- **Target**: 2-3× speedup

## Root Cause Analysis

### 1. Missing Prefetching (CRITICAL)
SplashAttention uses `data_next` arrays to prefetch the next K/V block while computing the current one. This hides memory latency.

**Our kernel**: Sequential reads - compute block i, then fetch block i+1
**SplashAttention**: Parallel - fetch block i+1 WHILE computing block i

**Impact**: ~30-40% performance loss from memory stalls

### 2. Scratch Buffer Overhead
Reading m_scratch, l_scratch on every iteration:
```python
m_prev = m_scratch_ref[...]  # Memory read
l_prev = l_scratch_ref[...]  # Memory read
```

**Better approach**: Keep in registers, only write back at end

**Impact**: ~10-15% performance loss from extra memory traffic

### 3. Block Size Suboptimal
Current: `block_q=512, block_kv_sparse=128, block_kv_compute=128`

SplashAttention uses dynamic block sizing based on:
- Head dimension
- Sequence length
- Memory constraints

**Impact**: ~15-20% performance loss from cache misses

### 4. Broadcasting Overhead
```python
m_next_all_lanes = jnp.broadcast_to(m_next, (bq, NUM_LANES))
l_curr_all_lanes = jnp.broadcast_to(l_curr, (bq, NUM_LANES))
l_next_all_lanes = jnp.broadcast_to(l_next, (bq, NUM_LANES))
```

3 broadcasts per iteration, each creating temporary arrays.

**Impact**: ~5-10% performance loss

### 5. Missing Compiler Hints
No `preferred_element_type` on second matmul, no explicit fusion hints.

**Impact**: ~5% performance loss

## Optimization Strategy

### Phase 1: Quick Wins (30 min)
1. **Remove scratch buffer reads in loop**
   - Keep m, l in local variables
   - Write back only at loop end
   - Expected: 10-15% improvement

2. **Optimize broadcasts**
   - Use slicing instead: `m_scratch_ref[:, :1]` 
   - Direct assignment without intermediate
   - Expected: 5-10% improvement

3. **Add compiler hints**
   - `preferred_element_type=float32` on V matmul
   - Expected: 3-5% improvement

**Total Phase 1**: ~18-30% improvement → **0.78-0.86× speedup**

### Phase 2: Block Size Tuning (1 hour)
1. **Benchmark different block sizes**
   - Test: (256, 128, 128), (512, 256, 256), (1024, 128, 128)
   - Profile cache hit rates
   - Expected: 15-25% improvement

**After Phase 2**: ~1.0-1.1× speedup (finally break even!)

### Phase 3: Prefetching (2-3 hours)
1. **Implement data_next style prefetching**
   - Add prefetch buffers for K/V
   - Overlap compute with memory
   - This is the CRITICAL optimization
   - Expected: 30-50% improvement

**After Phase 3**: ~1.3-1.65× speedup

### Phase 4: Advanced (if needed)
1. **Fused operations**: Combine exp and matmul
2. **Custom memory layout**: Optimize for TPU memory hierarchy
3. **Loop unrolling**: Manual unroll with pragmas

**After Phase 4**: Target 2-3× speedup

## Immediate Action Plan

### Step 1: Profile Current Kernel
```bash
# Get detailed timing breakdown
python benchmark_kascade_final.py --device tpu --seq_len 1024 --profile
```

Identify:
- Time in QK matmul
- Time in V matmul  
- Time in memory reads/writes
- Time in broadcasts/reshapes

### Step 2: Implement Phase 1 (Quick Wins)
File: `src/MaxText/kernels/kascade_kernel.py`

Changes:
```python
# Before loop: initialize locals
m_curr_acc = jnp.full((bq, 1), mask_value, dtype=float32)
l_curr_acc = jnp.zeros((bq, 1), dtype=float32)

def body(kv_compute_idx, carry):
    m_prev, l_prev = carry  # From carry, not scratch!
    
    # ... computation ...
    
    return (m_next, l_next)  # Pass via carry

# Run loop with carry
m_final, l_final = lax.fori_loop(0, num_iters, body, 
                                  (m_curr_acc, l_curr_acc))

# Write back once at end
m_scratch_ref[...] = jnp.broadcast_to(m_final, (bq, NUM_LANES))
l_scratch_ref[...] = jnp.broadcast_to(l_final, (bq, NUM_LANES))
```

### Step 3: Benchmark and Iterate
After each optimization:
1. Run `test_kascade_kernel.py` - verify correctness maintained
2. Run performance benchmark - measure speedup
3. If speedup < target, continue to next optimization

## Success Criteria
- **Minimum acceptable**: 1.5× speedup (better than baseline)
- **Target**: 2-3× speedup (original goal)
- **Stretch**: 3-4× speedup (beating SplashAttention for sparse case)

## Timeline
- Phase 1: 30 minutes → 0.8× speedup (still slower but improving)
- Phase 2: 1.5 hours → 1.1× speedup (break even)
- Phase 3: 4 hours → 1.5× speedup (minimum acceptable)
- Phase 4: 8+ hours → 2-3× speedup (target achieved)

## Notes
The current slowdown is NOT a fundamental limitation. The algorithm is sound. The issue is implementation efficiency. With proper optimizations (especially prefetching), we should achieve the target speedup.

The most critical optimization is **prefetching** (Phase 3), which alone could give us 30-50% improvement. But we need Phase 1+2 first to get to baseline performance.
