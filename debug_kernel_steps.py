"""
Step-by-step kernel debugging - output intermediate values
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools

NUM_LANES = 128

# Test 1: Can we output zeros?
def test_output_zeros():
    def kernel(o_ref):
        o_ref[...] = jnp.zeros_like(o_ref)
    
    grid = (2, 1, 1)  # 2 heads
    bq = 128
    head_dim = 128
    
    def out_map(h, i, j):
        return h, 0, 0  # Return 3 values for 3D grid
    
    out_spec = pl.BlockSpec((None, bq, head_dim), out_map)
    
    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 128, 128), jnp.float32),
        out_specs=out_spec,
        grid=grid,
    )()
    
    print("Test 1: Output zeros")
    print(f"  Result shape: {output.shape}")
    print(f"  Result range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  All zeros? {jnp.allclose(output, 0.0)}")
    return output

# Test 2: Can we output ones?
def test_output_ones():
    def kernel(o_ref):
        o_ref[...] = jnp.ones_like(o_ref)
    
    grid = (2, 1, 1)
    bq = 128
    head_dim = 128
    
    def out_map(h, i, j):
        return h, 0, 0
    
    out_spec = pl.BlockSpec((None, bq, head_dim), out_map)
    
    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 128, 128), jnp.float32),
        out_specs=out_spec,
        grid=grid,
    )()
    
    print("\nTest 2: Output ones")
    print(f"  Result shape: {output.shape}")
    print(f"  Result range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  All ones? {jnp.allclose(output, 1.0)}")
    return output

# Test 3: Can we read input and write it to output?
def test_passthrough():
    def kernel(q_ref, o_ref):
        o_ref[...] = q_ref[...]
    
    grid = (2, 1, 1)
    bq = 128
    head_dim = 128
    
    def q_map(h, i, j):
        return h, 0, 0
    
    def out_map(h, i, j):
        return h, 0, 0
    
    in_spec = pl.BlockSpec((None, bq, head_dim), q_map)
    out_spec = pl.BlockSpec((None, bq, head_dim), out_map)
    
    q = jax.random.normal(jax.random.PRNGKey(42), (2, 128, 128))
    
    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 128, 128), jnp.float32),
        in_specs=[in_spec],
        out_specs=out_spec,
        grid=grid,
    )(q)
    
    print("\nTest 3: Passthrough (read input, write to output)")
    print(f"  Input range: [{q.min():.4f}, {q.max():.4f}]")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Match? {jnp.allclose(q, output)}")
    return q, output

# Test 4: Can we use scratch buffer?
def test_scratch_buffer():
    def kernel(scratch_ref, o_ref):
        # Write to scratch
        scratch_ref[...] = jnp.full_like(scratch_ref, 42.0)
        # Read from scratch and write to output
        o_ref[...] = scratch_ref[...]
    
    grid = (2, 1, 1)
    bq = 128
    head_dim = 128
    
    def out_map(h, i, j):
        return h, 0, 0
    
    out_spec = pl.BlockSpec((None, bq, head_dim), out_map)
    scratch_shapes = [pltpu.VMEM((bq, head_dim), jnp.float32)]
    
    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 128, 128), jnp.float32),
        out_specs=out_spec,
        scratch_shapes=scratch_shapes,
        grid=grid,
    )()
    
    print("\nTest 4: Scratch buffer (write 42.0, read back)")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  All 42.0? {jnp.allclose(output, 42.0)}")
    return output

# Test 5: Multiple scratch buffers - can we read the right one?
def test_multiple_scratch():
    def kernel(scratch1_ref, scratch2_ref, scratch3_ref, o_ref):
        # Write different values to each scratch
        scratch1_ref[...] = jnp.full_like(scratch1_ref, 10.0)
        scratch2_ref[...] = jnp.full_like(scratch2_ref, 20.0)
        scratch3_ref[...] = jnp.full_like(scratch3_ref, 30.0)
        # Read from scratch3 and write to output
        o_ref[...] = scratch3_ref[...]
    
    grid = (2, 1, 1)
    bq = 128
    head_dim = 128
    
    def out_map(h, i, j):
        return h, 0, 0
    
    out_spec = pl.BlockSpec((None, bq, head_dim), out_map)
    scratch_shapes = [
        pltpu.VMEM((bq, NUM_LANES), jnp.float32),  # scratch1 (like m_scratch)
        pltpu.VMEM((bq, NUM_LANES), jnp.float32),  # scratch2 (like l_scratch)
        pltpu.VMEM((bq, head_dim), jnp.float32),   # scratch3 (like o_scratch)
    ]
    
    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 128, 128), jnp.float32),
        out_specs=out_spec,
        scratch_shapes=scratch_shapes,
        grid=grid,
    )()
    
    print("\nTest 5: Multiple scratch buffers")
    print(f"  Expected: 30.0 (from scratch3)")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Correct? {jnp.allclose(output, 30.0)}")
    if not jnp.allclose(output, 30.0):
        if jnp.allclose(output, 10.0):
            print(f"  ❌ Getting scratch1 instead of scratch3!")
        elif jnp.allclose(output, 20.0):
            print(f"  ❌ Getting scratch2 instead of scratch3!")
    return output

print("="*70)
print("STEP-BY-STEP KERNEL DEBUG")
print("="*70)

test_output_zeros()
test_output_ones()
test_passthrough()
test_scratch_buffer()
test_multiple_scratch()

print("\n" + "="*70)
print("If Test 5 shows wrong scratch buffer, that's the bug!")
print("="*70)
