"""
Debug Pallas parameter order
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_LANES = 128

# Test different parameter orders
def test_order_1():
    """Test: inputs, then scratch, then output"""
    def kernel(q_ref, scratch_ref, o_ref):
        # Write unique values to each
        o_ref[...] = scratch_ref[...] + q_ref[...] + 100.0
    
    grid = (1, 1, 1)
    bq = 128
    head_dim = 128
    
    def q_map(h, i, j): return 0, 0  # 2D array needs 2 return values
    def out_map(h, i, j): return 0, 0
    
    in_spec = pl.BlockSpec((bq, head_dim), q_map)
    out_spec = pl.BlockSpec((bq, head_dim), out_map)
    scratch_shapes = [pltpu.VMEM((bq, head_dim), jnp.float32)]
    
    q = jnp.ones((128, 128)) * 1.0
    
    try:
        output = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float32),
            in_specs=[in_spec],
            out_specs=out_spec,
            scratch_shapes=scratch_shapes,
            grid=grid,
        )(q)
        print(f"Order 1 (q, scratch, o): Output = {output[0, 0]:.1f}")
        return True
    except Exception as e:
        print(f"Order 1 failed: {e}")
        return False

def test_order_2():
    """Test: inputs, then output, then scratch"""
    def kernel(q_ref, o_ref, scratch_ref):
        # Write unique values to each
        scratch_ref[...] = jnp.full_like(scratch_ref, 42.0)
        o_ref[...] = scratch_ref[...] + q_ref[...] + 100.0
    
    grid = (1, 1, 1)
    bq = 128
    head_dim = 128
    
    def q_map(h, i, j): return 0, 0  # 2D array needs 2 return values
    def out_map(h, i, j): return 0, 0
    
    in_spec = pl.BlockSpec((bq, head_dim), q_map)
    out_spec = pl.BlockSpec((bq, head_dim), out_map)
    scratch_shapes = [pltpu.VMEM((bq, head_dim), jnp.float32)]
    
    q = jnp.ones((128, 128)) * 1.0
    
    try:
        output = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float32),
            in_specs=[in_spec],
            out_specs=out_spec,
            scratch_shapes=scratch_shapes,
            grid=grid,
        )(q)
        print(f"Order 2 (q, o, scratch): Output = {output[0, 0]:.1f}")
        return True
    except Exception as e:
        print(f"Order 2 failed: {e}")
        return False

print("="*70)
print("TESTING PALLAS PARAMETER ORDER")
print("="*70)
print()

success_1 = test_order_1()
print()
success_2 = test_order_2()

print()
print("="*70)
if success_1:
    print("✅ Correct order: inputs, scratch, output")
elif success_2:
    print("✅ Correct order: inputs, output, scratch")
else:
    print("❌ Neither order worked!")
print("="*70)
