# Running Optimization Suite on TPU Colab

## Quick Setup

### 1. Create a new Colab notebook with TPU runtime
- Go to Runtime → Change runtime type
- Select **TPU** as hardware accelerator
- Click Save

### 2. Clone the repository and setup

```python
# In Colab cell 1: Clone repo
!git clone https://github.com/amitsrivastava78/maxtext.git
%cd maxtext
```

```python
# In Colab cell 2: Install dependencies
!pip install -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
!pip install -q jaxlib
```

```python
# In Colab cell 3: Verify TPU is available
import jax
print(f"Devices: {jax.devices()}")
print(f"Backend: {jax.devices()[0].platform}")
# Should show: Backend: tpu
```

### 3. Run the optimization suite

```python
# In Colab cell 4: Run comprehensive optimization tests
!python optimize_for_2x_speedup.py
```

This will test **5 optimization strategies**:
1. **Block size tuning** - Try different memory access patterns
2. **Compiler optimizations** - donate_argnums, inline, XLA flags
3. **Memory layout** - Contiguous memory, pre-transposed tensors
4. **Input size sensitivity** - Find sweet spots
5. **Algorithmic ideas** - Document potential kernel improvements

Expected runtime: **5-10 minutes**

## Common Issues & Fixes

### ❌ Issue: "TPU is already in use by process"

**Error:**
```
RuntimeError: The TPU is already in use by process with pid XXXX
```

**Fix (Recommended):**
1. Go to **Runtime → Restart runtime** in Colab menu
2. Re-run from Cell 1 (clone might say "already exists" - that's fine)
3. Skip Cell 2 if JAX is already installed
4. Run Cell 4 again

**Alternative Fix:**
```python
# In a new cell:
import os
os.system('sudo lsof -t /dev/accel0 | xargs -r kill -9 2>/dev/null || true')
# Then: Runtime → Restart runtime
```

### ⚠️ Warning: "Transparent hugepages not enabled"

This is just a **warning**, not an error. You can safely **ignore** it.

The script will run fine. If you want to enable it anyway (optional):
```python
!sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
```

## What to expect

The script will output:
- ✅ Results for each strategy
- 🎯 Best speedup achieved
- 📊 Comparison vs baseline (0.957×)
- 💡 Recommendations if target not met

## Target Goals

- **Minimum**: 1.2× speedup (20% faster than JAX)
- **Ideal**: 2.0× speedup (2× faster than JAX)
- **Current**: 0.957× (4.5% slower - need improvement!)

## If you hit issues

### TPU not found
```python
# Restart runtime and ensure TPU is selected
# Runtime → Restart runtime
# Runtime → Change runtime type → TPU
```

### Import errors
```python
# Reinstall JAX with TPU support
!pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Out of memory
```python
# The script tests large configs - this is expected
# Results before OOM are still valid
```

## Quick alternative: Just test block sizes

If you want faster results (2-3 min), run just the block size sweep:

```python
!python tune_block_sizes_comprehensive.py
```

## Reporting Results

Please share:
1. Best speedup achieved: `X.XXX×`
2. Which strategy worked: `block_sizes / compiler / memory / input_size`
3. Best configuration found
4. Full output (copy from Colab)

We'll use this to:
- Update kernel defaults if >1.2× found
- Decide if kernel changes needed if <1.2×
- Document best practices for TPU

## Next Steps After Results

### If we achieve ≥1.2×:
✅ Update `kascade_kernel.py` with optimal config
✅ Re-run `test_phase1_standalone.py` to validate
✅ Ready for model integration!

### If we're still <1.2×:
🔧 Need kernel algorithm changes:
- Flash Attention style implementation
- bfloat16 for memory bandwidth
- Fused operations
- Consider hybrid approach (JAX for small, kernel for large)
