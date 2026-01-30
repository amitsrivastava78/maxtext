# Cascade Sparse Attention Development Notes

## Development Environment
- **Local**: macOS with CPU (Python 3.12.10, JAX 0.9.0)
- **Testing**: Colab with TPU
- **Virtual Environment**: `source venv/bin/activate`

## MaxText Attention Architecture

### Existing Attention Types
Located in `src/MaxText/common_types.py`:
- `GLOBAL` (default, causal)
- `LOCAL_SLIDING`
- `CHUNK`
- `MLA` (Multi-head Latent Attention)
- `FULL`

### Key Files for Attention Development
1. **`src/MaxText/layers/attentions.py`** (1129 lines)
   - Main `Attention` class (line 231)
   - Base for implementing custom attention patterns
   
2. **`src/MaxText/layers/attention_op.py`**
   - Low-level attention operations
   
3. **`src/MaxText/common_types.py`**
   - Add `CASCADE` to `AttentionType` enum (line 106)

4. **`src/MaxText/pyconfig.py` or `pyconfig_deprecated.py`**
   - Configuration validation for attention types

## Development Workflow

### Tiny Llama Model (115K parameters)
- **Purpose**: Fast CPU development and testing
- **Config**: [src/MaxText/configs/debug_kascade.yml](src/MaxText/configs/debug_kascade.yml)
- **Test script**: [test_tiny_llama.py](test_tiny_llama.py)
- **Architecture**: Exact match to Llama2-7B (smaller dimensions)
  - 2 layers (vs 32 in full model)
  - 64 embedding dim (vs 4096)
  - 2 attention heads (vs 32)
  - Same: RMSNorm, SwiGLU, causal masking, architecture
- **Performance**: ~0.1ms per forward pass on CPU
- **Memory**: ~0.5MB total

### Phase 1: Local Development (macOS CPU)
1. Add CASCADE attention type to enum
2. Implement cascade attention in `test_tiny_llama.py` first
3. Test with Tiny Llama (115K params, instant feedback)
4. Unit test on CPU with small models
5. Debug and iterate quickly

### Phase 2: Integration with MaxText
1. Port cascade attention to `src/MaxText/layers/attentions.py`
2. Update attention enum in `common_types.py`
3. Test with full MaxText infrastructure

### Phase 3: Colab TPU Testing
1. Export code to Colab
2. Test with TPU backend and Llama2-7B
3. Performance profiling
4. XLA optimization verification

## Quick Test Commands

### Test Tiny Llama Model
```bash
cd /Users/amitsrivasta/maxtext
source venv/bin/activate

# Test the Tiny Llama (115K params, runs in ~0.1ms)
python test_tiny_llama.py

# Expected output: 
# ✓ SUCCESS! Tiny Llama works perfectly on CPU
# ✓ Fast iteration: ~0ms per forward pass
```

### Local CPU Testing
```bash
# Run unit tests
python -m pytest tests/ -k attention

# Quick JAX test
python -c "import jax; print('Devices:', jax.devices())"
```

### Model Architecture Check
```bash
# List attention implementations
ls -la src/MaxText/layers/attention*.py

# View Tiny Llama config
cat src/MaxText/configs/debug_kascade.yml
```

## Development Tips for CPU→TPU

1. **Keep shapes explicit**: Avoid dynamic shapes that might work on CPU but fail on TPU
2. **Use `jax.jit` early**: Test compilation on CPU
3. **Profile on CPU first**: Use `jax.profiler` for initial checks
4. **Test with small batch sizes**: CPU has limited memory
5. **Use `jax.config.update('jax_platform_name', 'cpu')` for reproducibility

## Cascade Attention Implementation Checklist
- [ ] Add `CASCADE` to `AttentionType` enum
- [ ] Create cascade mask generation function
- [ ] Implement cascade attention mechanism
- [ ] Add config parameters for cascade settings
- [ ] Write unit tests
- [ ] CPU verification
- [ ] Colab TPU testing
- [ ] Performance benchmarking
- [ ] Documentation

## Resources
- MaxText Attention: `src/MaxText/layers/attentions.py`
- JAX Docs: https://jax.readthedocs.io/
- TPU Performance Guide: https://cloud.google.com/tpu/docs/performance-guide
