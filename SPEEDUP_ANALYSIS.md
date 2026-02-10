# Kascade Speedup Analysis — Full Derivation

## 1. Amdahl's Law

**If you speed up a fraction `f` of a program by a factor `s`, the overall speedup is:**

$$\text{Speedup} = \frac{1}{(1 - f) + \frac{f}{s}}$$

- `f` = fraction of runtime that is attention (the part Kascade speeds up)
- `s` = how much faster the attention portion becomes
- `(1 - f)` = everything else (Q/K/V/O projections + MLP), unchanged

### Example (top_k=5 at 32K):
- f = 66.7% (attention is 2/3 of compute)
- s = 3.78× (effective attention speedup across 16 layers)
- Speedup = 1 / (0.333 + 0.667/3.78) = 1 / (0.333 + 0.176) = **1.96×** theoretical

---

## 2. The Attention Fraction Formula (Attn%)

### What Kascade speeds up vs. what it doesn't

Per transformer layer, the FLOPs decompose into:

| Operation | FLOPs | Kascade speeds up? |
|---|---|---|
| Q = x · W_Q | S · d_embed² | ❌ |
| K = x · W_K | S · d_embed² | ❌ |
| V = x · W_V | S · d_embed² | ❌ |
| **Q · K^T** (attention scores) | **S² · d_embed** | **✅ tile-skippable** |
| **softmax(scores) · V** (value aggregation) | **S² · d_embed** | **✅ tile-skippable** |
| O = attn_out · W_O | S · d_embed² | ❌ |
| gate_proj (SwiGLU) | S · d_embed · d_MLP | ❌ |
| up_proj (SwiGLU) | S · d_embed · d_MLP | ❌ |
| down_proj (SwiGLU) | S · d_embed · d_MLP | ❌ |

**Total FLOPs per layer** (dividing everything by d_embed for simplification):

$$\text{Total} = \underbrace{2 \cdot S^2}_{\text{speedable}} + \underbrace{4 \cdot S \cdot d_\text{embed}}_{\text{Q,K,V,O}} + \underbrace{3 \cdot S \cdot d_\text{MLP}}_{\text{gate, up, down}}$$

### The formula

$$\text{Attn\%} = f = \frac{2 \cdot S^2}{2 \cdot S^2 + 4 \cdot S \cdot d_\text{embed} + 3 \cdot S \cdot d_\text{MLP}}$$

### Concrete numbers (LLaMA 3.2-1B at 32K)

| Term | Expression | Value |
|---|---|---|
| 2·S² | 2 × 32768² | 2,147,483,648 |
| 4·S·d_embed | 4 × 32768 × 2048 | 268,435,456 |
| 3·S·d_MLP | 3 × 32768 × 8192 | 805,306,368 |
| **Total** | | **3,221,225,472** |
| **Attn%** | 2,147M / 3,221M | **66.7%** |

At 32K, two-thirds of the compute is in the tile-skippable attention core.

---

## 3. How `s` (Effective Attention Speedup) Is Calculated

### Per-layer attention cost

| Layer type | Attention cost (relative) | Why |
|---|---|---|
| DENSE | 1.0 | Full causal attention, all S² FLOPs |
| ANCHOR | 1.05 | Full attention + tile scoring overhead (~5%) |
| REUSE | top_k / num_tiles | Only computes top_k blocks out of num_tiles |

### Across 16 layers with schedule 1D + 3A + 12R, top_k=5, 256 tiles

| Component | Layers | Per-layer cost | Total cost |
|---|---|---|---|
| DENSE | 1 | 1.0 | 1.0 |
| ANCHOR (→DENSE in timing) | 3 | 1.0 | 3.0 |
| REUSE | 12 | 5/256 = 0.0195 | 0.234 |
| **Total sparse** | 16 | | **4.234** |
| **Total dense** | 16 | 1.0 each | **16.0** |

$$s = \frac{16.0}{4.234} = 3.78\times$$

---

## 4. Tile Counting: Dense vs Sparse

### Dense (full causal attention)

With S=32768 and tile_size=128 → 256 tiles.

Causal mask: query tile `q` attends to key tiles `0..q` (lower triangle + diagonal).

$$\text{Dense tiles} = \sum_{q=0}^{255}(q+1) = \frac{256 \times 257}{2} = 32{,}896$$

This is computed for **every head** (32 heads) but the tile count per head is 32,896.

### Sparse (Kascade REUSE, top_k=5)

Each of the 256 query tiles picks its top-5 key tiles:

$$\text{Sparse tiles} = 256 \times 5 = 1{,}280$$

**Reduction: 1,280 / 32,896 = 3.9% of tiles computed.**

The Tokamax SplashAttention kernel uses a dynamic grid that literally skips the other 96.1% of tile blocks — no wasted FLOPs.

---

## 5. The Full 1.74× Result — Step by Step

### Given
- Measured: dense = 4,421 ms, sparse = 2,545 ms → **1.74×**
- Attn% = 66.7%
- Schedule: 1D + 3A + 12R
- Effective attention speedup s = 3.78×

### Amdahl's Law prediction

$$\text{Theoretical} = \frac{1}{(1 - 0.667) + \frac{0.667}{3.78}} = \frac{1}{0.333 + 0.176} = \frac{1}{0.510} = 1.96\times$$

### Why measured (1.74×) < theoretical (1.96×)

1. **Kernel launch overhead** — each REUSE layer still pays Tokamax dispatch cost
2. **Cache management** — KASCADE_CACHE.update(saved_cache) before each sparse run
3. **Memory bandwidth** — theoretical counts FLOPs, but TPU is often memory-bound
4. **Block granularity** — tiles at causal boundary are partially filled but fully computed

### Why measured (1.74×) > old theoretical (1.61×)

The old formula underestimated Attn% (51.6% vs 66.7%), so the old theoretical was too low. The HBM bandwidth cascade effect (sparse attention freeing memory bandwidth for subsequent MLP ops) also contributes to exceeding conservative estimates.

---

## 6. Summary Table

| Metric | Value | Formula |
|---|---|---|
| Sequence length | 32,768 | — |
| Tiles | 256 | S / tile_size |
| Top-K | 5 (2%) | — |
| Dense tiles/head | 32,896 | T(T+1)/2 |
| Sparse tiles/head | 1,280 | T × k |
| Tile reduction | 96.1% | 1 - k·2/(T+1) |
| **Attn%** | **66.7%** | 2S² / (2S² + 4·S·d + 3·S·d_MLP) |
| Effective attn speedup | 3.78× | 16 / (4 + 12·k/T) |
| Theoretical speedup | 1.96× | Amdahl's Law |
| **Measured speedup** | **1.74×** | dense_ms / sparse_ms |
| PPL degradation | 0.11% | (sparse-dense)/dense |
