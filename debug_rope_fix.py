"""
Verify the RoPE mismatch: HF half-split vs JAX adjacent-pair.
Show that using HF-style half-split RoPE in JAX fixes the divergence.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pickle
import jax
import jax.numpy as jnp

WEIGHTS_DIR = "llama_weights_chunked"
TOKEN_IDS = [128000, 791, 6864, 315, 9822, 374, 459, 3062, 13, 578, 6864]
SEQ_LEN = len(TOKEN_IDS)
EMBED_DIM = 2048
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
RMS_EPS = 1e-5
ROPE_THETA = 500000.0
ROPE_SCALING = {
    "rope_type": "llama3", "factor": 32.0,
    "low_freq_factor": 1.0, "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

def rms_norm_np(x, weight, eps=1e-5):
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * weight

# ── Load HF reference ──
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("llama-checkpoint", dtype=torch.float32)
model.eval()
captures = {}
def hook(name):
    def fn(mod, inp, out):
        captures[name] = out.detach().clone() if isinstance(out, torch.Tensor) else None
    return fn
l0 = model.model.layers[0]
l0.self_attn.q_proj.register_forward_hook(hook("q"))
l0.self_attn.k_proj.register_forward_hook(hook("k"))
l0.self_attn.v_proj.register_forward_hook(hook("v"))
with torch.no_grad():
    out = model(torch.tensor([TOKEN_IDS]), output_hidden_states=True)
hf_after_l0 = out.hidden_states[1].numpy()
hf_emb = out.hidden_states[0].numpy()
hf_q = captures["q"].numpy()  # [1, seq, 2048]
hf_k = captures["k"].numpy()  # [1, seq, 512]
hf_v = captures["v"].numpy()  # [1, seq, 512]
del model, out

# ── Load JAX weights ──
with open(f"{WEIGHTS_DIR}/embeddings.pkl", "rb") as f:
    emb_data = pickle.load(f)
with open(f"{WEIGHTS_DIR}/layer_00.pkl", "rb") as f:
    l0_data = pickle.load(f)

embed_table = emb_data['embed_tokens'].astype(np.float32)
q_kernel = l0_data['attention']['q_proj']['kernel'].astype(np.float32)
k_kernel = l0_data['attention']['k_proj']['kernel'].astype(np.float32)
v_kernel = l0_data['attention']['v_proj']['kernel'].astype(np.float32)
o_kernel = l0_data['attention']['o_proj']['kernel'].astype(np.float32)
input_ln = l0_data['input_layernorm']['scale'].astype(np.float32)
post_ln = l0_data['post_attention_layernorm']['scale'].astype(np.float32)
gate_k = l0_data['mlp']['gate_proj']['kernel'].astype(np.float32)
up_k = l0_data['mlp']['up_proj']['kernel'].astype(np.float32)
down_k = l0_data['mlp']['down_proj']['kernel'].astype(np.float32)

# ── Compute through layer 0 ──
jax_emb = embed_table[TOKEN_IDS][np.newaxis]
ln_out = rms_norm_np(jax_emb, input_ln, RMS_EPS)
jax_q = ln_out @ q_kernel
jax_k_full = ln_out @ k_kernel  # [1, seq, 2048] GQA-expanded
jax_v_full = ln_out @ v_kernel  # [1, seq, 2048] GQA-expanded

# Reshape to heads
jax_q_heads = jax_q.reshape(1, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
jax_k_heads = jax_k_full.reshape(1, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
jax_v_heads = jax_v_full.reshape(1, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

# ── Compute scaled frequencies (same for both) ──
def compute_freqs(dim, seq_len, theta, scaling):
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    if scaling and scaling.get("rope_type") == "llama3":
        factor = scaling["factor"]
        lff = scaling["low_freq_factor"]
        hff = scaling["high_freq_factor"]
        omp = scaling["original_max_position_embeddings"]
        lw = omp / lff
        hw = omp / hff
        new = []
        for f in inv_freq:
            wl = 2 * np.pi / f
            if wl < hw: new.append(f)
            elif wl > lw: new.append(f / factor)
            else:
                s = (omp / wl - lff) / (hff - lff)
                new.append((1 - s) * f / factor + s * f)
        inv_freq = np.array(new)
    t = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(t, inv_freq).astype(np.float32)  # [seq, dim/2]
    return np.cos(angles), np.sin(angles)

cos, sin = compute_freqs(HEAD_DIM, SEQ_LEN, ROPE_THETA, ROPE_SCALING)

# ── METHOD 1: Current JAX adjacent-pair RoPE (BROKEN for HF weights) ──
def apply_rope_adjacent(q, k, cos, sin):
    """Adjacent pairs: pairs (0,1), (2,3), ... — Meta/LLaMA original format."""
    # q/k: [1, heads, seq, dim]
    cos_b = cos[np.newaxis, np.newaxis, :, :]  # [1,1,seq,dim/2]
    sin_b = sin[np.newaxis, np.newaxis, :, :]
    
    def rotate(x):
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)  # [..., dim/2, 2]
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]
        # Rotate: (x0*cos - x1*sin, x0*sin + x1*cos)
        out0 = x0 * cos_b - x1 * sin_b
        out1 = x0 * sin_b + x1 * cos_b
        return np.stack([out0, out1], axis=-1).reshape(x.shape)
    
    return rotate(q), rotate(k)

# ── METHOD 2: HF half-split RoPE (rotate_half) ──
def apply_rope_half_split(q, k, cos, sin):
    """Half-split: pairs (0, dim/2), (1, dim/2+1), ... — HuggingFace format."""
    cos_b = np.concatenate([cos, cos], axis=-1)[np.newaxis, np.newaxis, :, :]
    sin_b = np.concatenate([sin, sin], axis=-1)[np.newaxis, np.newaxis, :, :]
    
    def rotate(x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = np.concatenate([-x2, x1], axis=-1)
        return x * cos_b + rotated * sin_b
    
    return rotate(q), rotate(k)

# ── Apply both ──
q_adj, k_adj = apply_rope_adjacent(jax_q_heads, jax_k_heads, cos, sin)
q_half, k_half = apply_rope_half_split(jax_q_heads, jax_k_heads, cos, sin)

# HF reference Q/K after RoPE
hf_q_h = hf_q.reshape(1, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
hf_k_h = hf_k.reshape(1, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
repeat = NUM_Q_HEADS // NUM_KV_HEADS
hf_k_exp = np.repeat(hf_k_h, repeat, axis=1)

hf_q_rope, hf_k_rope = apply_rope_half_split(hf_q_h, hf_k_h, cos, sin)
hf_k_rope_exp = np.repeat(hf_k_rope, repeat, axis=1)

print("=" * 70)
print("RoPE COMPARISON: Adjacent-pair vs Half-split")
print("=" * 70)
for pos in [0, 1, 5, 10]:
    diff_adj = np.max(np.abs(hf_q_rope[0,0,pos] - q_adj[0,0,pos]))
    diff_half = np.max(np.abs(hf_q_rope[0,0,pos] - q_half[0,0,pos]))
    print(f"  pos={pos}: Adjacent max_diff={diff_adj:.6e}  |  Half-split max_diff={diff_half:.6e}  {'✅ half-split wins' if diff_half < diff_adj else ''}")

for pos in [0, 1, 5, 10]:
    diff_adj = np.max(np.abs(hf_k_rope_exp[0,0,pos] - k_adj[0,0,pos]))
    diff_half = np.max(np.abs(hf_k_rope_exp[0,0,pos] - k_half[0,0,pos]))
    print(f"  K pos={pos}: Adjacent max_diff={diff_adj:.6e}  |  Half-split max_diff={diff_half:.6e}")

# ── Full layer 0 with half-split RoPE ──
print("\n" + "=" * 70)
print("FULL LAYER 0 with HALF-SPLIT RoPE")
print("=" * 70)

# Attention: Q @ K^T / sqrt(d)
causal = np.tril(np.ones((SEQ_LEN, SEQ_LEN)))
# k_half has 32 heads (GQA-expanded from JAX side), so no repeat needed
scores = np.einsum('bhqd,bhkd->bhqk', q_half, k_half) / np.sqrt(HEAD_DIM)
scores = np.where(causal[None, None], scores, -1e10)
sm = np.max(scores, axis=-1, keepdims=True)
w = np.exp(scores - sm); w = w / w.sum(-1, keepdims=True)

v_exp = np.repeat(
    hf_v.reshape(1, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0,2,1,3),
    repeat, axis=1
)
# Use JAX V (GQA-expanded)
attn_out = np.einsum('bhqk,bhkd->bhqd', w, jax_v_heads)
attn_flat = attn_out.transpose(0,2,1,3).reshape(1, SEQ_LEN, EMBED_DIM)
attn_final = attn_flat @ o_kernel

residual1 = jax_emb + attn_final
post_ln_out = rms_norm_np(residual1, post_ln, RMS_EPS)

def silu(x): return x / (1 + np.exp(-x))
gate_out = post_ln_out @ gate_k
up_out = post_ln_out @ up_k
mlp_out = silu(gate_out) * up_out @ down_k

jax_after_l0_fixed = residual1 + mlp_out

diff_old = np.max(np.abs(hf_after_l0 - (jax_emb + (jax_q_heads.transpose(0,2,1,3).reshape(1,SEQ_LEN,EMBED_DIM) @ o_kernel))))  # placeholder

# Compare
diff_fixed = np.abs(hf_after_l0 - jax_after_l0_fixed)
print(f"Max diff (half-split RoPE): {np.max(diff_fixed):.6e}")
print(f"Mean diff: {np.mean(diff_fixed):.6e}")
print(f"HF  after_l0[0,0,:5]  = {hf_after_l0[0,0,:5]}")
print(f"JAX after_l0[0,0,:5]  = {jax_after_l0_fixed[0,0,:5]}")
print(f"HF  after_l0[0,-1,:5] = {hf_after_l0[0,-1,:5]}")
print(f"JAX after_l0[0,-1,:5] = {jax_after_l0_fixed[0,-1,:5]}")
print(f"HF  after_l0[0,5,:5]  = {hf_after_l0[0,5,:5]}")
print(f"JAX after_l0[0,5,:5]  = {jax_after_l0_fixed[0,5,:5]}")

match = "✅ MATCH" if np.max(diff_fixed) < 1e-3 else "❌ STILL DIVERGING"
print(f"\nResult: {match}")

if np.max(diff_fixed) < 1e-3:
    print("\n🎉 ROOT CAUSE CONFIRMED: The JAX apply_rope uses ADJACENT-PAIR format")
    print("   but HF weights expect HALF-SPLIT (rotate_half) format.")
    print("   FIX: Change apply_rope in kascade_layers.py to use half-split pairing.")
else:
    print("\n   Additional investigation needed...")
    # Check per-position
    for t in range(SEQ_LEN):
        d = np.max(np.abs(hf_after_l0[0,t] - jax_after_l0_fixed[0,t]))
        print(f"   pos {t}: max_diff = {d:.6e}")
