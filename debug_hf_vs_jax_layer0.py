"""
Debug script: Compare HF vs JAX LLaMA 3.2-1B hidden states at layer 0.
Drills down into: embedding, Q/K/V projection, RoPE, attention, MLP.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pickle
import torch
import jax
import jax.numpy as jnp

# ─────────── CONFIG ───────────
WEIGHTS_DIR = "llama_weights_chunked"
HF_DIR = "llama-checkpoint"
TOKEN_IDS = [128000, 791, 6864, 315, 9822, 374, 459, 3062, 13, 578, 6864]
SEQ_LEN = len(TOKEN_IDS)
EMBED_DIM = 2048
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
MLP_DIM = 8192
RMS_EPS = 1e-5
ROPE_THETA = 500000.0
ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

def rms_norm_np(x, weight, eps=1e-5):
    """RMSNorm in numpy."""
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps)
    return x_normed * weight

def compare(name, a, b, detail_idx=None):
    """Compare two arrays and print diagnostics."""
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    diff = np.abs(a - b)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = max_diff / (np.max(np.abs(a)) + 1e-12)
    match = "✅" if max_diff < 1e-3 else ("⚠️" if max_diff < 1e-1 else "❌")
    print(f"  {match} {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, rel_diff={rel_diff:.6e}")
    if detail_idx is not None:
        print(f"       HF[{detail_idx}] = {a[detail_idx]}")
        print(f"      JAX[{detail_idx}] = {b[detail_idx]}")
    if max_diff >= 1e-3:
        worst = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"       worst at {worst}: HF={a[worst]:.8f}, JAX={b[worst]:.8f}")
    return max_diff

# ─────────── STEP 1: HF REFERENCE (with hooks) ───────────
print("=" * 70)
print("STEP 1: Running HF model with hooks to capture intermediates")
print("=" * 70)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.float32)
model.eval()

# Storage for hooked values
hf_captures = {}

def make_input_hook(name):
    def hook(module, inp, out):
        hf_captures[name + "_input"] = inp[0].detach().clone()
        hf_captures[name + "_output"] = out.detach().clone() if isinstance(out, torch.Tensor) else None
    return hook

def attn_hook(module, inp, out):
    """Capture full attention layer output (tuple: attn_output, attn_weights, past_kv)."""
    hf_captures["attn_full_output"] = out[0].detach().clone()

# Register hooks on layer 0 components
layer0 = model.model.layers[0]
layer0.input_layernorm.register_forward_hook(make_input_hook("input_ln"))
layer0.self_attn.q_proj.register_forward_hook(make_input_hook("q_proj"))
layer0.self_attn.k_proj.register_forward_hook(make_input_hook("k_proj"))
layer0.self_attn.v_proj.register_forward_hook(make_input_hook("v_proj"))
layer0.self_attn.o_proj.register_forward_hook(make_input_hook("o_proj"))
layer0.self_attn.register_forward_hook(attn_hook)
layer0.post_attention_layernorm.register_forward_hook(make_input_hook("post_attn_ln"))
layer0.mlp.gate_proj.register_forward_hook(make_input_hook("gate_proj"))
layer0.mlp.up_proj.register_forward_hook(make_input_hook("up_proj"))
layer0.mlp.down_proj.register_forward_hook(make_input_hook("down_proj"))

input_ids = torch.tensor([TOKEN_IDS])
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)

hf_hidden_states = [h.numpy() for h in outputs.hidden_states]
hf_emb = hf_hidden_states[0]   # [1, seq, 2048]
hf_after_l0 = hf_hidden_states[1]  # [1, seq, 2048]

print(f"\nHF embedding[0,0,:5]     = {hf_emb[0,0,:5]}")
print(f"HF embedding[0,-1,:5]    = {hf_emb[0,-1,:5]}")
print(f"HF after_layer0[0,0,:5]  = {hf_after_l0[0,0,:5]}")
print(f"HF after_layer0[0,-1,:5] = {hf_after_l0[0,-1,:5]}")

# Extract hooked Q/K/V projections (HF stores weight as [out, in], output = input @ W.T)
hf_q_proj_out = hf_captures["q_proj_output"].numpy()  # [1, seq, 2048]
hf_k_proj_out = hf_captures["k_proj_output"].numpy()  # [1, seq, 512]  (8 KV heads)
hf_v_proj_out = hf_captures["v_proj_output"].numpy()  # [1, seq, 512]
hf_o_proj_out = hf_captures["o_proj_output"].numpy()  # [1, seq, 2048]
hf_input_ln_out = hf_captures["input_ln_output"].numpy()  # [1, seq, 2048]

print(f"\nHF input_ln[0,0,:5]  = {hf_input_ln_out[0,0,:5]}")
print(f"HF q_proj[0,0,:5]    = {hf_q_proj_out[0,0,:5]}")
print(f"HF k_proj[0,0,:5]    = {hf_k_proj_out[0,0,:5]}")
print(f"HF v_proj[0,0,:5]    = {hf_v_proj_out[0,0,:5]}")

# Now manually apply RoPE the same way HF does, to get Q/K after RoPE
# HF reshapes Q: [1, seq, 2048] -> [1, seq, 32, 64] -> [1, 32, seq, 64]
# HF reshapes K: [1, seq, 512]  -> [1, seq, 8,  64] -> [1, 8,  seq, 64]
hf_q_heads = hf_q_proj_out.reshape(1, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
hf_k_heads = hf_k_proj_out.reshape(1, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
hf_v_heads = hf_v_proj_out.reshape(1, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

print(f"\nHF q_heads shape: {hf_q_heads.shape}")  # [1, 32, seq, 64]
print(f"HF k_heads shape: {hf_k_heads.shape}")  # [1, 8, seq, 64]

# HF RoPE uses HALF-SPLIT: pairs (d, d+dim/2)
# Let's compute HF's RoPE manually to get Q/K after RoPE
def hf_rope_freqs(dim, seq_len, theta=500000.0, rope_scaling=None):
    """Compute cos/sin the HF way."""
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    
    if rope_scaling and rope_scaling.get("rope_type") == "llama3":
        factor = rope_scaling["factor"]
        low_freq_factor = rope_scaling["low_freq_factor"]
        high_freq_factor = rope_scaling["high_freq_factor"]
        orig_max_pos = rope_scaling["original_max_position_embeddings"]
        low_freq_wavelen = orig_max_pos / low_freq_factor
        high_freq_wavelen = orig_max_pos / high_freq_factor
        
        new_freqs = []
        for freq in inv_freq:
            wavelen = 2.0 * np.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / factor)
            else:
                smooth = (orig_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
        inv_freq = np.array(new_freqs)
    
    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq).astype(np.float32)  # [seq, dim/2]
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    return cos, sin

hf_cos, hf_sin = hf_rope_freqs(HEAD_DIM, SEQ_LEN, ROPE_THETA, ROPE_SCALING)
print(f"\nHF cos shape: {hf_cos.shape}, cos[0,:5]: {hf_cos[0,:5]}")
print(f"HF sin shape: {hf_sin.shape}, sin[0,:5]: {hf_sin[0,:5]}")

def apply_hf_rope(x, cos, sin):
    """Apply RoPE the HF way (half-split: rotate_half)."""
    # x: [1, heads, seq, dim]
    seq_len = x.shape[2]
    cos = cos[:seq_len][np.newaxis, np.newaxis, :, :]  # [1, 1, seq, dim/2]
    sin = sin[:seq_len][np.newaxis, np.newaxis, :, :]
    
    # HF rotate_half: split into two halves
    x1 = x[..., :HEAD_DIM//2]
    x2 = x[..., HEAD_DIM//2:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    
    # cos/sin broadcast: [1,1,seq,dim/2] -> need [1,1,seq,dim]
    cos_full = np.concatenate([cos, cos], axis=-1)
    sin_full = np.concatenate([sin, sin], axis=-1)
    
    return x * cos_full + rotated * sin_full

hf_q_rope = apply_hf_rope(hf_q_heads, hf_cos, hf_sin)
hf_k_rope = apply_hf_rope(hf_k_heads, hf_cos, hf_sin)

print(f"\nHF q_after_rope[0,0,0,:5] = {hf_q_rope[0,0,0,:5]}")
print(f"HF k_after_rope[0,0,0,:5] = {hf_k_rope[0,0,0,:5]}")
print(f"HF q_after_rope[0,0,1,:5] = {hf_q_rope[0,0,1,:5]}")
print(f"HF k_after_rope[0,0,1,:5] = {hf_k_rope[0,0,1,:5]}")

# Compute HF attention output manually to verify
# GQA: expand K,V from 8 -> 32 heads
repeat_factor = NUM_Q_HEADS // NUM_KV_HEADS  # 4
hf_k_expanded = np.repeat(hf_k_rope, repeat_factor, axis=1)  # [1, 32, seq, 64]
hf_v_expanded = np.repeat(hf_v_heads, repeat_factor, axis=1)  # [1, 32, seq, 64]

# Attention: Q @ K^T / sqrt(d)
hf_attn_scores = np.einsum('bhqd,bhkd->bhqk', hf_q_rope, hf_k_expanded) / np.sqrt(HEAD_DIM)
# Causal mask
causal_mask = np.tril(np.ones((SEQ_LEN, SEQ_LEN)))
hf_attn_scores = np.where(causal_mask[None, None], hf_attn_scores, -1e10)
# Softmax
hf_attn_scores_max = np.max(hf_attn_scores, axis=-1, keepdims=True)
hf_attn_exp = np.exp(hf_attn_scores - hf_attn_scores_max)
hf_attn_weights = hf_attn_exp / np.sum(hf_attn_exp, axis=-1, keepdims=True)

# attn_output = weights @ V, then reshape
hf_attn_out_manual = np.einsum('bhqk,bhkd->bhqd', hf_attn_weights, hf_v_expanded)
# Transpose back: [1, 32, seq, 64] -> [1, seq, 32, 64] -> [1, seq, 2048]
hf_attn_out_manual = hf_attn_out_manual.transpose(0, 2, 1, 3).reshape(1, SEQ_LEN, EMBED_DIM)

# Apply o_proj
hf_o_weight = layer0.self_attn.o_proj.weight.detach().numpy()  # [2048, 2048]
hf_attn_final_manual = hf_attn_out_manual @ hf_o_weight.T

print(f"\nHF manual attn_out (before o_proj)[0,0,:5] = {hf_attn_out_manual[0,0,:5]}")
print(f"HF manual attn_final (after o_proj)[0,0,:5] = {hf_attn_final_manual[0,0,:5]}")
print(f"HF hooked o_proj output[0,0,:5] = {hf_o_proj_out[0,0,:5]}")

# Clean up HF model to save memory
del model
del outputs

# ─────────── STEP 2: JAX with converted weights ───────────
print("\n" + "=" * 70)
print("STEP 2: Running JAX layer 0 with converted weights")
print("=" * 70)

# Load converted weights
with open(f"{WEIGHTS_DIR}/embeddings.pkl", "rb") as f:
    emb_data = pickle.load(f)
with open(f"{WEIGHTS_DIR}/layer_00.pkl", "rb") as f:
    layer0_data = pickle.load(f)

jax_embed_table = emb_data['embed_tokens'].astype(np.float32)
jax_final_ln = emb_data['norm'].astype(np.float32)
jax_config = emb_data['config']

print(f"JAX config: {jax_config}")
print(f"Embedding table shape: {jax_embed_table.shape}")

# Layer 0 weights
jax_q_kernel = layer0_data['attention']['q_proj']['kernel'].astype(np.float32)  # [2048, 2048]
jax_k_kernel = layer0_data['attention']['k_proj']['kernel'].astype(np.float32)  # [2048, 2048] (GQA-expanded)
jax_v_kernel = layer0_data['attention']['v_proj']['kernel'].astype(np.float32)  # [2048, 2048] (GQA-expanded)
jax_o_kernel = layer0_data['attention']['o_proj']['kernel'].astype(np.float32)  # [2048, 2048]
jax_input_ln = layer0_data['input_layernorm']['scale'].astype(np.float32)
jax_post_ln = layer0_data['post_attention_layernorm']['scale'].astype(np.float32)
jax_gate_kernel = layer0_data['mlp']['gate_proj']['kernel'].astype(np.float32)
jax_up_kernel = layer0_data['mlp']['up_proj']['kernel'].astype(np.float32)
jax_down_kernel = layer0_data['mlp']['down_proj']['kernel'].astype(np.float32)

print(f"Q kernel shape: {jax_q_kernel.shape}")
print(f"K kernel shape: {jax_k_kernel.shape}")
print(f"V kernel shape: {jax_v_kernel.shape}")
print(f"O kernel shape: {jax_o_kernel.shape}")
print(f"Gate kernel shape: {jax_gate_kernel.shape}")
print(f"Up kernel shape: {jax_up_kernel.shape}")
print(f"Down kernel shape: {jax_down_kernel.shape}")

# ─────────── STEP 2A: Embedding comparison ───────────
print("\n" + "-" * 50)
print("COMPARISON A: Embeddings")
print("-" * 50)
jax_emb = jax_embed_table[TOKEN_IDS]  # [seq, 2048]
jax_emb = jax_emb[np.newaxis, :]  # [1, seq, 2048]
compare("Embedding", hf_emb, jax_emb, (0, 0, slice(0, 5)))

# ─────────── STEP 2B: Input LayerNorm ───────────
print("\n" + "-" * 50)
print("COMPARISON B: Input LayerNorm")
print("-" * 50)
jax_ln_out = rms_norm_np(jax_emb, jax_input_ln, RMS_EPS)
compare("Input LayerNorm", hf_input_ln_out, jax_ln_out, (0, 0, slice(0, 5)))

# ─────────── STEP 2C: Q/K/V Projection (before RoPE) ───────────
print("\n" + "-" * 50)
print("COMPARISON C: Q/K/V Projection (before RoPE)")
print("-" * 50)

# JAX: output = input @ kernel  (kernel is [in, out])
jax_q_proj = jax_ln_out @ jax_q_kernel  # [1, seq, 2048]
jax_k_proj = jax_ln_out @ jax_k_kernel  # [1, seq, 2048] (GQA-expanded!)
jax_v_proj = jax_ln_out @ jax_v_kernel  # [1, seq, 2048] (GQA-expanded!)

print(f"JAX q_proj shape: {jax_q_proj.shape}")
print(f"JAX k_proj shape: {jax_k_proj.shape}")

# For Q: HF [1,seq,2048] vs JAX [1,seq,2048] — should match directly
compare("Q projection", hf_q_proj_out, jax_q_proj, (0, 0, slice(0, 5)))

# For K: HF has [1,seq,512] (8 heads), JAX has [1,seq,2048] (32 heads, GQA-expanded)
# To compare, expand HF K to 32 heads
hf_k_expanded_proj = hf_k_proj_out.reshape(1, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
hf_k_expanded_proj = np.repeat(hf_k_expanded_proj, repeat_factor, axis=2)
hf_k_expanded_proj = hf_k_expanded_proj.reshape(1, SEQ_LEN, EMBED_DIM)
compare("K projection (GQA-expanded)", hf_k_expanded_proj, jax_k_proj, (0, 0, slice(0, 5)))

hf_v_expanded_proj = hf_v_proj_out.reshape(1, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
hf_v_expanded_proj = np.repeat(hf_v_expanded_proj, repeat_factor, axis=2)
hf_v_expanded_proj = hf_v_expanded_proj.reshape(1, SEQ_LEN, EMBED_DIM)
compare("V projection (GQA-expanded)", hf_v_expanded_proj, jax_v_proj, (0, 0, slice(0, 5)))

# ─────────── STEP 2D: RoPE ───────────
print("\n" + "-" * 50)
print("COMPARISON D: Q/K after RoPE")
print("-" * 50)

# JAX model reshapes to 32 heads for both Q and K (since K is GQA-expanded)
jax_q_heads = jax_q_proj.reshape(1, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)  # [1, 32, seq, 64]
jax_k_heads = jax_k_proj.reshape(1, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)  # [1, 32, seq, 64]

# JAX RoPE uses ADJACENT PAIRS (interleaved): pairs (d, d+1)
# Inline the functions from kascade_layers.py to avoid import issues
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "kascade_layers",
    "/Users/amitsrivasta/maxtext/src/MaxText/layers/kascade_layers.py",
)
_kmod = importlib.util.module_from_spec(_spec)
# Patch the module's import chain so exec_module doesn't trigger MaxText __init__
import sys
sys.modules["kascade_layers"] = _kmod

# We'll just inline precompute_freqs_cis and apply_rope directly:
def _precompute_freqs_cis(dim, end, theta=500000.0, rope_scaling=None):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float64) / dim))
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", ""))
        if rope_type == "llama3":
            factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
            orig_max_pos = rope_scaling["original_max_position_embeddings"]
            low_freq_wavelen = orig_max_pos / low_freq_factor
            high_freq_wavelen = orig_max_pos / high_freq_factor
            new_freqs = []
            for freq in freqs:
                wavelen = 2.0 * np.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / factor)
                else:
                    smooth = (orig_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                    new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
            freqs = np.array(new_freqs)
    freqs = jnp.array(freqs, dtype=jnp.float32)
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)
    return freqs_cis

def _apply_rope(xq, xk, freqs_cis):
    xq = xq.astype(jnp.float32)
    xk = xk.astype(jnp.float32)
    xq_pairs = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_pairs = xk.reshape(*xk.shape[:-1], -1, 2)
    xq_complex = jax.lax.complex(xq_pairs[..., 0], xq_pairs[..., 1])
    xk_complex = jax.lax.complex(xk_pairs[..., 0], xk_pairs[..., 1])
    freqs_cis = freqs_cis[None, None, :xq.shape[2], :]
    xq_rotated = xq_complex * freqs_cis
    xk_rotated = xk_complex * freqs_cis
    xq_out = jnp.stack([xq_rotated.real, xq_rotated.imag], axis=-1).reshape(xq.shape)
    xk_out = jnp.stack([xk_rotated.real, xk_rotated.imag], axis=-1).reshape(xk.shape)
    return xq_out, xk_out

freqs_cis = _precompute_freqs_cis(HEAD_DIM, SEQ_LEN, ROPE_THETA, ROPE_SCALING)
jax_q_rope, jax_k_rope = _apply_rope(
    jnp.array(jax_q_heads), jnp.array(jax_k_heads), freqs_cis
)
jax_q_rope = np.array(jax_q_rope)
jax_k_rope = np.array(jax_k_rope)

print(f"JAX q_after_rope[0,0,0,:5] = {jax_q_rope[0,0,0,:5]}")
print(f"JAX k_after_rope[0,0,0,:5] = {jax_k_rope[0,0,0,:5]}")

# Compare Q after RoPE — HF has 32 heads, JAX has 32 heads
compare("Q after RoPE (head 0, pos 0)", hf_q_rope[0, 0, 0, :], jax_q_rope[0, 0, 0, :])
compare("Q after RoPE (head 0, pos 1)", hf_q_rope[0, 0, 1, :], jax_q_rope[0, 0, 1, :])
compare("Q after RoPE (head 0, pos 5)", hf_q_rope[0, 0, 5, :], jax_q_rope[0, 0, 5, :])

# For K: HF has 8 heads, JAX has 32 heads (expanded)
# Compare head 0 of HF with head 0 of JAX (should be same since heads 0-3 of expanded == head 0 of original)
compare("K after RoPE (HF h0 vs JAX h0, pos 0)", hf_k_rope[0, 0, 0, :], jax_k_rope[0, 0, 0, :])
compare("K after RoPE (HF h0 vs JAX h0, pos 1)", hf_k_rope[0, 0, 1, :], jax_k_rope[0, 0, 1, :])
compare("K after RoPE (HF h0 vs JAX h1, pos 1)", hf_k_rope[0, 0, 1, :], jax_k_rope[0, 1, 1, :])
compare("K after RoPE (HF h1 vs JAX h4, pos 1)", hf_k_rope[0, 1, 1, :], jax_k_rope[0, 4, 1, :])

# ─────────── STEP 2E: Attention scores & output ───────────
print("\n" + "-" * 50)
print("COMPARISON E: Attention scores & output")
print("-" * 50)

# JAX attention: all 32 heads for both Q and K (GQA already expanded)
jax_v_heads = jax_v_proj.reshape(1, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

jax_attn_scores = np.einsum('bhqd,bhkd->bhqk', jax_q_rope, jax_k_rope) / np.sqrt(HEAD_DIM)
jax_attn_scores = np.where(causal_mask[None, None], jax_attn_scores, -1e10)
jax_attn_max = np.max(jax_attn_scores, axis=-1, keepdims=True)
jax_attn_exp = np.exp(jax_attn_scores - jax_attn_max)
jax_attn_weights = jax_attn_exp / np.sum(jax_attn_exp, axis=-1, keepdims=True)

compare("Attn scores (head 0, q=0)", hf_attn_scores[0, 0, 0, :], jax_attn_scores[0, 0, 0, :])
compare("Attn scores (head 0, q=5)", hf_attn_scores[0, 0, 5, :SEQ_LEN], jax_attn_scores[0, 0, 5, :SEQ_LEN])
compare("Attn weights (head 0, q=5)", hf_attn_weights[0, 0, 5, :SEQ_LEN], jax_attn_weights[0, 0, 5, :SEQ_LEN])

jax_attn_out = np.einsum('bhqk,bhkd->bhqd', jax_attn_weights, jax_v_heads)
jax_attn_out_flat = jax_attn_out.transpose(0, 2, 1, 3).reshape(1, SEQ_LEN, EMBED_DIM)

compare("Attn output (before o_proj)", hf_attn_out_manual, jax_attn_out_flat, (0, 0, slice(0, 5)))

# Apply o_proj: JAX kernel is [in, out], so output = input @ kernel
jax_attn_final = jax_attn_out_flat @ jax_o_kernel
compare("Attn output (after o_proj)", hf_attn_final_manual, jax_attn_final, (0, 0, slice(0, 5)))

# ─────────── STEP 2F: Residual + Post-attn LN + MLP ───────────
print("\n" + "-" * 50)
print("COMPARISON F: Residual + Post-attn LayerNorm + MLP")
print("-" * 50)

# HF: residual = embedding + attn_output
hf_residual1 = hf_emb + hf_o_proj_out
jax_residual1 = jax_emb + jax_attn_final

compare("Residual after attention", hf_residual1, jax_residual1, (0, 0, slice(0, 5)))

# Post-attention LayerNorm
hf_post_ln_out = hf_captures["post_attn_ln_output"].numpy()
jax_post_ln_out = rms_norm_np(jax_residual1, jax_post_ln, RMS_EPS)
compare("Post-attn LayerNorm", hf_post_ln_out, jax_post_ln_out, (0, 0, slice(0, 5)))

# MLP: gate_proj, up_proj, down_proj
# HF: gate = silu(input @ gate_weight.T) * (input @ up_weight.T)
#     output = gate_up @ down_weight.T
hf_gate_out = hf_captures["gate_proj_output"].numpy()
hf_up_out = hf_captures["up_proj_output"].numpy()
hf_down_out = hf_captures["down_proj_output"].numpy()

jax_gate_out = jax_post_ln_out @ jax_gate_kernel
jax_up_out = jax_post_ln_out @ jax_up_kernel

compare("MLP gate_proj", hf_gate_out, jax_gate_out, (0, 0, slice(0, 5)))
compare("MLP up_proj", hf_up_out, jax_up_out, (0, 0, slice(0, 5)))

# SiLU activation
def silu(x):
    return x / (1.0 + np.exp(-x))

jax_mlp_intermediate = silu(jax_gate_out) * jax_up_out
jax_down_out = jax_mlp_intermediate @ jax_down_kernel

compare("MLP down_proj", hf_down_out, jax_down_out, (0, 0, slice(0, 5)))

# ─────────── STEP 2G: Final hidden state after layer 0 ───────────
print("\n" + "-" * 50)
print("COMPARISON G: Final hidden state after layer 0")
print("-" * 50)

jax_after_l0 = jax_residual1 + jax_down_out
compare("After layer 0 (full)", hf_after_l0, jax_after_l0, (0, 0, slice(0, 5)))
compare("After layer 0 (last token)", hf_after_l0[0, -1, :10], jax_after_l0[0, -1, :10])

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"HF  after_l0[0,0,:5]  = {hf_after_l0[0,0,:5]}")
print(f"JAX after_l0[0,0,:5]  = {jax_after_l0[0,0,:5]}")
print(f"HF  after_l0[0,-1,:5] = {hf_after_l0[0,-1,:5]}")
print(f"JAX after_l0[0,-1,:5] = {jax_after_l0[0,-1,:5]}")
