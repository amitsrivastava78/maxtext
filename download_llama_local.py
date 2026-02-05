#!/usr/bin/env python3
"""
Download Llama 3.2-1B locally and convert to PyTorch format
"""

from huggingface_hub import snapshot_download, login
from safetensors.torch import load_file
import torch
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("Llama 3.2-1B Local Download & Setup")
    print("=" * 60)
    
    # Step 1: Login (will use cached token if available)
    print("\n📝 Checking Hugging Face authentication...")
    try:
        login(token=None)  # Uses cached token
        print("✅ Already authenticated")
    except:
        print("⚠️  Need to authenticate with Hugging Face")
        print("Get token from: https://huggingface.co/settings/tokens")
        login()
    
    # Step 2: Download model
    print("\n📥 Downloading Llama 3.2-1B (this may take 5-10 minutes)...")
    checkpoint_dir = snapshot_download(
        repo_id="meta-llama/Llama-3.2-1B",
        local_dir="./llama-checkpoint",
        local_dir_use_symlinks=False
    )
    print(f"✅ Downloaded to: {checkpoint_dir}")
    
    # Step 3: Check format
    print("\n🔍 Checking downloaded format...")
    ckpt_path = Path(checkpoint_dir)
    
    has_safetensors = list(ckpt_path.glob("*.safetensors"))
    has_pytorch = list(ckpt_path.glob("pytorch_model*.bin"))
    
    if has_safetensors and not has_pytorch:
        print("📦 Found safetensors format, converting to PyTorch...")
        
        # Load safetensors
        state_dict = {}
        for st_file in has_safetensors:
            print(f"   Loading {st_file.name}...")
            state_dict.update(load_file(str(st_file)))
        
        # Save as PyTorch
        output_file = ckpt_path / "pytorch_model.bin"
        torch.save(state_dict, output_file)
        
        size_gb = output_file.stat().st_size / 1e9
        print(f"✅ Converted to pytorch_model.bin ({size_gb:.2f} GB)")
        print(f"   {len(state_dict)} tensors")
    elif has_pytorch:
        print("✅ Already in PyTorch format")
    else:
        print("❌ No model files found!")
        return
    
    # Step 4: Verify config
    config_file = ckpt_path / "config.json"
    if config_file.exists():
        import json
        with open(config_file) as f:
            config = json.load(f)
        print(f"\n📋 Model configuration:")
        print(f"   Layers: {config.get('num_hidden_layers', 'N/A')}")
        print(f"   Hidden size: {config.get('hidden_size', 'N/A')}")
        print(f"   Attention heads: {config.get('num_attention_heads', 'N/A')}")
        print(f"   KV heads: {config.get('num_key_value_heads', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✅ Download and setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python convert_llama_weights.py --input ./llama-checkpoint --chunked")
    print("2. Run: python benchmark_kascade_final.py --device cpu")

if __name__ == "__main__":
    main()
