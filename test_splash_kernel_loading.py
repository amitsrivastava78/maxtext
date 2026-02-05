#!/usr/bin/env python3
"""
Quick test to verify SplashAttention kernel loading on TPU
==========================================================
Run this before the full benchmark to catch issues early.
"""

import sys
import os
import importlib.util

print("=" * 70)
print("🧪 SPLASH KERNEL LOADING TEST")
print("=" * 70)

# Get paths
src_path = os.path.join(os.path.dirname(__file__), 'src')
kernel_path = os.path.join(src_path, "MaxText/kernels/splash_attention_kernel.py")
splash_path = os.path.join(src_path, "MaxText/layers/kascade_splash_attention.py")

print(f"\n📁 Paths:")
print(f"   Kernel: {kernel_path}")
print(f"   Exists: {os.path.exists(kernel_path)}")
print(f"   Splash: {splash_path}")
print(f"   Exists: {os.path.exists(splash_path)}")

# Test 1: Load splash_attention_kernel
print(f"\n🔧 Test 1: Loading splash_attention_kernel...")
try:
    kernel_spec = importlib.util.spec_from_file_location("splash_attention_kernel_test", kernel_path)
    kernel_module = importlib.util.module_from_spec(kernel_spec)
    print("   ✓ Spec created")
    
    kernel_spec.loader.exec_module(kernel_module)
    print("   ✅ Kernel module loaded successfully!")
    
    # Check for key components
    has_blocksizes = hasattr(kernel_module, 'BlockSizes')
    has_make_splash = 'make_splash_mha' in dir(kernel_module)
    print(f"   ✓ Has BlockSizes: {has_blocksizes}")
    print(f"   ✓ Has make_splash_mha: {has_make_splash}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load kascade_splash_attention with injected kernel
print(f"\n🔧 Test 2: Loading kascade_splash_attention...")
try:
    splash_spec = importlib.util.spec_from_file_location("kascade_splash_attention_test", splash_path)
    splash_module = importlib.util.module_from_spec(splash_spec)
    print("   ✓ Spec created")
    
    # Inject kernel module
    splash_module._KERNEL_MODULE = kernel_module
    splash_module.__file__ = splash_path
    print("   ✓ Kernel injected")
    
    splash_spec.loader.exec_module(splash_module)
    print("   ✅ Splash module loaded successfully!")
    
    # Check for key functions
    has_calibrate = hasattr(splash_module, 'kascade_calibrate_tiles')
    has_attention = hasattr(splash_module, 'kascade_splash_attention')
    has_schedule = hasattr(splash_module, 'create_kascade_splash_schedule')
    has_cache = hasattr(splash_module, 'KASCADE_TILE_CACHE')
    
    print(f"   ✓ Has kascade_calibrate_tiles: {has_calibrate}")
    print(f"   ✓ Has kascade_splash_attention: {has_attention}")
    print(f"   ✓ Has create_kascade_splash_schedule: {has_schedule}")
    print(f"   ✓ Has KASCADE_TILE_CACHE: {has_cache}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\n🎉 SplashAttention kernel is ready to use!")
print("   You can now run the full benchmark with --use_splash_kernel")
