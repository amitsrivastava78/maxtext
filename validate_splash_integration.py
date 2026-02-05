#!/usr/bin/env python3
"""
Quick validation that SplashAttention integration can be imported.
This doesn't test TPU execution, just that the code structure is correct.
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

print("Testing SplashAttention integration...")

try:
    # Test 1: Import the module
    print("\n1. Testing import...")
    from MaxText.layers import kascade_splash_attention
    print("   ✅ kascade_splash_attention module imported successfully")
    
    # Test 2: Check main functions exist
    print("\n2. Checking functions...")
    assert hasattr(kascade_splash_attention, 'kascade_calibrate_tiles'), "Missing kascade_calibrate_tiles"
    print("   ✅ kascade_calibrate_tiles found")
    
    assert hasattr(kascade_splash_attention, 'kascade_splash_attention'), "Missing kascade_splash_attention"
    print("   ✅ kascade_splash_attention found")
    
    assert hasattr(kascade_splash_attention, 'create_kascade_splash_schedule'), "Missing create_kascade_splash_schedule"
    print("   ✅ create_kascade_splash_schedule found")
    
    assert hasattr(kascade_splash_attention, 'KascadeMask'), "Missing KascadeMask"
    print("   ✅ KascadeMask class found")
    
    # Test 3: Check the cache exists
    print("\n3. Checking cache...")
    assert hasattr(kascade_splash_attention, 'KASCADE_TILE_CACHE'), "Missing KASCADE_TILE_CACHE"
    print("   ✅ KASCADE_TILE_CACHE found")
    
    # Test 4: Verify function signatures
    print("\n4. Checking function signatures...")
    import inspect
    
    sig = inspect.signature(kascade_splash_attention.kascade_splash_attention)
    params = list(sig.parameters.keys())
    expected_params = ['query', 'key', 'value', 'layer_id', 'is_anchor_layer', 
                       'anchor_layer_id', 'tile_size', 'top_k_ratio']
    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"
    print(f"   ✅ All {len(expected_params)} parameters present")
    
    print("\n" + "=" * 70)
    print("✅ ALL VALIDATION TESTS PASSED!")
    print("=" * 70)
    print()
    print("Ready to test on Colab TPU with:")
    print("  python benchmark_kascade_final.py --device tpu --use_splash_kernel \\")
    print("         --seq_len 2048 --top_k 8 --tile_size 64")
    print()
    
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except AssertionError as e:
    print(f"\n❌ Validation failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
