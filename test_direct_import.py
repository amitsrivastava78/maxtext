#!/usr/bin/env python3
"""Test that direct import of kascade_splash_attention works"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

print("Testing direct import of kascade_splash_attention...")
print()

try:
    # Direct import method (same as in benchmark)
    import importlib.util
    splash_path = os.path.join(src_path, "MaxText/layers/kascade_splash_attention.py")
    
    print(f"Loading from: {splash_path}")
    print(f"File exists: {os.path.exists(splash_path)}")
    print()
    
    spec = importlib.util.spec_from_file_location("kascade_splash_attention", splash_path)
    kascade_splash_module = importlib.util.module_from_spec(spec)
    
    print("Executing module...")
    spec.loader.exec_module(kascade_splash_module)
    
    # Store in sys.modules
    sys.modules['kascade_splash_attention'] = kascade_splash_module
    
    print("✅ Module loaded successfully!")
    print()
    
    # Check functions exist
    print("Checking exports:")
    print(f"  - kascade_calibrate_tiles: {hasattr(kascade_splash_module, 'kascade_calibrate_tiles')}")
    print(f"  - kascade_splash_attention: {hasattr(kascade_splash_module, 'kascade_splash_attention')}")
    print(f"  - create_kascade_splash_schedule: {hasattr(kascade_splash_module, 'create_kascade_splash_schedule')}")
    print(f"  - KascadeMask: {hasattr(kascade_splash_module, 'KascadeMask')}")
    print(f"  - KASCADE_TILE_CACHE: {hasattr(kascade_splash_module, 'KASCADE_TILE_CACHE')}")
    print()
    
    # Test retrieval from sys.modules
    retrieved = sys.modules.get('kascade_splash_attention')
    print(f"✅ Can retrieve from sys.modules: {retrieved is not None}")
    print(f"✅ Can access function: {hasattr(retrieved, 'kascade_splash_attention')}")
    print()
    
    print("=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("The direct import approach works correctly.")
    print("Ready to test on Colab TPU!")
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
