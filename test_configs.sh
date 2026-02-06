#!/bin/bash
# Test different configurations to find optimal settings

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var before running}"

echo "Testing configurations for <5% degradation target..."
echo "==========================================================="
echo ""

# Test 1: Paper config with lower threshold (more REUSE)
echo "Test 1: tile=16, top_k=12, threshold=0.50 (more REUSE layers)"
python3 benchmark_kascade_final.py --device cpu --tile_size 16 --top_k 12 --threshold 0.50 2>&1 | grep -A 6 "RESULTS ON REAL TEXT"
echo ""
echo "---"
echo ""

# Test 2: Higher top_k for less sparsity
echo "Test 2: tile=16, top_k=24, threshold=0.50 (75% coverage)"
python3 benchmark_kascade_final.py --device cpu --tile_size 16 --top_k 24 --threshold 0.50 2>&1 | grep -A 6 "RESULTS ON REAL TEXT"
echo ""
echo "---"
echo ""

# Test 3: Even higher top_k
echo "Test 3: tile=16, top_k=28, threshold=0.50 (87.5% coverage)"
python3 benchmark_kascade_final.py --device cpu --tile_size 16 --top_k 28 --threshold 0.50 2>&1 | grep -A 6 "RESULTS ON REAL TEXT"
echo ""

echo "==========================================================="
echo "Summary: Find config with degradation < 5%"
