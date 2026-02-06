#!/bin/bash
# Script to find optimal top_k value for <5% perplexity degradation

echo "Testing different top_k values to find optimal setting..."
echo "=========================================================="
echo ""

for topk in 20 24 28 30; do
    echo "Testing top_k=$topk (this takes ~2 minutes)..."
    python3 benchmark_kascade_final.py --threshold 0.50 --device cpu --top_k $topk 2>&1 | tail -20 | grep -A 3 "RESULTS ON REAL TEXT"
    echo ""
    echo "---"
    echo ""
done

echo "=========================================================="
echo "Summary: Find the top_k value where degradation is < 5%"
