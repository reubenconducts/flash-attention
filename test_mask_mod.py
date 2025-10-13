#!/usr/bin/env python3

import sys
sys.path.append('flash_attn/cute')

def causal_mask_mod_py(head, batch, m_idx, n_idx):
    """Causal mask - only attend to positions before or at current position"""
    return n_idx <= m_idx  # Allow when column <= row (attend to previous positions)

# Test the causal mask function
print("Testing causal mask function:")
print(f"(0,0): {causal_mask_mod_py(0, 0, 0, 0)} (should be True)")
print(f"(0,1): {causal_mask_mod_py(0, 0, 0, 1)} (should be False)")
print(f"(1,0): {causal_mask_mod_py(0, 0, 1, 0)} (should be True)")
print(f"(1,1): {causal_mask_mod_py(0, 0, 1, 1)} (should be True)")
print(f"(1,2): {causal_mask_mod_py(0, 0, 1, 2)} (should be False)")

# Test with some specific positions from the error output
print("\nTesting specific error positions:")
print(f"seq_q=0, head_v=0: {causal_mask_mod_py(0, 0, 0, 0)} (should be True)")
print(f"seq_q=0, head_v=1: {causal_mask_mod_py(0, 0, 0, 1)} (should be False)")
print(f"seq_q=0, head_v=4: {causal_mask_mod_py(0, 0, 0, 4)} (should be False)")