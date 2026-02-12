import torch
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import os

def attention(query, key, value, is_causal=False):
    """
    Explicit implementation of scaled dot-product attention.
    This is needed for cases where torch's built-in SDPA is not suited for model export (e.g., ONNX).
    It supports GQA automatically by handling different head dimensions for query and key/value.
    key_cache and value_cache are provided separately to minimize the concatenation overhead. They are used only if is_causal is False.

    Args:
        query: Tensor of shape (batch_size, num_heads, seq_length, q_head_dim)
        key:   Tensor of shape (batch_size, num_heads, seq_length, kv_head_dim)
        value: Tensor of shape (batch_size, num_heads, seq_length, kv_head_dim)
        key_cache [optional]: Tensor of shape (batch_size, num_heads, cache_seq_length, kv_head_dim)
        value_cache [optional]: Tensor of shape (batch_size, num_heads, cache_seq_length, kv_head_dim)
        is_causal: Whether to apply causal masking (for decoder use)
    """
    batch_size, q_head_num,  seq_length, head_dim  = query.size()
    _,          kv_head_num, _,          _         = key.size()
    group_size = q_head_num // kv_head_num

    # reshape query, key, value for possible GQA
    query = query.view(batch_size, kv_head_num, group_size, seq_length, head_dim) # (batch_size, kv_head_num, group_size, seq_length, head_dim)
    key   = key.unsqueeze(2)    # (batch_size, kv_head_num, 1, seq_length, head_dim)
    value = value.unsqueeze(2)  # (batch_size, kv_head_num, 1, seq_length, head_dim)

    device = query.device
    dtype = query.dtype
    
    # Scaled QK^T
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(query, key.permute(0, 1, 2, 4, 3))*scale

    # Cast to float32 for numerical stability
    scores = scores.to(torch.float32)

    # Generate negative causal mask
    if is_causal:
        i = torch.arange(seq_length, device=device).view(1,1,seq_length,1)
        j = torch.arange(seq_length, device=device).view(1,1,1,seq_length)
        mask = i < j
        # Mask the future positions
        scores = scores.masked_fill(mask, torch.finfo(torch.float32).min)

    # Stabilize softmax by subtracting max
    scores = scores - scores.max(dim=-1, keepdim=True).values

    # Softmax and back to original dtype
    scores = torch.nn.functional.softmax(scores, dim=-1).to(dtype)

    # Project the values over the scores
    attn_output = torch.matmul(scores, value)  # (batch_size, kv_head_num, group_size, seq_length, head_dim)

    # reshape back to original shape
    attn_output = attn_output.view(batch_size, q_head_num, seq_length, head_dim) # (batch_size, num_heads, seq_length, head_dim)

    return attn_output

def attention_benchmark(Q, K, V):
    tic = time.perf_counter()
    attn_output = attention(Q, K, V, is_causal=False)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    return toc - tic

def sdpa_benchmark(Q, K, V):
    tic = time.perf_counter()
    attn_output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False, enable_gqa=True)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    return toc - tic

def cat_benchmark(K, V, K_cache, V_cache):
    tic = time.perf_counter()
    K_full = torch.cat([K_cache, K], dim=2)
    V_full = torch.cat([V_cache, V], dim=2)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    return toc - tic

def benchmark(target_device="cuda"):
    
    repeated_tests = 50
    seq_max_len = 2**16
    heads = 60
    kv_heads = 4
    head_dim = 128

    Q = torch.randn(1, heads, seq_max_len, head_dim)
    K = torch.randn(1, kv_heads, seq_max_len, head_dim)
    V = torch.randn(1, kv_heads, seq_max_len, head_dim)

    print(f"Tests on {target_device}")

    Q = Q.to(target_device)
    K = K.to(target_device)
    V = V.to(target_device)

    results_dict = {
        "seq_len": [],
        "attention": [],
        "sdpa": [],
        "cat": [],
    }

    seq_len_list = torch.linspace(1, seq_max_len, steps=100, dtype=torch.int32).tolist()
    for t in seq_len_list:
        results_dict["seq_len"].append(t+1)
        # Prepare inputs
        new_q = Q[:, :, t:t+1, :]
        new_k = K[:, :, t:t+1, :]
        new_v = V[:, :, t:t+1, :]
        k_cache = K[:, :, :t, :]
        v_cache = V[:, :, :t, :]
        k_slice = K[:, :, :t+1, :]
        v_slice = V[:, :, :t+1, :]
        # Run benchmarks
        # Attention
        for _ in range(5): # warm-up
            attention_benchmark(new_q, k_slice, v_slice)
        attn_times = []
        for i in range(repeated_tests):
            attn_time = attention_benchmark(new_q, k_slice, v_slice)
            attn_times.append(attn_time)
        results_dict["attention"].append(sum(attn_times)/repeated_tests)
        # SDPA
        for _ in range(5): # warm-up
            sdpa_benchmark(new_q, k_slice, v_slice)
        sdpa_times = []
        for i in range(repeated_tests):
            sdpa_time = sdpa_benchmark(new_q, k_slice, v_slice)
            sdpa_times.append(sdpa_time)
        results_dict["sdpa"].append(sum(sdpa_times)/repeated_tests)
        # Cat
        for _ in range(5): # warm-up
            cat_benchmark(new_k, new_v, k_cache, v_cache)
        cat_times = []
        for i in range(repeated_tests):
            cat_time = cat_benchmark(new_k, new_v, k_cache, v_cache)
            cat_times.append(cat_time)
        results_dict["cat"].append(sum(cat_times)/repeated_tests)

    # Store results to CSV
    df = pd.DataFrame(results_dict)
    output_path = "attention_benchmark_cpu.csv"
    df.to_csv(output_path, index=False)
    print(f"CPU benchmark results saved to {os.path.abspath(output_path)}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results_dict["seq_len"], results_dict["attention"], label="Custom Attention", color='blue')
    plt.plot(results_dict["seq_len"], results_dict["sdpa"], label="Built-in SDPA", color='orange')
    plt.plot(results_dict["seq_len"], results_dict["cat"], label="Cat K/V Cache", color='green')
    plt.yscale('log')
    plt.xlabel("Sequence Length")
    plt.ylabel("Time (seconds, log scale)")
    plt.title(f"Attention Mechanism Benchmark on {target_device}")
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    filename = f"attention_benchmark_{target_device}.png"
    plt.savefig(filename)
    print(f"{target_device} benchmark plot saved to {os.path.abspath(filename)}")

if __name__ == "__main__":
    benchmark(target_device="cpu")
    if torch.cuda.is_available():
        benchmark(target_device="cuda")
