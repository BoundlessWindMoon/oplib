import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_kernel(
    Q, K, V, 
    l, m, O,
    N, d, 
    Tc, Tr, Bc, Br,
    softmax_scale,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
):
    # Parallel over batch and heads
    b = tl.program_id(0)
    h = tl.program_id(1)
    
    # Offsets for this block
    q_offset = b * stride_qb + h * stride_qh
    kv_offset = b * stride_kb + h * stride_kh
    o_offset = b * stride_ob + h * stride_oh
    lm_offset = b * (N * V.shape[1]) + h * N  # Simplified offset
    
    # Allocate shared memory (automatically managed by Triton)
    Qi = tl.zeros((Br, d), dtype=tl.float32)
    Kj = tl.zeros((Bc, d), dtype=tl.float32)
    Vj = tl.zeros((Bc, d), dtype=tl.float32)
    S = tl.zeros((Br, Bc), dtype=tl.float32)
    
    # Initialize output tile
    i = tl.arange(0, Br)
    O_block = tl.zeros((Br, d), dtype=tl.float32)
    
    # Load initial l and m
    row_m = tl.load(m + lm_offset + i)
    row_l = tl.load(l + lm_offset + i)
    
    for j in range(Tc):
        # Load K, V tiles
        k = tl.arange(0, Bc)
        Kj = tl.load(
            K + kv_offset + (j * Bc) * stride_kn + k[:, None] * stride_kd,
            mask=(j * Bc + k[:, None]) < N, other=0.0
        )
        Vj = tl.load(
            V + kv_offset + (j * Bc) * stride_vn + k[:, None] * stride_vd,
            mask=(j * Bc + k[:, None]) < N, other=0.0
        )
        
        for i_tile in range(Tr):
            # Load Q tile
            i = tl.arange(0, Br)
            Qi = tl.load(
                Q + q_offset + (i_tile * Br) * stride_qn + i[:, None] * stride_qd,
                mask=(i_tile * Br + i[:, None]) < N, other=0.0
            )
            
            # Compute S = Q @ K.T
            S = tl.dot(Qi, Kj, trans_b=True) * softmax_scale
            
            # Compute rowmax and softmax
            row_m_new = tl.maximum(tl.max(S, axis=1), row_m)
            row_l_new = tl.sum(tl.exp(S - row_m_new[:, None]), axis=1) + row_l * tl.exp(row_m - row_m_new)
            
            # Update output
            P = tl.exp(S - row_m_new[:, None])
            PV = tl.dot(P, Vj)
            
            O_block = (row_l[:, None] * tl.exp(row_m - row_m_new[:, None]) * O_block + 
                      tl.exp(row_m_new - row_m_new[:, None]) * PV) / row_l_new[:, None]
            
            # Update l and m for next iteration
            row_m = row_m_new
            row_l = row_l_new
            
    # Store final output
    i = tl.arange(0, Br)
    tl.store(
        O + o_offset + (i_tile * Br) * stride_on + i[:, None] * stride_od,
        O_block,
        mask=(i_tile * Br + i[:, None]) < N
    )
    tl.store(l + lm_offset + i, row_l)
    tl.store(m + lm_offset + i, row_m)

def attn_v0(Q, K, V):
    B, nh, N, d = Q.shape
    Br = Bc = 32  # Tile sizes
    Tc = triton.cdiv(N, Bc)
    Tr = triton.cdiv(N, Br)
    softmax_scale = 1.0 / (d ** 0.5)
    
    # Initialize outputs
    O = torch.zeros_like(Q)
    l = torch.zeros((B, nh, N), device='cuda')
    m = torch.full((B, nh, N), -float('inf'), device='cuda')
    
    # Launch kernel
    grid = (B, nh)
    flash_attn_kernel[grid](
        Q, K, V, l, m, O,
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        *[s for s in Q.stride()],
        *[s for s in K.stride()],
        *[s for s in V.stride()],
        *[s for s in O.stride()],
    )
    
    return O
