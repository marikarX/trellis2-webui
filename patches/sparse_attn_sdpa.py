"""Patch sparse-attention source to support PyTorch SDPA.

This is needed for environments without flash_attn. Upstream's attention
implementation has an SDPA branch, but its environment selector must also
accept 'sdpa' or it silently keeps its Flash Attention default. Applied by
setup_local.sh and vastai_setup.sh during provisioning.
"""
import sys
import os

SDPA_BLOCK = '''    elif config.ATTN == 'sdpa':
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        # Pad variable-length sequences into a batch for SDPA
        N = len(q_seqlen)
        max_q = max(q_seqlen)
        max_kv = max(kv_seqlen)

        if num_all_args == 1:
            q_flat, k_flat, v_flat = qkv.unbind(dim=1)  # each [T, H, C]
        elif num_all_args == 2:
            q_flat = q
            k_flat, v_flat = kv.unbind(dim=1)
        else:
            q_flat, k_flat, v_flat = q, k, v

        H = q_flat.shape[1]
        C_q = q_flat.shape[2]
        C_v = v_flat.shape[2]

        # Pad into [N, H, max_len, C] batches
        q_padded = torch.zeros(N, max_q, H, C_q, dtype=q_flat.dtype, device=device)
        k_padded = torch.zeros(N, max_kv, H, q_flat.shape[2], dtype=k_flat.dtype, device=device)
        v_padded = torch.zeros(N, max_kv, H, C_v, dtype=v_flat.dtype, device=device)

        q_offset = 0
        kv_offset = 0
        for i in range(N):
            ql = q_seqlen[i]
            kvl = kv_seqlen[i]
            q_padded[i, :ql] = q_flat[q_offset:q_offset + ql]
            k_padded[i, :kvl] = k_flat[kv_offset:kv_offset + kvl]
            v_padded[i, :kvl] = v_flat[kv_offset:kv_offset + kvl]
            q_offset += ql
            kv_offset += kvl

        # [N, L, H, C] -> [N, H, L, C]
        q_padded = q_padded.permute(0, 2, 1, 3)
        k_padded = k_padded.permute(0, 2, 1, 3)
        v_padded = v_padded.permute(0, 2, 1, 3)

        # Create attention mask for variable lengths
        attn_mask = None
        if not all(sl == max_kv for sl in kv_seqlen):
            attn_mask = torch.zeros(N, 1, max_q, max_kv, dtype=torch.bool, device=device)
            for i in range(N):
                attn_mask[i, :, :q_seqlen[i], :kv_seqlen[i]] = True
            # Convert bool mask: True = attend, use float mask for SDPA
            float_mask = torch.zeros_like(attn_mask, dtype=q_flat.dtype)
            float_mask[~attn_mask] = float('-inf')
            attn_mask = float_mask

        out_padded = sdpa(q_padded, k_padded, v_padded, attn_mask=attn_mask)  # [N, H, max_q, C_v]
        out_padded = out_padded.permute(0, 2, 1, 3)  # [N, max_q, H, C_v]

        # Unpad back to flat
        parts = []
        for i in range(N):
            parts.append(out_padded[i, :q_seqlen[i]])
        out = torch.cat(parts, dim=0)  # [T_Q, H, C_v]'''


def patch(trellis_dir):
    target = os.path.join(trellis_dir, 'trellis2', 'modules', 'sparse', 'attention', 'full_attn.py')
    if not os.path.exists(target):
        print(f"  [SKIP] {target} not found")
    else:
        src = open(target).read()
        if "config.ATTN == 'sdpa'" in src:
            print("  [SKIP] full_attn.py SDPA branch already present")
        else:
            # Insert before the else/raise ValueError branch
            marker = "    else:\n        raise ValueError(f\"Unknown attention module: {config.ATTN}\")"
            if marker not in src:
                print("  [WARN] Could not find insertion point in full_attn.py")
            else:
                src = src.replace(marker, SDPA_BLOCK + "\n" + marker)
                open(target, 'w').write(src)
                print("  Patched full_attn.py with sdpa backend")

    # Current upstream contains the SDPA implementation but omits it from the
    # environment allowlist, making ATTN_BACKEND=sdpa silently use flash_attn.
    config_target = os.path.join(trellis_dir, 'trellis2', 'modules', 'sparse', 'config.py')
    if not os.path.exists(config_target):
        print(f"  [SKIP] {config_target} not found")
        return
    config_src = open(config_target).read()
    old_backends = "['xformers', 'flash_attn', 'flash_attn_3']"
    new_backends = "['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']"
    if any(
        "env_sparse_attn_backend in [" in line and "'sdpa'" in line
        for line in config_src.splitlines()
    ):
        print("  [SKIP] config.py already accepts sdpa")
    elif old_backends in config_src:
        config_src = config_src.replace(old_backends, new_backends)
        open(config_target, 'w').write(config_src)
        print("  Patched config.py to accept sdpa")
    else:
        print("  [WARN] Could not find sparse attention backend allowlist in config.py")


if __name__ == '__main__':
    trellis_dir = sys.argv[1] if len(sys.argv) > 1 else '/workspace/TRELLIS.2'
    patch(trellis_dir)
