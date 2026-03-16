"""
Shared gradient and head bottleneck metrics for training scripts.

Usage (baseline with single lm_head):
    from grad_metrics import compute_grad_metrics
    metrics = compute_grad_metrics(model)
    if metrics:
        print(metrics["log_line"])

Usage (multi-head with output_heads):
    metrics = compute_grad_metrics(model, multi_head=True)
"""

import torch


def _svd_effective_rank(S):
    """Effective rank from singular values via entropy."""
    S_n = S / (S.sum() + 1e-10)
    return torch.exp(-torch.sum(S_n * torch.log(S_n + 1e-10))).item()


@torch.no_grad()
def compute_grad_metrics(model, multi_head=False):
    """
    Compute gradient bottleneck metrics from current .grad buffers.
    Call AFTER backward, BEFORE optimizer.step().

    Returns dict with metrics + "log_line" string, or None if no grads.
    """
    # Backbone gradient norm
    bb_norm = sum(
        p.grad.float().norm() ** 2
        for p in model.transformer.h.parameters()
        if p.grad is not None
    ).sqrt().item()

    if multi_head and hasattr(model, "output_heads"):
        return _multi_head_metrics(model, bb_norm)
    elif hasattr(model, "lm_head"):
        return _single_head_metrics(model, bb_norm)
    return None


def _single_head_metrics(model, bb_norm):
    """Metrics for baseline single-linear LM head."""
    hd_norm = sum(
        p.grad.float().norm() ** 2
        for p in model.lm_head.parameters()
        if p.grad is not None
    ).sqrt().item()

    hg = model.lm_head.weight.grad
    if hg is None:
        return None

    S = torch.linalg.svdvals(hg.float())
    eff_rank = _svd_effective_rank(S)
    max_rank = min(hg.shape)
    rank_ratio = eff_rank / max_rank
    survival = bb_norm / (hd_norm + 1e-10)
    top10_energy = (S[:10].sum() / (S.sum() + 1e-10)).item() * 100

    log_line = (
        f"  grads | backbone: {bb_norm:.4f} | head: {hd_norm:.4f} "
        f"| survival: {survival:.4f} | head_effrank: {eff_rank:.1f} "
        f"| rank_ratio: {rank_ratio:.3f} | top10e: {top10_energy:.1f}%"
    )

    return dict(
        bb_norm=bb_norm, hd_norm=hd_norm, survival=survival,
        eff_rank=eff_rank, rank_ratio=rank_ratio, top10_energy=top10_energy,
        log_line=log_line,
    )


def _multi_head_metrics(model, bb_norm):
    """Metrics for multi-head output (EXP-007/011 style)."""
    heads = model.output_heads

    hd_norm = sum(
        p.grad.float().norm() ** 2
        for p in heads.parameters()
        if p.grad is not None
    ).sqrt().item()

    # Gradient effective rank (concatenated proj grads)
    proj_grads = [h.proj.weight.grad for h in heads if h.proj.weight.grad is not None]
    if not proj_grads:
        return None

    G = torch.cat([g.float() for g in proj_grads], dim=0)
    S = torch.linalg.svdvals(G)
    eff_rank = _svd_effective_rank(S)

    # Derived metrics
    max_rank = min(G.shape)
    rank_ratio = eff_rank / max_rank
    survival = bb_norm / (hd_norm + 1e-10)

    # Weight-space head similarity
    n_h = len(heads)
    proj_ws = [h.proj.weight.float() for h in heads]

    # Pairwise cosine similarity (avg)
    cos_sims = []
    for i in range(n_h):
        for j in range(i + 1, n_h):
            a, b = proj_ws[i].flatten(), proj_ws[j].flatten()
            cos_sims.append((a @ b / (a.norm() * b.norm() + 1e-10)).item())
    head_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0

    # Union effective rank via SVD subspaces
    subspaces = [torch.linalg.svd(w, full_matrices=False)[2] for w in proj_ws]

    all_Vt = torch.cat(subspaces, dim=0)
    S_u = torch.linalg.svdvals(all_Vt)
    union_rank = _svd_effective_rank(S_u)

    log_line = (
        f"  grads | backbone: {bb_norm:.4f} | head: {hd_norm:.4f} "
        f"| survival: {survival:.4f} | head_effrank: {eff_rank:.1f} "
        f"| rank_ratio: {rank_ratio:.3f} | cos: {head_cos:.4f} "
        f"| union_rank: {union_rank:.1f}"
    )

    return dict(
        bb_norm=bb_norm, hd_norm=hd_norm, survival=survival,
        eff_rank=eff_rank, rank_ratio=rank_ratio,
        head_cos=head_cos, union_rank=union_rank,
        log_line=log_line,
    )
