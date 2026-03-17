"""
Gradient Bottleneck Analysis
Based on "Lost in Backpropagation: The LM Head is a Gradient Bottleneck" (Godey & Artzi, 2026)

Measures how much gradient signal survives the backward pass through the LM head,
comparing baseline (single linear) vs EXP-007 (3 multi-head with bottleneck).

Key metrics:
  1. Gradient Survival Ratio: ||dL/dh|| / ||dL/dlogits||
  2. Effective Rank of dL/dh — how many backbone gradient dimensions are utilized
  3. SVD spectrum of the LM head weight(s) — structural bottleneck
  4. Head similarity (EXP-007) — cosine sim between heads
  5. Jacobian rank analysis — theoretical gradient flow capacity

Both models are loaded by exec'ing only the class/function definitions from the
training scripts (before the global training code starts).

Usage:
  python gradient_bottleneck_analysis.py              # compare both
  python gradient_bottleneck_analysis.py --baseline   # baseline only
  python gradient_bottleneck_analysis.py --exp007     # EXP-007 only
  python gradient_bottleneck_analysis.py --steps=50   # train N steps before measuring
"""

import os
import sys
import argparse

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader


def _load_classes_from_script(filepath, stop_marker="# Hyperparameters"):
    """
    Load GPTConfig, GPT (and helpers) from a training script by exec'ing only
    the class/function definitions — everything BEFORE the global training code.
    Returns a namespace dict with the defined symbols.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    # Find where global training code begins (Hyperparameters section)
    # We look for common markers that signal end of class definitions
    markers = [
        "# Hyperparameters",
        "# ---------------------------------------------------------------------------\n# Hyperparameters",
        "\nASPECT_RATIO",
        "\nTOTAL_BATCH_SIZE",
    ]
    cut_idx = len(source)
    for marker in markers:
        idx = source.find(marker)
        if idx != -1 and idx < cut_idx:
            cut_idx = idx

    class_source = source[:cut_idx]

    # Build a namespace with necessary imports already available
    ns = {
        "__builtins__": __builtins__,
        "__name__": "__analysis__",
        "__file__": filepath,
    }
    exec(class_source, ns)
    return ns


def compute_effective_rank(matrix):
    """Effective rank via entropy of normalized singular values."""
    S = torch.linalg.svdvals(matrix.float().detach())
    S_n = S / (S.sum() + 1e-10)
    eff_rank = torch.exp(-torch.sum(S_n * torch.log(S_n + 1e-10))).item()
    return eff_rank, S.cpu().numpy()


def compute_gradient_survival(grad_h, grad_logits):
    h_norm = grad_h.float().norm().item()
    logit_norm = grad_logits.float().norm().item()
    return h_norm / (logit_norm + 1e-10), h_norm, logit_norm


def cosine_similarity_matrices(A, B):
    a = A.float().flatten()
    b = B.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-10)).item()


def analyze_baseline(device, tokenizer, train_steps=0, n_embd=512):
    print("\n" + "=" * 70)
    print("  BASELINE: Single LM Head (D -> V)")
    print("=" * 70)

    ns = _load_classes_from_script(os.path.join(_root, "train.py"))
    BaseConfig = ns["GPTConfig"]
    BaseGPT = ns["GPT"]

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    V = tokenizer.get_vocab_size()
    if n_embd % 128 != 0:
        raise ValueError(f"n_embd must be a multiple of 128, got {n_embd}")
    n_head = n_embd // 128

    config = BaseConfig(
        sequence_len=512, vocab_size=V,
        n_layer=12, n_head=n_head, n_kv_head=n_head, n_embd=n_embd,
        window_pattern="SSSL", compute_dtype=torch.bfloat16,
    )

    with torch.device("meta"):
        model = BaseGPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.train()

    D = config.n_embd
    print(f"\n  D={D}, V={V}, V/D ratio={V / D:.1f}")
    print(f"  LM head shape: ({V}, {D}), Max gradient rank: {D}")

    # SVD of LM head weight
    W = model.lm_head.weight.detach().float()
    eff_rank_W, svd_W = compute_effective_rank(W)
    print(f"\n  [Weight SVD] LM head eff_rank: {eff_rank_W:.1f} / {min(V, D)}")
    print(f"  Top-5 SVs: {svd_W[:5]}")
    print(f"  Condition number: {svd_W[0] / (svd_W[-1] + 1e-10):.1f}")
    print(f"  Energy in top-10 SVs: {svd_W[:10].sum() / (svd_W.sum() + 1e-10) * 100:.1f}%")

    # Train if requested
    train_loader = make_dataloader(tokenizer, 16, config.sequence_len, "train")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    if train_steps > 0:
        print(f"\n  Training {train_steps} steps before measuring...")
        optimizer = model.setup_optimizer()
        for step in range(train_steps):
            x, y, _ = next(train_loader)
            with autocast_ctx:
                loss = model(x, y)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print(f"    step {step}: loss={loss.item():.4f}")
        W = model.lm_head.weight.detach().float()
        eff_rank_W, svd_W = compute_effective_rank(W)
        print(f"\n  [After training] LM head eff_rank: {eff_rank_W:.1f} / {min(V, D)}")
        print(f"  Condition number: {svd_W[0] / (svd_W[-1] + 1e-10):.1f}")

    # Gradient flow analysis via hooks
    captured = {}

    def hook_logits(module, grad_input, grad_output):
        captured["grad_logits"] = grad_output[0].detach()

    h1 = model.lm_head.register_full_backward_hook(hook_logits)

    n_samples = 5
    survival_ratios = []
    eff_ranks_h = []
    eff_ranks_logits = []

    for _ in range(n_samples):
        x, y, _ = next(train_loader)
        model.zero_grad(set_to_none=True)

        with autocast_ctx:
            backbone_out = model.forward_backbone(x)
        backbone_out.retain_grad()
        with autocast_ctx:
            loss = model.forward_head(backbone_out, y)
        loss.backward()

        grad_h = backbone_out.grad
        grad_logits = captured.get("grad_logits")

        if grad_h is not None and grad_logits is not None:
            BT = grad_h.shape[0] * grad_h.shape[1]
            gh = grad_h.reshape(BT, -1)
            gl = grad_logits.reshape(BT, -1)

            ratio, _, _ = compute_gradient_survival(gh, gl)
            survival_ratios.append(ratio)

            sample_size = min(512, BT)
            idx = torch.randperm(BT)[:sample_size]
            eff_rank_h, _ = compute_effective_rank(gh[idx])
            eff_ranks_h.append(eff_rank_h)
            eff_rank_l, _ = compute_effective_rank(gl[idx])
            eff_ranks_logits.append(eff_rank_l)

    h1.remove()

    print(f"\n  [Gradient Flow] ({n_samples} samples)")
    print(f"  Survival ratio ||dL/dh|| / ||dL/dlogits||: {np.mean(survival_ratios):.6f} +/- {np.std(survival_ratios):.6f}")
    print(f"  Effective rank of dL/dh (backbone):  {np.mean(eff_ranks_h):.1f} / {D}")
    print(f"  Effective rank of dL/dlogits (logits): {np.mean(eff_ranks_logits):.1f} / {V}")
    print(f"  Gradient info preserved: ~{np.mean(survival_ratios) * 100:.2f}% of norm")

    result = (np.mean(survival_ratios), np.mean(eff_ranks_h))
    del model
    torch.cuda.empty_cache()
    return result


def analyze_exp007(device, tokenizer, train_steps=0):
    print("\n" + "=" * 70)
    print("  EXP-007: Multi-Head Output (3x D->160->GELU->V)")
    print("=" * 70)

    ns = _load_classes_from_script(
        os.path.join(_root, "experiments", "train_exp007_multi_head.py"))
    Exp007Config = ns["GPTConfig"]
    Exp007GPT = ns["GPT"]

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    V = tokenizer.get_vocab_size()
    D = 512
    D_k = 160
    N_heads = 3

    config = Exp007Config(
        sequence_len=512, vocab_size=V,
        n_layer=12, n_head=4, n_kv_head=4, n_embd=D,
        window_pattern="SSSL", compute_dtype=torch.bfloat16,
        num_output_heads=N_heads, output_head_dim=D_k,
    )

    with torch.device("meta"):
        model = Exp007GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.train()

    print(f"\n  D={D}, V={V}, D_k={D_k}, N_heads={N_heads}")
    print(f"  Each head: ({D} -> {D_k} -> GELU -> {V})")
    print(f"  Max grad rank per head: {D_k}")
    print(f"  Max grad rank total: min({N_heads}*{D_k}, {D}) = {min(N_heads * D_k, D)}")
    print(f"  Bottleneck factor vs baseline: {min(N_heads * D_k, D) / D:.2f}")

    # SVD per head
    print(f"\n  [Weight SVD per output head]")
    all_proj_weights = []
    all_out_weights = []
    for i, head in enumerate(model.output_heads):
        W_proj = head.proj.weight.detach().float()
        W_out = head.out.weight.detach().float()
        eff_rank_proj, svd_proj = compute_effective_rank(W_proj)
        eff_rank_out, svd_out = compute_effective_rank(W_out)
        print(f"  Head {i}: proj eff_rank={eff_rank_proj:.1f}/{min(D_k, D)}, "
              f"out eff_rank={eff_rank_out:.1f}/{min(V, D_k)}, "
              f"proj cond={svd_proj[0] / (svd_proj[-1] + 1e-10):.1f}")
        all_proj_weights.append(W_proj)
        all_out_weights.append(W_out)

    # Head similarity
    print(f"\n  [Head Similarity (cosine)]")
    for i in range(N_heads):
        for j in range(i + 1, N_heads):
            sim_proj = cosine_similarity_matrices(all_proj_weights[i], all_proj_weights[j])
            sim_out = cosine_similarity_matrices(all_out_weights[i], all_out_weights[j])
            print(f"  Heads {i}-{j}: proj_sim={sim_proj:.4f}, out_sim={sim_out:.4f}")

    # Train if requested
    train_loader = make_dataloader(tokenizer, 16, config.sequence_len, "train")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    if train_steps > 0:
        print(f"\n  Training {train_steps} steps before measuring...")
        optimizer = model.setup_optimizer()
        for step in range(train_steps):
            x, y, _ = next(train_loader)
            with autocast_ctx:
                loss = model(x, y)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print(f"    step {step}: loss={loss.item():.4f}")

        print(f"\n  [After training] Head similarity:")
        for i in range(N_heads):
            for j in range(i + 1, N_heads):
                sim_proj = cosine_similarity_matrices(
                    model.output_heads[i].proj.weight, model.output_heads[j].proj.weight)
                sim_out = cosine_similarity_matrices(
                    model.output_heads[i].out.weight, model.output_heads[j].out.weight)
                print(f"  Heads {i}-{j}: proj_sim={sim_proj:.4f}, out_sim={sim_out:.4f}")

    # Gradient flow analysis
    n_samples = 5
    backbone_norms = []
    eff_ranks_h = []

    for _ in range(n_samples):
        x, y, _ = next(train_loader)
        model.zero_grad(set_to_none=True)

        B, T = x.size()
        with autocast_ctx:
            cos_sin = model.cos[:, :T], model.sin[:, :T]
            backbone_x = model.transformer.wte(x)
            backbone_x = F.rms_norm(backbone_x, (backbone_x.size(-1),))
            x0 = backbone_x
            for i, block in enumerate(model.transformer.h):
                backbone_x = model.resid_lambdas[i] * backbone_x + model.x0_lambdas[i] * x0
                ve = model.value_embeds[str(i)](x) if str(i) in model.value_embeds else None
                backbone_x = block(backbone_x, ve, cos_sin, model.window_sizes[i])
            backbone_x = F.rms_norm(backbone_x, (backbone_x.size(-1),))

        backbone_x.retain_grad()

        with autocast_ctx:
            logits = sum(head(backbone_x) for head in model.output_heads)
        logits_f = logits.float()
        softcap = 15
        logits_f = softcap * torch.tanh(logits_f / softcap)
        loss = F.cross_entropy(logits_f.view(-1, logits_f.size(-1)), y.view(-1), ignore_index=-1)
        loss.backward()

        grad_h = backbone_x.grad
        if grad_h is not None:
            BT = B * T
            gh = grad_h.reshape(BT, -1)
            backbone_norms.append(gh.float().norm().item())
            sample_size = min(512, BT)
            idx = torch.randperm(BT)[:sample_size]
            eff_rank_h, _ = compute_effective_rank(gh[idx])
            eff_ranks_h.append(eff_rank_h)

    print(f"\n  [Gradient Flow] ({n_samples} samples)")
    print(f"  ||dL/dh|| (backbone grad norm): {np.mean(backbone_norms):.6f}")
    print(f"  Effective rank of dL/dh: {np.mean(eff_ranks_h):.1f} / {D}")

    # Jacobian rank analysis (linear approximation, ignoring GELU)
    print(f"\n  [Jacobian Rank Analysis (current weights)]")
    J_combined = torch.zeros(V, D, device=device)
    per_head_J = []
    for head in model.output_heads:
        J_k = head.out.weight.float() @ head.proj.weight.float()  # (V, D)
        per_head_J.append(J_k)
        J_combined += J_k

    eff_rank_J, svd_J = compute_effective_rank(J_combined)
    print(f"  Combined Jacobian eff_rank: {eff_rank_J:.1f} / {min(V, D)}")
    print(f"  Top-5 SVs: {svd_J[:5]}")
    print(f"  Condition number: {svd_J[0] / (svd_J[min(D, V) - 1] + 1e-10):.1f}")

    for i, J_k in enumerate(per_head_J):
        eff_rank_k, _ = compute_effective_rank(J_k)
        print(f"  Head {i} Jacobian eff_rank: {eff_rank_k:.1f} / {min(V, D)} (max possible: {D_k})")

    # Jacobian similarity
    print(f"\n  [Jacobian Similarity between heads]")
    for i in range(N_heads):
        for j in range(i + 1, N_heads):
            sim = cosine_similarity_matrices(per_head_J[i], per_head_J[j])
            print(f"  J_{i} vs J_{j}: cosine_sim={sim:.4f}")

    # Gradient subspace overlap
    print(f"\n  [Backward Gradient Subspace Analysis]")
    subspaces = []
    for i, head in enumerate(model.output_heads):
        _, _, Vt = torch.linalg.svd(head.proj.weight.float(), full_matrices=False)
        subspaces.append(Vt)  # (D_k, D)

    for i in range(N_heads):
        for j in range(i + 1, N_heads):
            overlap = torch.linalg.norm(subspaces[i] @ subspaces[j].T).item() / D_k
            print(f"  Heads {i}-{j} subspace overlap: {overlap:.4f} "
                  f"(1.0=identical, ~{1/D_k:.3f}=random)")

    all_V = torch.cat(subspaces, dim=0)
    eff_rank_union, _ = compute_effective_rank(all_V)
    print(f"  Union eff_rank: {eff_rank_union:.1f} / {min(N_heads * D_k, D)}")
    print(f"  (If ={N_heads * D_k} → fully diverse; if ={D_k} → collapsed)")

    result = (np.mean(backbone_norms) if backbone_norms else 0,
              np.mean(eff_ranks_h) if eff_ranks_h else 0)
    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--exp007", action="store_true")
    parser.add_argument("--steps", type=int, default=0,
                        help="Train N steps before measuring (0 = init only)")
    parser.add_argument("--tokenizer-dir", type=str, default=None,
                        help="Tokenizer directory override for vocab sweep experiments")
    parser.add_argument("--n-embd", type=int, default=512,
                        help="Hidden dimension D for baseline analysis (multiple of 128)")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_directory(args.tokenizer_dir)
    V = tokenizer.get_vocab_size()

    print("=" * 70)
    print(f"  Gradient Bottleneck Analysis")
    print(f"  Tokenizer dir: {getattr(tokenizer, 'source_dir', '(default)')}")
    print(f"  Vocab size: {V:,}, Train steps before measure: {args.steps}")
    print(f"  Baseline D: {args.n_embd}")
    print("=" * 70)

    run_bl = not args.exp007 or args.baseline
    run_exp = not args.baseline or args.exp007
    if not args.baseline and not args.exp007:
        run_bl = run_exp = True

    bl_survival = bl_rank = exp_survival = exp_rank = None

    if run_bl:
        bl_survival, bl_rank = analyze_baseline(device, tokenizer, args.steps, args.n_embd)
    if run_exp:
        exp_survival, exp_rank = analyze_exp007(device, tokenizer, args.steps)

    if bl_survival is not None and exp_survival is not None:
        print("\n" + "=" * 70)
        print("  COMPARISON")
        print("=" * 70)
        print(f"  Backbone grad norm — BL: {bl_survival:.6f}, EXP: {exp_survival:.6f}")
        print(f"  Backbone eff rank  — BL: {bl_rank:.1f}/{args.n_embd}, EXP: {exp_rank:.1f}/512")
        print()
        print("  Interpretation:")
        print("  - If EXP eff_rank < BL eff_rank -> multi-head bottleneck is worse")
        print("  - If head subspace overlap > 0.5 -> heads are collapsing")
        print("  - If Jacobian eff_rank << D -> gradient can't explore full space")

    print("\n" + "=" * 70)
    print(f"  Structural regime: D={args.n_embd}, V={V} -> V/D={V / args.n_embd:.1f}")
    print(f"  Baseline: rank(J^T) <= {args.n_embd}")
    print(f"  EXP-007:  rank(sum J_k^T) <= min(3*160, 512) = {min(480, 512)}")
    print(f"  => Multi-head reduces gradient capacity by ~6% even best case")
    print(f"  => If heads collapse, effective rank drops to ~160 (69% reduction)")
    print("=" * 70)


if __name__ == "__main__":
    main()
