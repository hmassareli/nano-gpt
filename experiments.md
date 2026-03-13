# Experiment Log

Investigation into the LM Head gradient bottleneck during pretraining.
Reference paper: "Lost in Backpropagation: The LM Head is a Gradient Bottleneck" (Godey & Artzi, 2026)

Base model: 71M params, vocab=8192, n_embd=512, 12 layers, seq_len=512

---

## EXP-001: Two-Stage Head Update

**Date:** 2026-03-13
**Hypothesis:** Updating the head before the backbone provides less noisy gradients to the backbone, improving convergence.
**Mechanism:** Stage 1 — forward backbone with no_grad, only head gets gradients, step on head. Stage 2 — full forward with updated head, only backbone gets gradients, step on backbone. Same batch reused.

**Results:**
- Initial loss: 7.66 (two-stage) vs 9.01 (baseline) — pre-adjusted head helps
- Loss at step 50: 5.09 vs 5.08 — tied
- Final loss (100 steps): 4.49 vs 4.49 — tied
- val_bpb: 1.5636 vs 1.5645 — negligible delta (0.0009)
- Cost per step: ~53s vs ~38s (+40% overhead from double forward)

**Conclusion:** NEGATIVE. The head initialization advantage is transient — the baseline catches up by ~25 steps. The double forward cost nullifies the gain. Confirms that the bottleneck problem is **continuous** (not just initialization): the head degrades its conditioning throughout training, fixing it once at the start is not enough.

**Insight:** The early gain validates the existence of the bottleneck (aligns with the paper). But the solution needs to be continuous (regularization) or structural (architecture), not one-shot (initialization).

**Logs:** `benchmark_logs/bench_20260313_015605_100steps.log`, `benchmark_logs/teste1_two_stage`

---

## Discarded ideas (from early brainstorming)

### Lookahead on head (N head steps per backbone step)
**Idea:** Normal end-to-end forward/backward, then N extra optimizer steps on just the head using the same gradients already computed.
**Why discarded:** The extra head steps use hidden states from the backbone *before* its update — same staleness problem as two-stage, just milder. Also, Adam's momentum buffers get corrupted by the extra steps (different effective learning rate trajectory). Low novelty (similar to inner-loop optimization in meta-learning).

### Differential learning rate for head
**Idea:** Simply increase lm_head LR by 2-3x relative to backbone.
**Why discarded:** This is a well-known technique (popularized by fastai). Zero novelty. Also, Adam normalizes by sqrt(v_t), so scaling LR has diminishing returns once momentums adapt.

### Partial gradient accumulation (head accumulates more batches)
**Idea:** Run everything end-to-end but accumulate more micro-batches for the head before stepping the backbone.
**Why discarded:** Variant of two-stage with the same desynchronization problem. The head trains on backbone features that are about to change.

---

## Planned experiments

### EXP-002: Spectral Head Init (Embedding SVD)
**Hypothesis:** Initializing lm_head with the pseudo-inverse of wte provides high effective dimension from step 0, with zero training overhead.
**Status:** Not started

### EXP-003: Head Conditioning Regularization
**Hypothesis:** Penalizing ||W·W^T - αI|| keeps singular values uniform, preserving gradient flow throughout training.
**Status:** Not started

### EXP-004: Soft Weight Tying (head ↔ embedding)
**Hypothesis:** λ||W_head - W_te^T||² continuously anchors the head to the semantic structure of the embedding, preventing conditioning degradation.
**Status:** Not started

### EXP-005: Contrastive Auxiliary Loss (Head Bypass)
**Hypothesis:** A contrastive loss on the hidden states gives the backbone a gradient channel that bypasses the head bottleneck.
**Status:** Not started

### EXP-006: Expanded Factored Head (512→2048→8192)
**Hypothesis:** Expanding the intermediate head dimension increases the effective gradient dimension. Paper shows +8 pts on benchmarks with d=4096 vs d=32.
**Status:** Not started
