# Experiment Revisions

This file tracks corrective follow-ups for existing experiments.

Use the suffix convention `EXP-XX.Y` for revisions that preserve the same core idea but change geometry, targets, sampling, or measurement. Reserve new integer experiment IDs for genuinely new families.

## Why This File Exists

Several results are now better interpreted as "promising idea, wrong implementation or wrong geometric interface" rather than simple negative outcomes. The main examples are EXP-005 and EXP-022.

## Current Re-read Of Existing Results

### EXP-016 changed the diagnosis

The most informative result so far is not the best-performing model, but the negative result from EXP-016.

- `surv` improved materially.
- `eff` improved materially.
- Loss still got worse.

Working interpretation: improving how the model uses the available hidden space $D$ is not enough. The dominant issue appears more structural than purely optimization-internal.

### Multi-head likely delays, not solves

The multi-head family improved diversity and sometimes helped early training, but did not sustain a win. Working interpretation: multiple heads can delay collapse, but they still feed gradients through the same structural compression bottleneck.

### EXP-022 should be treated as a geometry failure, not a bypass failure

The core bypass idea remains attractive. The likely issue is forcing the hidden state toward the raw input embedding geometry, which may conflict with cross-entropy.

## Proposed Corrective Sub-Experiments

## EXP-005.1: Contrastive Loss With Projection Head

Keep the core EXP-005 idea, but add a small MLP projection head before InfoNCE.

- Goal: let the contrastive objective shape a dedicated latent space instead of directly fighting the main hidden representation.
- Expected benefit: less interference with CE, stronger bypass signal.

## EXP-005.2: Structured Positives For Contrastive Bypass

Use richer positive definitions instead of only same-token identity.

- Same token under similar context.
- Neighboring positions with strong local dependency.
- Teacher/EMA-matched representations when available.

Goal: make the bypass objective less myopic and more semantically useful.

## EXP-022.1: Cosine-Normalized Embedding Loss

Keep EXP-022, but normalize both `h` and the target embedding, and use cosine loss.

- Goal: reduce scale conflict with CE.
- Expected benefit: preserve the bypass while avoiding MSE geometry mismatch.

## EXP-022.2: Predictor + Stop-Grad Embedding Bypass

Insert a small predictor `P(h)` and match `P(h)` to a stop-grad target embedding.

- Goal: BYOL-style asymmetry to reduce direct conflict between the CE geometry and the auxiliary target geometry.

## EXP-022.3: Learned-Projection Embedding Bypass

Project the token embedding through a learned target projection before matching.

- Goal: bridge EXP-022 toward EXP-026 without introducing a full learned target codec yet.

## EXP-029.1: Layer-6 To Layer-12 Same-Token Prediction

Predict `sg(h_12)` from `h_6` with a small predictor.

- Goal: the cleanest first JEPA-style latent bypass.
- Advantage: no dependency on raw embeddings and no direct pass through the LM head.

## EXP-029.2: Next-Token Latent Prediction

Predict a latent target associated with the next token rather than the same-position final hidden.

- Goal: align the latent objective more directly with autoregressive modeling.

## EXP-029.3: EMA Teacher Latent Target

Use a slow-moving teacher/EMA copy to generate the latent target.

- Goal: stabilize the target and reduce collapse risk.

## EXP-016.1: Ortho-Reg As Diagnostic Control Only

Do not treat this as a primary optimization direction anymore.

- Use it only as a diagnostic control in future vocab sweeps.
- Goal: test whether stronger structural bottlenecks make `surv`/`eff` gains diverge even more clearly from loss gains.

## Decision Rules

- If `EXP-005.1` or `EXP-005.2` helps, contrastive bypass becomes a top-tier line.
- If `EXP-022.1` or `EXP-022.2` helps, the original EXP-022 failure was mostly geometric.
- If `EXP-029.1` helps quickly, JEPA-style latent supervision becomes the cleanest next family.
- If all bypass revisions fail, then the diagnosis needs to be revisited more aggressively.

## Recommended Order

1. EXP-022.1
2. EXP-005.1
3. EXP-029.1
4. EXP-022.2
5. EXP-005.2
6. EXP-029.3