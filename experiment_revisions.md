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

### EXP-029.2 changed the diagnosis again

The most informative recent signal is not that the JEPA-style line failed, but that `EXP-029.2` behaved like a phase-specific helper.

- `EXP-029.2` was the strongest member of the `029` family.
- `EXP-029.2.2` (`lambda=0.25`) improved materially through the middle of training.
- The gain did not hold to the end of the 1000-step run.

Working interpretation: the auxiliary latent objective is probably carrying some real predictive signal, but the fixed target interface and/or fixed auxiliary weight become misaligned with the late-stage CE objective. This now looks more like a target-design problem than evidence that "future latent signal is useless".

Important measurement note: for `EXP-029.x`, comparisons against baseline should be done with experiment `ce` versus baseline `loss`, not experiment total `loss`, because the auxiliary term is not directly comparable to baseline CE.

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

## EXP-029.2.4: Projected Next-Token Latent Target

Keep the `EXP-029.2` next-token setup, but match a learned projection of the future hidden state instead of the raw `h_12(t+1)` vector.

- Goal: move the auxiliary target into a smaller, more head-adjacent subspace.
- Expected benefit: preserve the useful future signal from `029.2` while removing irrelevant hidden-state detail that may fight CE late in training.

## EXP-029.2.5: Next-Token Logit Distillation

Keep the `EXP-029.2` source layer and predictor, but distill the next-token output distribution instead of matching hidden coordinates.

- Goal: pass a signal that is directly tied to next-token ranking rather than full hidden geometry.
- Expected benefit: retain the useful predictive part of the auxiliary signal while avoiding the need to copy the whole late hidden state.

## EXP-029.2.6: Future-Window Latent Target

Replace the single-step future target with a short-horizon summary over `t+1:t+k`.

- Goal: make the auxiliary target less noisy and less dependent on one exact future hidden state.
- Expected benefit: encourage short-horizon predictive structure without locking the model to one brittle target geometry.

## EXP-016.1: Ortho-Reg As Diagnostic Control Only

Do not treat this as a primary optimization direction anymore.

- Use it only as a diagnostic control in future vocab sweeps.
- Goal: test whether stronger structural bottlenecks make `surv`/`eff` gains diverge even more clearly from loss gains.

## Decision Rules

- If `EXP-005.1` or `EXP-005.2` helps, contrastive bypass becomes a top-tier line.
- If `EXP-022.1` or `EXP-022.2` helps, the original EXP-022 failure was mostly geometric.
- If `EXP-029.1` helps quickly, JEPA-style latent supervision becomes the cleanest next family.
- If `EXP-029.2.4`, `EXP-029.2.5`, or `EXP-029.2.6` helps late, then the `029` family likely failed on target interface, not on the core idea of future latent supervision.
- If all bypass revisions fail, then the diagnosis needs to be revisited more aggressively.

## Recommended Order

1. EXP-022.1
2. EXP-005.1
3. EXP-022.2
4. EXP-005.2
5. EXP-029.2.3
6. EXP-029.2.4
7. EXP-029.2.5
8. EXP-029.2.6
