# Experiment Log

Investigation into the LM Head gradient bottleneck during pretraining.
Reference paper: "Lost in Backpropagation: The LM Head is a Gradient Bottleneck" (Godey & Artzi, 2026)

Base model: 71M params, vocab=8192, n_embd=512, 12 layers, seq_len=512

## Summary by Experiment

| Exp     | Status                       | Result                                                     | Conclusion                                                                        |
| ------- | ---------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------- |
| EXP-001 | Completed                    | Tied baseline by 100 steps; +40% step cost                 | Negative. One-shot head-first update does not solve a continuous bottleneck.      |
| EXP-002 | Benchmarked (100 steps)      | Worse than baseline mid-run; no durable gain               | Negative. Init-only fixes decay too fast.                                         |
| EXP-003 | Benchmarked (100 steps)      | No decisive improvement recorded in current notes          | Neutral. Orthogonality idea is plausible, but current setup looks too weak.       |
| EXP-004 | Benchmarked (100 steps)      | Harmful due to scale mismatch and conflicting roles        | Harmful. Soft tying fights head/embedding specialization.                         |
| EXP-005 | In progress                  | Initial implementation bug fixed; benchmark still pending  | Pending. Training-time bypass is plausible, but not validated yet.                |
| EXP-006 | Completed (100 steps)        | val_bpb 1.663 vs baseline 1.564                            | Negative. GELU alone does not help.                                               |
| EXP-007 | Completed (300 steps)        | Early crossover, but final val_bpb 1.157 vs 1.144 baseline | Mixed. Gradient diversification helps early, but collapses later.                 |
| EXP-008 | Planned / awaiting benchmark | No benchmark yet                                           | Pending. Speed-focused variant of EXP-007.                                        |
| EXP-009 | Reference only               | Not pursued as a real candidate                            | Rejected. Two-stage logic is fundamentally flawed.                                |
| EXP-010 | Planned / awaiting benchmark | No benchmark yet                                           | Pending. Tests whether head dropout preserves diversity.                          |
| EXP-011 | Planned                      | No benchmark yet                                           | Pending. Separate head dropout variant of EXP-007.                                |
| EXP-012 | Planned                      | No benchmark yet                                           | Pending. Tests multi-head with full rank and GELU.                                |
| EXP-013 | Planned                      | No benchmark yet                                           | Pending. Tests 4-head full-rank variant.                                          |
| EXP-014 | Planned                      | No benchmark yet                                           | Pending. Tests weight-space diversity preservation.                               |
| EXP-015 | Planned                      | No benchmark yet                                           | Pending. Tests activation-space diversity preservation.                           |
| EXP-016 | Planned                      | No benchmark yet                                           | Pending. Key test of whether single-head conditioning is enough.                  |
| EXP-017 | Planned                      | No benchmark yet                                           | Pending. Tests linear multi-head with per-head supervision.                       |
| EXP-018 | Planned                      | No benchmark yet                                           | Pending. Tests whether more supervised heads help at fixed rank.                  |
| EXP-019 | Planned                      | No benchmark yet                                           | Pending. Tests if geometric overlap regularization adds value beyond per-head CE. |
| EXP-020 | Planned                      | No benchmark yet                                           | Pending. Tests input-dependent routing as the anti-collapse mechanism.            |
| EXP-021 | Planned                      | No benchmark yet                                           | Pending. Direct bypass of the final LM head via deep supervision.                 |
| EXP-022 | Implemented                  | Code path exists; benchmark not recorded yet               | Pending. Very strong zero-parameter bypass hypothesis.                            |
| EXP-023 | Planned                      | No benchmark yet                                           | Pending. Capacity-heavy upper-bound control.                                      |
| EXP-024 | Planned                      | No benchmark yet                                           | Pending. Tests whether early softmax smoothing improves gradient rank.            |
| EXP-025 | Planned                      | No benchmark yet                                           | Pending. Most direct multi-head test of backward-channel diversity.               |

## Diagnostic Metrics to Track

Keep these in a separate diagnostics table, not in the main outcome table. The main table should answer "did it work?"; this one should answer "why did it work or fail?"

| Column   | Meaning                              | Why it matters                                                              | Applies to      |
| -------- | ------------------------------------ | --------------------------------------------------------------------------- | --------------- |
| `surv`   | Gradient survival ratio              | Direct proxy for how much signal reaches the backbone through the head      | All experiments |
| `eff`    | Head effective rank                  | Measures how much of the available gradient subspace is actually being used | All experiments |
| `rr`     | Rank ratio (`eff / D`)               | Normalized version of effective rank, easier to compare across widths       | All experiments |
| `top10e` | Top-10 singular value energy         | Detects concentration/collapse of gradient energy in a few directions       | All experiments |
| `cos`    | Mean cosine similarity between heads | Detects head collapse / loss of diversity                                   | Multi-head only |
| `uni`    | Union rank across heads              | Measures whether multiple heads cover more gradient directions jointly      | Multi-head only |

Recommended rule: use the same-step matched baseline for every metric comparison. Do not compare a 200-step experiment against a 1000-step baseline metric row; the diagnostics become misleading.

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
**Why discarded:** The extra head steps use hidden states from the backbone _before_ its update — same staleness problem as two-stage, just milder. Also, Adam's momentum buffers get corrupted by the extra steps (different effective learning rate trajectory). Low novelty (similar to inner-loop optimization in meta-learning).

### Differential learning rate for head

**Idea:** Simply increase lm_head LR by 2-3x relative to backbone.
**Why discarded:** This is a well-known technique (popularized by fastai). Zero novelty. Also, Adam normalizes by sqrt(v_t), so scaling LR has diminishing returns once momentums adapt.

### Partial gradient accumulation (head accumulates more batches)

**Idea:** Run everything end-to-end but accumulate more micro-batches for the head before stepping the backbone.
**Why discarded:** Variant of two-stage with the same desynchronization problem. The head trains on backbone features that are about to change.

---

## Planned experiments

#

## Design Experimental: Comparações Planejadas

### Tabela de variáveis controladas

| Exp      | Tipo               | Heads  | D_k | Max Rank | GELU    | Diversidade                     | Params extras |
| -------- | ------------------ | ------ | --- | -------- | ------- | ------------------------------- | ------------- |
| Baseline | single head        | 1      | —   | 512      | —       | —                               | 0             |
| 016      | single + ortho reg | 1      | —   | 512      | —       | $W^TW≈I$                        | 0             |
| 007      | multi-head         | 3      | 160 | 480      | sim     | —                               | 0             |
| 011      | multi-head         | 3      | 160 | 480      | sim     | head dropout                    | 0             |
| 012      | multi-head         | 2      | 256 | 512      | sim     | —                               | +0.3M         |
| 013      | multi-head         | 4      | 128 | 512      | sim     | —                               | +0.3M         |
| 014      | multi-head         | 4      | 128 | 512      | sim     | cosine penalty                  | +0.3M         |
| 015      | multi-head         | 4      | 128 | 512      | sim     | DeCov                           | +0.3M         |
| 017      | multi-head         | 2      | 256 | 512      | **não** | per-head CE                     | +0.3M         |
| 018      | multi-head         | 4      | 128 | 512      | **não** | per-head CE                     | +0.3M         |
| 019      | multi-head         | 4      | 128 | 512      | **não** | per-head CE + projector overlap | +0.3M         |
| 020      | multi-head         | 4      | 128 | 512      | **não** | token gating + load balance     | +0.3M         |
| 025      | multi-head         | 4      | 128 | 512      | **não** | per-head CE + grad diversity    | +0.3M         |
| 021      | deep supervision   | 1      | —   | 512      | —       | —                               | +4.2M         |
| 022      | emb space loss     | 1      | —   | 512      | —       | —                               | 0             |
| 023      | MoS                | 4 full | D   | 4×512    | —       | —                               | +16.8M        |
| 024      | temp schedule      | 1      | —   | 512      | —       | —                               | 0             |

### Comparações diretas planejadas

1. **Per-head supervision preserva canais?**
   - 017 vs baseline linear implícito: se 017 funcionar melhor que a leitura teórica do "soma linear colapsada", a CE por head está impedindo carona funcional.

2. **Número de heads sob supervisão individual**:
   - 017 ↔ 018 (2 vs 4, ambos sem GELU e com per-head CE)
   - Se 018 > 017, mais canais supervisionados ajudam mesmo com mesmo rank total.

3. **Supervisão individual basta ou ainda precisa regularização geométrica?**
   - 018 ↔ 019 (per-head CE vs per-head CE + projector overlap)
   - Se 019 > 018, a supervisão individual evita carona, mas não evita colapso geométrico do subespaço.

4. **Soma fixa vs roteamento dependente do input**:
   - 019 ↔ 020
   - Se 020 > 019, o ganho vem de preservar canais por contexto, não só de manter pesos diferentes.

5. **Regularizar subespaço ou regularizar o backward diretamente?**
   - 019 ↔ 025
   - Se 025 > 019, até overlap de subespaço ainda é proxy e o objeto certo é mesmo o gradiente em $h$.

6. **Multi-head preservado vs single-head regularizado**:
   - 019/020/025 vs 016
   - Se 016 ≥ 019/020/025 → conditioning do head único resolve quase tudo
   - Se 019/020/025 > 016 → decomposição em canais tem valor próprio quando o colapso é controlado

7. **Bypass do head vs melhorar o head**:
   - 021/022 (bypass) vs 016/020/025
   - Se bypass ganha → o gargalo V→D segue dominante
   - Se 020/025/016 ganha → o principal problema está em como o espaço D é usado

8. **Custo-benefício**:
   - 022 (0 params) vs 023 (16.8M params) → quão longe vai a solução zero-custo?

---

## Recommended Next Benchmarks

1. **017 vs 018**
   - Primeiro teste barato da hipótese nova. Mede se mais heads supervisionadas individualmente ajudam mesmo sem GELU.

2. **018 vs 019**
   - Isola se a per-head CE já basta ou se ainda existe colapso geométrico entre projetores.

3. **019 vs 020**
   - Testa soma fixa versus roteamento por contexto. É o confronto mais importante da linha multi-head preservada.

4. **019 vs 025**
   - Compara diversidade em pesos versus diversidade no próprio gradiente do hidden. Deve ser rodado depois dos testes acima porque o 025 é o mais caro.

5. **Melhor entre 019/020/025 vs 016**
   - Decide se o esforço multi-head está realmente comprando algo além de um head único bem condicionado.

6. **Melhor entre 016 e multi-head vs 022**
   - Decide se vale mais insistir em melhorar o head ou bypassar o problema.

---

## Strategic Insight

O esforço de multi-head **pode sim ser em vão** se a maior parte do problema vier do gargalo estrutural $V \to D$ ou se um head único bem condicionado (EXP-016) capturar quase todo o ganho disponível. Em outras palavras: se o SGD já consegue usar quase todo o espaço $D$ depois de uma regularização simples, então separar em heads vira complexidade desnecessária.

O caso em que multi-head ainda vale a pena é mais específico: quando há subutilização persistente de dimensões **dentro** de $D$ e quando diferentes canais conseguem entregar sinais de backward complementares ao backbone. É por isso que a nova linha 017-020/025 mudou de foco: ela não assume mais que "separar heads" por si só resolve. Ela testa explicitamente se esses canais continuam distintos no task-space, no roteamento por contexto, ou no próprio gradiente em $h$.

Em termos práticos, eu trataria a linha multi-head agora como uma hipótese forte, mas já bem falsificável:

- Se 017/018 falharem, a separação linear pura com supervisão individual já não sustenta a tese.
- Se 020/025 não superarem 016, a decomposição em canais provavelmente não compensa a complexidade.
- Se 022 funcionar muito bem, o diagnóstico mais forte passa a ser: o problema relevante não é subutilização de dimensões dentro de $D$, e sim o fato de forçar toda supervisão a passar por um LM head final.
