# Experiment Log

Investigation into the LM Head gradient bottleneck during pretraining.
Reference paper: "Lost in Backpropagation: The LM Head is a Gradient Bottleneck" (Godey & Artzi, 2026)

Base model: 71M params, vocab=8192, n_embd=512, 12 layers, seq_len=512

## Summary by Experiment

| Exp     | Status                       | Result                                                        | Conclusion                                                                                     |
| ------- | ---------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| EXP-001 | Completed                    | Tied baseline by 100 steps; +40% step cost                    | Negative. One-shot head-first update does not solve a continuous bottleneck.                   |
| EXP-002 | Benchmarked (100 steps)      | Worse than baseline mid-run; no durable gain                  | Negative. Init-only fixes decay too fast.                                                      |
| EXP-003 | Benchmarked (100 steps)      | No decisive improvement recorded in current notes             | Neutral. Orthogonality idea is plausible, but current setup looks too weak.                    |
| EXP-004 | Benchmarked (100 steps)      | Harmful due to scale mismatch and conflicting roles           | Harmful. Soft tying fights head/embedding specialization.                                      |
| EXP-005 | Revisions ready              | `EXP-005.1/.2` implemented; main benchmark still pending      | Promising. Contrastive bypass still looks live if the auxiliary space is better designed.      |
| EXP-006 | Completed (100 steps)        | val_bpb 1.663 vs baseline 1.564                               | Negative. GELU alone does not help.                                                            |
| EXP-007 | Completed (300 steps)        | Early crossover, but final val_bpb 1.157 vs 1.144 baseline    | Mixed. Gradient diversification helps early, but collapses later.                              |
| EXP-008 | Planned / awaiting benchmark | No benchmark yet                                              | Pending. Speed-focused variant of EXP-007.                                                     |
| EXP-009 | Reference only               | Not pursued as a real candidate                               | Rejected. Two-stage logic is fundamentally flawed.                                             |
| EXP-010 | Planned / awaiting benchmark | No benchmark yet                                              | Pending. Tests whether head dropout preserves diversity.                                       |
| EXP-011 | Planned                      | No benchmark yet                                              | Pending. Separate head dropout variant of EXP-007.                                             |
| EXP-012 | Planned                      | No benchmark yet                                              | Pending. Tests multi-head with full rank and GELU.                                             |
| EXP-013 | Planned                      | No benchmark yet                                              | Pending. Tests 4-head full-rank variant.                                                       |
| EXP-014 | Planned                      | No benchmark yet                                              | Pending. Tests weight-space diversity preservation.                                            |
| EXP-015 | Planned                      | No benchmark yet                                              | Pending. Tests activation-space diversity preservation.                                        |
| EXP-016 | Benchmarked / reinterpreted  | `surv`/`eff` subiram, mas o loss piorou                       | Important negative. Better use of $D$ alone did not translate into better learning.            |
| EXP-017 | Planned                      | No benchmark yet                                              | Pending. Tests linear multi-head with per-head supervision.                                    |
| EXP-018 | Planned                      | No benchmark yet                                              | Pending. Tests whether more supervised heads help at fixed rank.                               |
| EXP-019 | Planned                      | No benchmark yet                                              | Pending. Tests if geometric overlap regularization adds value beyond per-head CE.              |
| EXP-020 | Planned                      | No benchmark yet                                              | Pending. Tests input-dependent routing as the anti-collapse mechanism.                         |
| EXP-021 | Planned                      | No benchmark yet                                              | Pending. Direct bypass of the final LM head via deep supervision.                              |
| EXP-022 | Benchmarked / needs redesign | Initial embedding-loss run was harmful                        | Promising idea, but current geometry likely conflicts with CE rather than disproving bypass.   |
| EXP-023 | Planned                      | No benchmark yet                                              | Pending. Capacity-heavy upper-bound control.                                                   |
| EXP-024 | Planned                      | No benchmark yet                                              | Pending. Tests whether early softmax smoothing improves gradient rank.                         |
| EXP-025 | Planned                      | No benchmark yet                                              | Pending. Most direct multi-head test of backward-channel diversity.                            |
| EXP-026 | Planned                      | No benchmark yet                                              | Pending. Learned latent target for supervision instead of reusing input embeddings.            |
| EXP-027 | Planned                      | No benchmark yet                                              | Pending. Adaptive extra compute only on ambiguous tokens.                                      |
| EXP-028 | Planned                      | No benchmark yet                                              | Pending. Coarse-to-fine output decomposition for large vocabularies.                           |
| EXP-029 | Benchmarked / mixed          | `029.2` strongest mid-run, but `029.2.2` gave gains back late | Mixed but alive. The family now looks more limited by target interface than by lack of signal. |
| EXP-030 | Planned                      | No benchmark yet                                              | Pending. Input-dependent routed latent readout instead of static multi-head sum.               |

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

| Exp      | Tipo                    | Heads       | D_k | Max Rank  | GELU     | Diversidade                     | Params extras |
| -------- | ----------------------- | ----------- | --- | --------- | -------- | ------------------------------- | ------------- |
| Baseline | single head             | 1           | —   | 512       | —        | —                               | 0             |
| 016      | single + ortho reg      | 1           | —   | 512       | —        | $W^TW≈I$                        | 0             |
| 007      | multi-head              | 3           | 160 | 480       | sim      | —                               | 0             |
| 011      | multi-head              | 3           | 160 | 480       | sim      | head dropout                    | 0             |
| 012      | multi-head              | 2           | 256 | 512       | sim      | —                               | +0.3M         |
| 013      | multi-head              | 4           | 128 | 512       | sim      | —                               | +0.3M         |
| 014      | multi-head              | 4           | 128 | 512       | sim      | cosine penalty                  | +0.3M         |
| 015      | multi-head              | 4           | 128 | 512       | sim      | DeCov                           | +0.3M         |
| 017      | multi-head              | 2           | 256 | 512       | **não**  | per-head CE                     | +0.3M         |
| 018      | multi-head              | 4           | 128 | 512       | **não**  | per-head CE                     | +0.3M         |
| 019      | multi-head              | 4           | 128 | 512       | **não**  | per-head CE + projector overlap | +0.3M         |
| 020      | multi-head              | 4           | 128 | 512       | **não**  | token gating + load balance     | +0.3M         |
| 025      | multi-head              | 4           | 128 | 512       | **não**  | per-head CE + grad diversity    | +0.3M         |
| 021      | deep supervision        | 1           | —   | 512       | —        | —                               | +4.2M         |
| 022      | emb space loss          | 1           | —   | 512       | —        | —                               | 0             |
| 023      | MoS                     | 4 full      | D   | 4×512     | —        | —                               | +16.8M        |
| 024      | temp schedule           | 1           | —   | 512       | —        | —                               | 0             |
| 026      | learned target codec    | 1           | —   | 512       | —        | learned latent target           | TBD           |
| 027      | adaptive head compute   | 1+refine    | —   | 512       | opcional | uncertainty-triggered compute   | small/TBD     |
| 028      | hierarchical vocab head | coarse+fine | —   | TBD       | opcional | coarse-to-fine routing          | TBD           |
| 029      | JEPA latent aux         | 1           | —   | 512       | opcional | latent prediction bypass        | small/TBD     |
| 030      | gated latent readout    | K gated     | TBD | token-dep | opcional | token-dependent routing         | TBD           |

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

9. **Bypass com target herdado vs target aprendido**:
   - 022 ↔ 026
   - Se 026 > 022, o problema do 022 é geometria/alvo ruim, não a ideia de bypass em si.

10. **Bypass latente vs bypass supervisionado por embedding**:

- 026/029 ↔ 022
- Se 029 > 022, prever um alvo latente interno é mais natural do que empurrar o hidden para o embedding do token.

11. **Mexer no head vs alocar compute adaptativo**:

- 027/028/030 ↔ 016/019/020
- Se 027/028/030 ganham, o problema não é apenas geometria do head, mas esforço uniforme demais para um problema de saída heterogêneo.

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

7. **022 vs 026 vs 029**
   - Decide qual forma de bypass merece virar a linha principal: embedding herdado, target aprendido, ou predição latente interna.

8. **Melhor bypass vs 027/028/030**
   - Decide se o próximo salto vem de abandonar o head como canal principal de treino ou de tornar a leitura de saída mais adaptativa.

---

## Strategic Insight

### Current Working Interpretation

O resultado mais informativo até aqui nao e o melhor benchmark isolado, e sim o comportamento do **EXP-016**: as metricas que o paper aponta como desejaveis (`surv`, `eff`, `rr`) podem melhorar bastante sem produzir melhor loss final. A leitura mais forte disso e que **melhorar apenas o uso interno do espaco $D$ nao basta**. Em outras palavras, ha evidencia de que o problema relevante nao e so "otimizacao ruim dentro de $D$", mas sim a propria compressao estrutural $V \to D$.

Essa releitura tambem organiza melhor a familia multi-head. O ganho inicial do EXP-007 nao contradiz a tese estrutural: ele sugere apenas que mais diversidade de gradiente ajuda no early training. O fato de a vantagem desaparecer e de a familia multi-head nao sustentar ganho duravel aponta para a interpretacao mais dura: **multi-head atrasa o problema, mas nao remove o gargalo fundamental**.

O resultado ruim do EXP-022 tambem precisa ser interpretado com cuidado. Ele nao invalida bypass. O mais provavel e que a implementacao atual force uma geometria inadequada (`h` aproximando embedding bruto do token) e entre em conflito com a CE. O ponto importante permanece: um gradiente direto no hidden, sem passar pelo LM head, continua sendo a linha conceitualmente mais alinhada com o diagnostico.

### What The Current Results Already Rule Out

- Melhorar condicionamento do head por si so nao parece suficiente.
- Fatorar ou multiplicar heads sem alterar o gargalo estrutural nao parece suficiente.
- Diversidade temporaria no backward ajuda cedo, mas nao se sustenta sozinha.

### What Still Looks Promising

- `EXP-005`: bypass contrastivo, desde que a implementacao passe a usar positivos mais estruturados e projection head.
- `EXP-022`: bypass por loss auxiliar ainda parece promissor, mas precisa correcoes de geometria.
- `EXP-026` e `EXP-029`: sao as extensoes mais diretas da intuicao "nao depender do LM head como canal principal de supervisao".

### New Planned Directions

## EXP-026: Learned Target Codec

**Hypothesis:** O bypass do EXP-022 fica mais forte se o alvo latente for aprendido especificamente para supervisao, em vez de herdar a geometria da tabela de embeddings de entrada.
**Type:** Auxiliary loss. Mantem a CE normal e adiciona um alvo latente aprendido por token.
**Mechanism:** Introduz uma pequena tabela ou codec de targets latentes `z_y`. O hidden final e pressionado a aproximar `z_y` por MSE, cosine loss ou variante BYOL-like. O alvo entra com stop-grad e pode ter normalizacao propria.
**Why now:** Continua a linha do EXP-022, mas remove a suposicao forte de que `wte(y)` e a melhor interface geometrica para ensinar o backbone.
**Risk:** Colapso do codec, ou o codec virar apenas uma copia disfarçada do embedding de entrada.

## EXP-027: Adaptive Head Compute

**Hypothesis:** Um unico head uniforme desperdiça compute em tokens faceis e continua insuficiente nos ambiguos. Dar segunda passada apenas onde a entropia e alta melhora custo-beneficio e capacidade efetiva de decodificacao.
**Type:** Structural. Head em duas fases com refinamento seletivo.
**Mechanism:** Uma primeira passada barata produz logits e entropia. Apenas posicoes ambiguas recebem refinamento extra por outro head, MLP residual, ou pequeno conjunto de experts. Em posicoes faceis, usa-se a primeira previsao diretamente.
**Why now:** E a traducao mais direta da ideia de "mais esforco onde ha incerteza" para a saida do LM.
**Risk:** Gate de incerteza mal calibrado, complexidade extra e dificuldade de benchmarking justo.

## EXP-028: Hierarchical Vocab Head

**Hypothesis:** Um softmax plano sobre todo o vocab trata erros grosseiros e erros finos como se fossem o mesmo tipo de decisao. Uma decomposicao coarse-to-fine reduz a violencia da compressao estrutural e organiza melhor o gradiente.
**Type:** Structural. Head hierarquico em duas etapas.
**Mechanism:** Primeiro prediz cluster/classe/coarse code; depois prediz o token dentro do cluster escolhido. Os clusters podem vir de frequencia, embedding clustering ou aprendizagem conjunta.
**Why now:** Com vocab maior, uma saida hierarquica fica mais plausivel do que insistir em um unico softmax plano.
**Risk:** Se os clusters forem ruins, adiciona atrito sem ganho real.

## EXP-029: JEPA-Style Latent Prediction

**Hypothesis:** Uma camada intermediaria pode aprender melhor se for supervisionada para prever um alvo latente interno estavel, sem passar pelo head final.
**Type:** Auxiliary loss. Bypass encoder-first.
**Mechanism:** Um hidden intermediario, como `h_6`, passa por um pequeno preditor `P`. O alvo pode ser `sg(h_12)`, um hidden de teacher EMA, ou um target latente do proximo token. A loss latente roda junto com a CE final.
**Why now:** E o bypass mais alinhado com a leitura atual: supervisionar o backbone em espaco de representacao, nao em espaco de vocabulario.
**Clarification:** Isso nao significa treinar um modelinho separado de traducao token->hidden. Significa usar uma loss auxiliar onde uma parte do proprio modelo aprende a prever uma representacao latente-alvo produzida por outra parte do sistema.
**Risk:** Colapso representacional ou melhora de alinhamento interno sem ganho em perplexity.

**Current read after the first 1000-step runs:**

- `EXP-029.1` e `EXP-029.3` nao se sustentaram como linha principal.
- `EXP-029.2` foi a variante mais interessante da familia.
- `EXP-029.2.2` (`lambda=0.25`) mostrou que existe sinal util no meio do treino, mas perdeu a vantagem tarde.

Isso muda a interpretacao da familia. O problema agora parece menos "a ideia JEPA nao ajuda" e mais "o target atual e o schedule atual ainda nao sao a interface certa para o CE no fim do treino".

**Practical measurement note:** compare `EXP-029.x` against baseline using experiment `ce`, not total `loss`, because the auxiliary latent term is not directly comparable to baseline CE.

**Next corrective variants now implemented:**

- `EXP-029.2.4`: projected next-token latent target
- `EXP-029.2.5`: next-token logit distillation
- `EXP-029.2.6`: future-window latent target

These variants all test the same updated thesis: the future signal may be useful, but the current raw-hidden target is probably too blunt.

## EXP-030: Gated Latent Readout

**Hypothesis:** Somar heads fixas tende a convergir para um operador efetivo quase unico. Um roteamento por token preserva subespacos de leitura realmente distintos.
**Type:** Structural. Readout com roteamento dependente do input.
**Mechanism:** Um gate por token decide a combinacao entre varios leitores latentes especializados. O operador efetivo de saida passa a depender do contexto local, com regularizacao de load balancing para evitar colapso.
**Why now:** E a extensao natural da linha multi-head quando a leitura estatica ja mostrou limite claro.
**Risk:** Colapso do gate e comparacao injusta se o compute medio nao for controlado.

### Note On Revisions

As revisoes e subexperimentos corretivos da agenda atual estao em [experiment_revisions.md](c:/Users/Henrique/studies/nano%20gpt/experiment_revisions.md). A convencao proposta e usar sufixos como `EXP-022.1`, `EXP-022.2`, `EXP-029.1` para variacoes corretivas do mesmo nucleo experimental, em vez de abrir uma familia numerica totalmente nova para cada ajuste.
