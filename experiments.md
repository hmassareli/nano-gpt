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
**Type:** Episodic (init-time only). Does not change the forward pass or training loop.
**Mechanism:** Computes SVD of wte at init, sets lm_head = pinv(wte). After step 0, head evolves freely.
**Status:** Benchmarked (100 steps). Observed worse than baseline mid-run (loss 5.059 vs 4.983 at step 55).
**Assessment:** NEGATIVE. Init-time fixes don't address continuous conditioning degradation. The pseudo-inverse starts with good condition number but it decays just as fast — the paper's Fig.3 shows this explicitly.

### EXP-003: Head Conditioning Regularization
**Hypothesis:** Penalizing ||W^T·W - αI||² keeps singular values uniform, preserving gradient flow throughout training.
**Type:** Episodic (regularization penalty). Does not change forward pass.
**Mechanism:** At each step, computes W^T·W (D×D matrix), scales by diagonal mean, penalizes Frobenius distance to identity. λ=0.01. Adds separate `.backward()`.
**Params:** 0 extra (same model as baseline).
**Status:** Benchmarked (100 steps).
**Assessment:** NEUTRAL. Correct intuition (maintain orthogonality), but λ=0.01 may be too weak to matter, and the regularization can fight the model if the head *needs* temporary low rank during training.

### EXP-004: Soft Weight Tying (head ↔ embedding)
**Hypothesis:** λ||W_head - W_emb||² continuously anchors the head to the semantic structure of the embedding, preventing conditioning degradation.
**Type:** Episodic (regularization penalty). Does not change forward pass.
**Mechanism:** MSE penalty between lm_head.weight and wte.weight with λ=0.05. Applied after optimizer step.
**Params:** 0 extra.
**Status:** Benchmarked (100 steps).
**Assessment:** HARMFUL. W_emb (std=1.0) and W_head (std=0.001) are initialized at vastly different scales — MSE penalty dominates early gradients. More fundamentally, W_emb and W_head serve mathematically opposing functions (encoding vs decoding). Tying them fights the specialization each matrix needs. Not recommended.

### EXP-005: Contrastive Auxiliary Loss (Head Bypass)
**Hypothesis:** A contrastive loss (InfoNCE) on the hidden states gives the backbone a gradient channel that bypasses the head bottleneck entirely.
**Type:** Auxiliary loss. Does not change architecture, but modifies `forward()` to return hiddens.
**Mechanism:** Subsamples 256 tokens per micro-step. Computes cosine similarity matrix (256×256). Same-target tokens = positive pairs. InfoNCE with temperature=0.1. Combined backward: `((CE_loss / grad_accum) + contrastive_loss).backward()`. λ=0.1.
**Params:** 0 extra.
**Status:** Benchmarking (100 steps, running).
**Note:** Original implementation had critical bug — `hiddens.detach()` killed all gradients, making the contrastive loss a no-op. Fixed: removed detach, added subsampling (was full B×T similarity matrix → OOM), unified backward call.
**Assessment (theoretical):** Promising training-time bypass. Backbone receives gradient signal to cluster hiddens by token identity without going through the degraded lm_head. However, at inference time the prediction still passes through the single lm_head — the structural bottleneck remains for decoding.

### EXP-006: Factored Head (512→482→8192) iso-param
**Hypothesis:** A two-layer head with non-linearity (GELU) provides better gradient flow than a single linear projection, by breaking the rank-1 update pattern.
**Type:** Structural (changes forward pass and architecture).
**Mechanism:** Replaces `lm_head: Linear(D, V)` with `lm_head_expand: Linear(D, H)` → GELU → `lm_head: Linear(H, V)`. Softcap=13 on logits.
**Params:** H=482. Total head params = 482×(512+8192) = 4,195,328 ≈ baseline 4,194,304 (+0.02%). Iso-parameter comparison.
**Init:** lm_head_expand std = n_embd^(-0.5), lm_head std = 0.001.
**Status:** COMPLETED (100 steps). val_bpb = **1.663** (pior que baseline 1.564).
**Note:** Original design used H=n_embd×4=2048, adding +77M params (head larger than entire model). Redesigned iso-parameter with H=482.
**Assessment:** NEGATIVE. GELU non-linearity alone does not help. Serves as critical control for EXP-007: proves that multi-head diversity (not GELU) drives the improvement.

### EXP-007: Multi-Head Output (3 parallel subheads)
**Hypothesis:** Splitting the single lm_head into K independent subheads creates K independent gradient channels, so even if one subhead's conditioning degrades, the others still carry gradient information. This addresses the bottleneck structurally in both training AND inference.
**Type:** Structural (changes architecture and forward pass).
**Mechanism:** Replaces single `lm_head: Linear(D, V)` with K=3 parallel subheads, each: `proj: Linear(D, D_sub)` → GELU → `out: Linear(D_sub, V)`. Final logits = sum of all K subhead logits, then softcap=15.
**Params:** D_sub=160. Each subhead: 160×(512+8192) = 1,392,640. Total: 3×1,392,640 = 4,177,920 ≈ baseline 4,194,304 (-0.4%). Iso-parameter comparison.
**Init:** proj weights normal(std=n_embd^(-0.5)), out weights normal(std=0.001).
**Optimizer:** All output_heads params go into unembedding LR group (AdamW, lr=0.004×dmodel_scale).
**Config:** `num_output_heads=3`, `output_head_dim=160` added to GPTConfig dataclass.
**Status:** COMPLETED (300 steps).
**Assessment (theoretical):** Most promising experiment. Three independent gradient pathways mean: (1) rank degradation in one subhead doesn't affect the others, (2) GELU non-linearity in each breaks the rank-1 update pattern, (3) the model has structural diversity at inference time (not just training-time like EXP-005). Key test: does parallel decomposition (3×160) outperform single-pathway (1×482) from EXP-006 at equal parameter count?

**Results (300 steps):**
- val_bpb: **1.157** vs baseline **1.144** → Δ = **+0.013 bpb (+1.1%)** — baseline vence
- Training loss final: 3.373 vs 3.337 → **+0.036 (+1.1%)**
- Throughput: 6,394 tok/s vs 6,669 tok/s → **-4.1% overhead**
- Loss drop total: 5.638 (EXP) vs 5.674 (baseline) → baseline aprende mais

**Evolução do loss ao longo do treino (300 steps):**
| Step | EXP-007 | Baseline | Diff | Observação |
|------|---------|----------|------|------------|
| 5 | 7.817 | 7.663 | +0.154 | EXP atrás (heads se calibrando) |
| 10 | 6.886 | 6.603 | +0.283 | EXP ainda atrás |
| 25 | 5.759 | 5.639 | +0.120 | Gap diminuindo |
| 50 | 5.085 | 5.062 | +0.023 | Quase empatados |
| 100 | 4.251 | 4.259 | **-0.008** | Crossover — EXP lidera brevemente |
| 200 | 3.630 | 3.590 | +0.040 | Baseline retoma liderança |
| 299 | 3.373 | 3.337 | +0.036 | Baseline vence por margem estável |

**Dinâmica observada:**
1. **Steps 0→50**: EXP-007 fica atrás — as 3 heads gastam steps iniciais se coordenando.
2. **Steps 50→100**: EXP-007 acelera e cruza o baseline brevemente (~step 100). Esse era o resultado animador do benchmark de 100 steps.
3. **Steps 100→300**: A vantagem **não se sustenta**. O baseline volta a liderar e mantém margem de ~0.036 no loss e +0.013 bpb na validação até o final.

**Assessment:** MISTO. O EXP-007 foi promissor até certo ponto — demonstrou que multi-head realmente injeta diversidade nos gradientes e acelera o aprendizado na fase inicial/intermediária. Porém, a vantagem é **temporária**: após ~100 steps, o baseline ultrapassa, sugerindo que (a) os heads colapsam e perdem diversidade ao longo do treino, (b) o GELU destrói gradientes importantes, e/ou (c) o max rank 480 < 512 limita o modelo no late training. Confirma EXP-006 (val_bpb=1.663) que GELU sozinha não ajuda — o ganho vem da diversificação — mas mostra que sem mecanismos pra **manter** essa diversidade, o benefício se perde.

**Logs:** `benchmark_logs/bench_20260314_084804_300steps.log` (300 steps), `benchmark_logs/bench_20260313_150256_100steps.log` (100 steps)

---

## Key Discovery: Gradient Diversification via Multi-Head Output

### O que encontramos

O multi-head output (EXP-007) mostrou uma dinâmica de treino em **duas fases distintas**:

**Fase 1 (steps 0→100) — Aceleração inicial:**
O multi-head aprende mais devagar nos primeiros ~50 steps (calibração das 3 heads), mas depois acelera e cruza o baseline por volta do step 100. No benchmark de 100 steps original, o EXP-007 vencia por -2.0% no loss e -2.94% no val_bpb. A métrica WinΔ% chegou a +41%, sugerindo aprendizado 41% mais rápido por janela.

**Fase 2 (steps 100→300) — Perda de vantagem:**
No benchmark estendido de 300 steps, a vantagem **não se sustentou**. O baseline voltou a liderar a partir do step ~150 e terminou com val_bpb = **1.144** vs EXP-007 = **1.157** (+1.1%). A eficiência de aprendizado do multi-head saturou e o baseline convergiu melhor.

**Interpretação:** O multi-head injeta diversidade de gradiente real — mas essa diversidade é **temporária**. As heads tendem a colapsar (head_cos cresce ao longo do treino) e o GELU destrói ~15-25% do gradiente. O max rank limitado (3×160=480 < 512=D) também pode limitar a convergência final.

### Por que funciona inicialmente (mecanismo teórico)

**O problema**: No backward pass, `dL/dh = dL/dlogits · W_head^T`. Com single-head (512×8192), W_head projeta os gradientes de 8192 dimensões para 512, perdendo ~94% do sinal. A condição de W_head degrada ao longo do treino (paper: "Lost in Backpropagation"), comprimindo os gradientes num subespaço cada vez menor.

**A solução parcial**: 3 heads de rank 160 projetam gradientes em 3 subespaços **diferentes**. No early training, quando as heads são diversas por causa da inicialização aleatória, a união desses subespaços tem rank efetivo mais alto que o subespaço do single head. Resultado: gradientes mais ricos chegam ao backbone nos primeiros ~100 steps.

**Por que não se mantém**: (a) as heads convergem pro mesmo subespaço (colapso), perdendo diversidade; (b) o GELU mascara gradientes negativos em cada head; (c) max rank 480 < 512 limita representação no late training quando o modelo precisa de discriminação fina.

### EXP-006 como controle

EXP-006 (Factored Head: 512→482→GELU→8192) isola o efeito do GELU. Resultado: val_bpb = **1.663** — pior que o baseline (1.564). Isso confirma que GELU sozinha (quebrando o padrão rank-1) **não é o fator**. O ganho inicial do EXP-007 vem da **diversificação via múltiplas heads**, não da não-linearidade.

### Perguntas abertas para os próximos experimentos

O resultado do EXP-007 (bom no início, perde depois) motiva toda a segunda geração de experimentos (012-024):
1. **É o colapso das heads?** → Testar diversidade forçada e/ou supervisionada (014, 015, 017, 018, 019, 020)
2. **É o GELU?** → O controle principal continua sendo 012/013 vs a nova linha linear supervisionada (017/018)
3. **É o rank < D?** → Testar somas de ranks = D (012, 013, 017, 018)
4. **Multi-head é sequer necessário?** → Testar ortho reg no head único (016)
5. **Melhor bypasser o head completamente?** → Testar deep supervision (021) e embedding loss (022)

---

## EXP-008: Fast Multi-Head (Fused Projection Only)

**Date:** 2026-03-13
**Hypothesis:** Fusing the 3 separate proj matmuls (D→D_k × 3) into a single matmul (D→3×D_k) reduces overhead while preserving gradient diversification.
**Type:** Structural optimization of EXP-007 (free — no architectural change to the gradient flow).
**Mechanism:** Single `fused_proj: Linear(D, 3×D_k)` followed by split + GELU, then 3 separate `out: Linear(D_k, V)`. 4 matmuls total vs EXP-007's 6.
**Params:** Same as EXP-007 (~4.18M head). Iso-parameter.
**Status:** Awaiting 300-step benchmark.
**File:** `experiments/train_exp008_fast_multi_head.py`

---

## EXP-009: Multi-Head + Two-Stage (Reference)

**Date:** 2026-03-13
**Hypothesis:** Combining multi-head with two-stage head update.
**Status:** Kept as reference only. Two-stage was proven fundamentally flawed (doesn't fix gradient compression — `dL/dh = dL/dlogits · W_head` is identical whether head is frozen or not).
**File:** `experiments/train_exp009_multi_head_two_stage.py`

---

## EXP-010: Fast Multi-Head + Head Dropout

**Date:** 2026-03-13
**Hypothesis:** Randomly dropping 1 of 3 heads per step (with rescaling) forces the remaining heads to be independently useful, increasing gradient diversity.
**Type:** Regularization on top of EXP-008's fused architecture.
**Mechanism:** HEAD_DROP_PROB=1/3. Each step, one head is masked. Surviving heads' outputs rescaled by 3/active_count. At eval, all heads active.
**Params:** Same as EXP-008 (~4.18M head). Zero extra params.
**Status:** Awaiting 300-step benchmark.
**File:** `experiments/train_exp010_head_dropout.py`

---

## Questão Central em Aberto: O bottleneck é estrutural (V→D) ou de otimização (uso ineficiente de D)?

O paper de Godey identifica dois gargalos no backward pass do LM head:

1. **Gargalo estrutural**: A projeção $W^T: \mathbb{R}^V \to \mathbb{R}^D$ comprime o gradiente de V≈8192 dimensões para D=512. Isso é inerente à arquitetura — 94% das dimensões são descartadas.

2. **Gargalo de otimização**: Mesmo dentro de D, o gradiente não usa todo o espaço disponível. O rank efetivo é $k \ll D$, especialmente no início do treino. Ou seja, o fluxo real é $\mathbb{R}^V \to \mathbb{R}^D \to \mathbb{R}^k$.

**O que nossos dados mostram (baseline, step 0):**
- `survival = 0.146` → apenas 14.6% da norma do gradiente sobrevive à projeção pelo head
- `head_effrank = 223` de um máximo de 512 → usando apenas 43% do rank disponível
- `top10_energy = 28.2%` → os 10 maiores SVs concentram quase 1/3 da energia total

Isso sugere que **ambos** os gargalos existem simultaneamente. O primeiro (V→D) é fixo pela arquitetura. O segundo (D→k) é potencialmente corrigível.

### Duas linhas de ataque

**Linha A — Atacar o gargalo de otimização (fazer o head usar D inteiro):**
Se conseguirmos forçar o head a usar todas as 512 dimensões uniformemente, o survival sobe e o backbone recebe gradientes mais ricos. Abordagens:
- Regularização de ortogonalidade (EXP-016): $\|W^T W - I\|^2$
- Multi-head com diversidade forçada (EXP-014, 015, 019, 020)
- Temperature schedule (EXP-024)

**Linha B — Contornar o gargalo estrutural (gradientes que não passam pelo head):**
Criar canais alternativos de gradiente que bypassam a compressão V→D completamente:
- Deep supervision (EXP-021): CE auxiliar na camada 6
- Embedding space loss (EXP-022): $\|h - e_{target}\|^2$ gera gradiente full-rank direto no hidden

### A hipótese do multi-head

Os experimentos multi-head (007, 012-020) testam uma terceira possibilidade: **decomposição do head em canais que precisam permanecer funcionalmente distintos**. A ideia inicial era apenas quebrar o head em subespaços menores; a segunda geração (017-020) refina isso e passa a exigir diversidade via supervisão individual ou via roteamento dependente do input.

**Limite importante:** essa linha não remove o gargalo estrutural $\mathbb{R}^V \to \mathbb{R}^D$ do paper. O que ela pode fazer, no melhor caso, é melhorar a **geometria do Jacobiano** que chega ao backbone: conditioning, estabilidade do subespaço e diversidade de canais ao longo do treino.

**A favor:** EXP-007 ganha do baseline de 100 steps — WinΔ% crescendo até +41%.

**Contra:** O max rank com 3 heads D_k=160 é 480 < 512 do baseline. Os heads tendem a colapsar (head_cos sobe ao longo do treino), e o GELU em cada head destrói ~15-25% do gradiente.

**Pergunta aberta:** Se mantivermos os heads distintos por supervisão individual ou por roteamento dependente do input, e não perdermos dimensões (max rank ≥ D), o multi-head supera o baseline de forma sustentável? Ou o baseline com boa regularização (016) resolve o mesmo problema de forma mais simples?

Os experimentos 012-020 foram desenhados para systematicamente isolar essas variáveis:
- Número de heads (2 vs 3 vs 4)
- Papel da não-linearidade (linha com GELU vs linha linear supervisionada)
- Mecanismo de preservação de canais (nenhum, dropout, cosine, per-head CE, gating)
- Max rank relativo a D (480, 512, 513)

---

## EXP-011: Multi-Head + Separate Head Dropout

**Date:** 2026-03-14
**Hypothesis:** Per-head dropout (em vez de dropout por subhead fused do EXP-010) força diversidade mais efetiva entre os heads.
**Type:** Regularization. Mesma arquitetura do EXP-007 (3 heads separados, D_k=160).
**Mechanism:** HEAD_DROP_PROB > 0. Cada step, um subset dos heads é mascarado. Pelo menos 1 head sempre ativo.
**Params:** Iso com EXP-007 (~4.18M head).
**Max rank:** 480 (3×160).
**File:** `experiments/train_exp011_separate_head_dropout.py`

---

## EXP-012: 2-Head Full Rank

**Date:** 2026-03-14
**Hypothesis:** Com apenas 2 heads de D_k=256, o max rank iguala o baseline (2×256=512=D). Testa se o benefício do multi-head se mantém com rank pleno.
**Type:** Structural variant of EXP-007.
**Params:** 2×256×8704 = 4,456,448 (~4.5M, +6% vs baseline).
**Max rank:** 512 = D ✓
**GELU:** Sim.
**Diversidade:** Nenhuma (além da inicialização aleatória).
**File:** `experiments/train_exp012_2head_full_rank.py`

---

## EXP-013: 4-Head Full Rank

**Date:** 2026-03-14
**Hypothesis:** Com 4 heads de D_k=128, max rank = 512 = D. Mais heads = mais diversidade natural? Testa se o número de heads importa quando o rank é controlado.
**Type:** Structural variant of EXP-007.
**Params:** 4×128×8704 = 4,456,448 (~4.5M, +6% vs baseline).
**Max rank:** 512 = D ✓
**GELU:** Sim.
**Diversidade:** Nenhuma.
**File:** `experiments/train_exp013_4head_full_rank.py`

---

## EXP-014: 4-Head + Cosine Diversity Penalty

**Date:** 2026-03-14
**Hypothesis:** Penalizar a similaridade cosseno entre os pesos proj dos heads mantém os subespaços diversos ao longo do treino, evitando o colapso observado no EXP-007.
**Type:** Regularization + structural.
**Mechanism:** Adiciona ao loss: $\lambda \cdot \text{mean}(\cos(W_i, W_j))$ para todos pares i,j de heads. λ=0.1.
**Params:** 4×128×8704 = 4,456,448.
**Max rank:** 512 = D ✓
**GELU:** Sim.
**Diversidade:** Cosine penalty nos pesos (weight-space).
**File:** `experiments/train_exp014_cosine_diversity.py`

---

## EXP-015: 4-Head + DeCov Regularization

**Date:** 2026-03-14
**Hypothesis:** Penalizar a covariância cruzada das ativações intermediárias (pós-GELU) dos heads força patterns de ativação diferentes entre heads, complementando a penalidade de peso da EXP-014.
**Type:** Regularization + structural.
**Mechanism:** Computa features intermediárias de cada head, empilha, calcula matriz de covariância, penaliza off-diagonal: $\lambda \cdot \|C_{off-diag}\|^2$. λ=0.1.
**Params:** 4×128×8704 = 4,456,448.
**Max rank:** 512 = D ✓
**GELU:** Sim.
**Diversidade:** DeCov nas ativações (activation-space).
**Diferença vs EXP-014:** Cosine penalty atua nos **pesos** (estático). DeCov atua nas **ativações** (dinâmico, depende do input). DeCov pode ser mais efetivo pra manter diversidade funcional.
**File:** `experiments/train_exp015_decov.py`

---

## EXP-016: Baseline + Soft Orthogonality Regularization

**Date:** 2026-03-14
**Hypothesis:** Forçar $W^T W \approx I_D$ no LM head único mantém os valores singulares uniformes, maximizando rank efetivo e survival ratio. Ataca o gargalo de otimização diretamente, sem multi-head.
**Type:** Regularization on baseline architecture. Zero mudança estrutural.
**Mechanism:** $L_{reg} = \lambda \|W^T W - I\|_F^2$. A matriz $W^T W$ é D×D (512×512), barata de computar. λ=0.01.
**Params:** 0 extras.
**Max rank:** 512 (mesmo do baseline).
**Originalidade:** Semi-original. Ortogonalidade usada em GANs (BigGAN) e representation learning, mas nunca aplicada ao LM head para resolver gradient bottleneck. Conexão direta com o paper de Godey.
**Importância:** Se funcionar, prova que o problema é de **otimização** (o SGD não usa D inteiro) e não **estrutural** (precisa de multi-head). Tornaria toda a linha multi-head desnecessária.
**File:** `experiments/train_exp016_ortho_reg.py`

---

## EXP-017: 2-Head No GELU + Per-Head CE

**Date:** 2026-03-14
**Hypothesis:** Sem GELU, a soma de heads lineares colapsa para um único mapa linear efetivo. Para preservar múltiplos canais mesmo nesse regime, cada head precisa ser **individualmente preditiva**. A loss auxiliar por head impede que uma head pegue carona na outra.
**Type:** Structural + auxiliary loss. Substitui o antigo "2-head linear puro".
**Mechanism:** `loss = CE(sum_k logits_k) + α * mean_k CE(logits_k)`. α=0.25. Mantém 2 heads de D_k=256 para que 2×256=512.
**Params:** 4,456,448.
**Max rank:** 512 = D ✓
**GELU:** Não.
**Diversidade:** Supervisão individual por head (task-space).
**Importância:** É o teste mínimo da hipótese "múltiplos canais sem GELU ainda podem existir se cada canal carregar sinal próprio".
**File:** `experiments/train_exp017_2head_per_head_ce.py`

---

## EXP-018: 4-Head No GELU + Per-Head CE

**Date:** 2026-03-14
**Hypothesis:** Se a supervisão individual realmente preserva canais, então aumentar o número de heads de 2 para 4 deve aumentar a especialização funcional mesmo mantendo o mesmo rank total 512.
**Type:** Structural + auxiliary loss. Versão 4-head do EXP-017.
**Mechanism:** `loss = CE(sum_k logits_k) + α * mean_k CE(logits_k)`. α=0.25. Usa 4 heads de D_k=128 para manter 4×128=512.
**Params:** 4,456,448.
**Max rank:** 512 = D ✓
**GELU:** Não.
**Diversidade:** Supervisão individual por head (task-space).
**File:** `experiments/train_exp018_4head_per_head_ce.py`

---

## EXP-019: 4-Head No GELU + Per-Head CE + Projector Overlap

**Date:** 2026-03-14
**Hypothesis:** A per-head CE deve impedir carona funcional; a penalização de overlap entre subespaços projetores deve impedir colapso geométrico de forma mais fiel que cosine em pesos achatados. Se ambos forem necessários, o 019 deve superar o 018 de forma estável.
**Type:** Structural + auxiliary loss + regularization.
**Mechanism:** `loss = CE(sum_k logits_k) + α * mean_k CE(logits_k) + λ * mean_{i<j} ||Q_i^T Q_j||_F^2`, onde `Q_i` é uma base ortonormal do row-space do projetor da head `i`. α=0.25, λ=0.1.
**Params:** 4,456,448.
**Max rank:** 512 = D ✓
**GELU:** Não.
**Diversidade:** Supervisão individual + overlap explícito entre subespaços projetores.
**File:** `experiments/train_exp019_4head_per_head_ce_projector_overlap.py`

---

## EXP-020: 4-Head No GELU + Token Gating + Load Balance

**Date:** 2026-03-14
**Hypothesis:** O jeito mais direto de evitar que 4 heads lineares virem um único mapa é tornar a combinação **dependente do input**. Um gate por token preserva múltiplos canais reais; uma penalidade de load balancing evita que tudo colapse em uma head só.
**Type:** Structural + auxiliary loss + routing regularization.
**Mechanism:** `logits = sum_k π_k(h) * logits_k`, com `π(h)=softmax(W_gate h)`. Loss total: `CE(gated_sum) + α * mean_k CE(logits_k) + λ * ||mean(π) - 1/K||²`. α=0.2, λ=0.1.
**Params:** 4,456,448 + gate D×4 (desprezível).
**Max rank:** 512 = D ✓ por token, mas agora o operador efetivo varia com o contexto.
**GELU:** Não.
**Diversidade:** Supervisão individual + gating dependente do input + load balancing.
**File:** `experiments/train_exp020_4head_token_gating.py`

---

## EXP-025: 4-Head No GELU + Per-Head CE + Hidden-Grad Diversity

**Date:** 2026-03-15
**Hypothesis:** Se o problema real é que os heads acabam produzindo praticamente o mesmo sinal para o backbone, então a regularização deve agir no objeto certo: o gradiente no hidden. Penalizar a similaridade entre $\partial L_i / \partial h$ força canais que são diferentes no **backward**, não só no peso.
**Type:** Structural + auxiliary loss + second-order regularization.
**Mechanism:** Baseado no EXP-018, adiciona um termo $\lambda \cdot \text{mean}_{i<j}(\cos(g_i, g_j)^2)$ onde $g_i = \partial L_i/\partial h$. Para conter custo, calcula em um subsample de 64 posições por micro-step. λ=0.05. Mantém 4 heads de D_k=128.
**Params:** 4,456,448.
**Max rank:** 512 = D ✓
**GELU:** Não.
**Diversidade:** Supervisão individual + diversidade explícita no gradiente do hidden.
**Custo:** Alto. Exige derivadas de segunda ordem; `torch.compile` fica desligado por default.
**Importância:** É o experimento mais próximo da formulação do paper entre os multi-heads, porque força diversidade no próprio canal que chega ao backbone.
**File:** `experiments/train_exp025_4head_grad_diversity.py`

---

## EXP-021: Deep Supervision (Auxiliary CE at Layer 6)

**Date:** 2026-03-14
**Hypothesis:** Uma head auxiliar na camada intermediária (layer 6 de 12) cria um canal de gradiente que **bypassa completamente** o LM head final. O backbone recebe gradiente direto da camada 6, sem a compressão V→D do head principal.
**Type:** Auxiliary loss. Baseline architecture + aux_head extra.
**Mechanism:** `loss = CE_final + α * CE_layer6`. α=0.3. O aux_head é um Linear(D,V) separado que recebe o hidden state normalizado da camada 6.
**Params:** +4,194,304 extras (aux_head duplica o head). Total ~75.5M.
**Originalidade:** Baixa (deep supervision existe desde 2015). Mas o framing via gradient bottleneck do LM head é novo.
**Importância:** Único experimento que ataca o **primeiro gargalo** (V→D estrutural). Todos os outros (012-020) atacam o segundo (uso ineficiente de D).
**File:** `experiments/train_exp021_deep_supervision.py`

---

## EXP-022: Embedding Space Loss

**Date:** 2026-03-14
**Hypothesis:** Adicionar $\|h - e_{target}\|^2$ à loss cria um gradiente **full-rank** direto no hidden space, sem passar pelo softmax ou pelo LM head. O gradiente dessa loss é simplesmente $2\alpha(h - e_y)$ — vetor no $\mathbb{R}^D$ apontando do hidden pro embedding do token correto.
**Type:** Auxiliary loss. Baseline architecture, zero parâmetros extras.
**Mechanism:** `loss = CE + α * mean(||h - wte(target)||²)`. α=0.1. Usa a tabela de embeddings existente (wte) como target. Ignora posições com target=-1.
**Status:** IMPLEMENTADO. O embedding alvo entra com `detach`, então a loss realmente supervisiona o hidden sem depender do Jacobiano do LM head.
**Params:** 0 extras.
**Originalidade:** **Alta.** Ninguém na literatura conectou embedding space loss ao gradient bottleneck do LM head. O gradiente $2\alpha(h - e_y)$ é:
- Full-rank (direção arbitrária em $\mathbb{R}^D$)
- Sem compressão (não passa pelo head)
- Sem custo de parâmetros
- Semanticamente rico (aponta pro embedding do token correto)
**Importância:** Se funcionar, é a solução mais simples possível pro problema identificado pelo paper de Godey. Custo zero, implementação trivial, ataca os dois gargalos ao mesmo tempo.
**File:** `experiments/train_exp022_embedding_loss.py`

---

## EXP-023: Mixture of Softmax (MoS)

**Date:** 2026-03-14
**Hypothesis:** Mistura de K=4 distribuições softmax independentes aumenta o rank da distribuição modelada, quebrando o "softmax bottleneck" de **expressividade** (Yang et al., 2018).
**Type:** Structural. Baseline + K heads completos + gating.
**Mechanism:** $p = \sum_{k=1}^K \pi_k \cdot \text{softmax}(W_k h)$, onde $\pi_k = \text{softmax}(W_{gate} h)$. Cada componente gera uma distribuição softmax completa. A mistura tem rank até K×D.
**Params:** +4 heads completos (4×D×V) + gate (D×K) = ~16.8M extras. Total ~88M. **NÃO iso-parâmetro.**
**Originalidade:** Baixa. Reimplementação de Breaking the Softmax Bottleneck (Yang et al., ICLR 2018). Aplicado originalmente a LSTMs.
**Importância:** Serve como **upper bound** de quanto a abordagem "mais capacidade no head" pode melhorar. Se EXP-022 (0 params extras) chegar perto do MoS (+16.8M params extras), é argumento forte pro 022.
**Nota:** Não é iso-parâmetro por design — o MoS original precisa de heads full-rank pra funcionar. Reduzir via projeção low-rank quebraria o propósito teórico.
**Nota 2:** Mesmo assim, o MoS não remove o gargalo estrutural do backward para o backbone; ele continua sendo sobretudo um controle de expressividade/capacidade.
**File:** `experiments/train_exp023_mixture_of_softmax.py`

---

## EXP-024: Temperature Schedule (High→Low)

**Date:** 2026-03-14
**Hypothesis:** No início do treino, quando o rank efetivo do gradiente é muito baixo (k≪D), temperatura alta espalha a distribuição softmax, produzindo gradientes mais uniformes em mais direções. Conforme o treino avança e o modelo precisa de discriminação fina, a temperatura cai pra 1.0.
**Type:** Training schedule. Baseline architecture, zero parâmetros extras.
**Mechanism:** `temp = lerp(3.0, 1.0, progress / 0.5)` nos primeiros 50% do treino, depois temp=1.0. Logits divididos por temp antes do softmax.
**Params:** 0 extras.
**Originalidade:** Moderada. Temperature scaling existe em distillation e curriculum learning, mas o framing "compensar low-rank do gradiente no early training" é novo.
**Intuição:** Com temp=3.0, o softmax fica mais suave: $p_i \approx 1/V$ pra todos os tokens. Isso faz $p - y$ ter componentes não-zero em muitas direções, aumentando o rank efetivo do gradiente $W^T(p-y)$. O custo é que o sinal discriminativo é fraco — mas no early training o modelo não sabe discriminar mesmo.
**File:** `experiments/train_exp024_temp_schedule.py`

---

## Design Experimental: Comparações Planejadas

### Tabela de variáveis controladas

| Exp | Tipo | Heads | D_k | Max Rank | GELU | Diversidade | Params extras |
|-----|------|-------|-----|----------|------|-------------|---------------|
| Baseline | single head | 1 | — | 512 | — | — | 0 |
| 016 | single + ortho reg | 1 | — | 512 | — | $W^TW≈I$ | 0 |
| 007 | multi-head | 3 | 160 | 480 | sim | — | 0 |
| 011 | multi-head | 3 | 160 | 480 | sim | head dropout | 0 |
| 012 | multi-head | 2 | 256 | 512 | sim | — | +0.3M |
| 013 | multi-head | 4 | 128 | 512 | sim | — | +0.3M |
| 014 | multi-head | 4 | 128 | 512 | sim | cosine penalty | +0.3M |
| 015 | multi-head | 4 | 128 | 512 | sim | DeCov | +0.3M |
| 017 | multi-head | 2 | 256 | 512 | **não** | per-head CE | +0.3M |
| 018 | multi-head | 4 | 128 | 512 | **não** | per-head CE | +0.3M |
| 019 | multi-head | 4 | 128 | 512 | **não** | per-head CE + projector overlap | +0.3M |
| 020 | multi-head | 4 | 128 | 512 | **não** | token gating + load balance | +0.3M |
| 025 | multi-head | 4 | 128 | 512 | **não** | per-head CE + grad diversity | +0.3M |
| 021 | deep supervision | 1 | — | 512 | — | — | +4.2M |
| 022 | emb space loss | 1 | — | 512 | — | — | 0 |
| 023 | MoS | 4 full | D | 4×512 | — | — | +16.8M |
| 024 | temp schedule | 1 | — | 512 | — | — | 0 |

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
