# CALM e o problema principal do repo

## Resumo executivo

O paper **Continuous Autoregressive Language Models (CALM)** nao resolve diretamente o problema atual do repo, mas fortalece bastante a direcao estrategica que ja apareceu nos resultados mais informativos: **melhorar apenas a geometria interna do LM head nao parece suficiente; a linha mais promissora e reduzir a dependencia do canal discreto de supervisao via vocabulario**.

A leitura mais util do paper nao e "vamos migrar o projeto inteiro para CALM agora". A leitura util e:

- **Tratar bypass latente como linha principal, nao como loss auxiliar improvisada.**
- **Parar de usar embedding bruto como alvo privilegiado.**
- **Parar de usar hidden futuro bruto como alvo final sem uma projecao ou codec.**
- **Introduzir um alvo latente aprendido e estavel, idealmente com compressao de janela curta de tokens, nao so do proximo token isolado.**
- **Medir se o ganho vem de aliviar o gargalo V -> D ou so de regularizacao extra, usando os probes de perturbacao e freeze que ja estao planejados em experiments.md.**

Em outras palavras: o paper sugere que a tese estrutural do repo esta mais viva na linha **026/029/bypass latente robusto** do que na linha "consertar melhor o mesmo LM head".

## O que o paper adiciona a leitura atual

### 1. Ele reforca que o problema pode ser estrutural, nao apenas de conditioning

Os resultados do repo ja apontam nessa direcao:

- `EXP-016` melhorou `surv` e `eff`, mas piorou loss.
- a familia multi-head ajudou cedo, mas nao sustentou ganho.
- `EXP-029.2` ajudou no meio do treino, mas perdeu a vantagem tarde.

A melhor leitura combinada desses resultados e do paper e:

> melhorar a passagem de gradiente dentro do espaco `D` ajuda, mas nao parece suficiente quando a supervisao principal continua acoplada ao gargalo discreto `V -> D`.

O CALM ataca isso pela raiz: em vez de treinar o modelo principalmente para acertar um token em um grande vocabulario, ele treina para prever uma representacao continua mais densa.

### 2. Ele melhora a interpretacao dos fracassos de bypass atuais

O repo ja chegou perto dessa intuicao, mas ainda com interfaces geometricas fracas:

- `EXP-022` provavelmente falhou mais por **geometria errada do alvo** do que por falha da ideia de bypass.
- `EXP-029.2` provavelmente mostrou sinal real, mas o **target interface** atual e bruto demais e conflita com CE tarde.

O paper sugere uma regra de projeto mais forte:

> se for supervisionar em espaco latente, esse espaco precisa ser robusto, estavel e aprendido para essa funcao, nao herdado diretamente de embedding bruto ou de um hidden futuro cru.

### 3. Ele muda a pergunta certa

A pergunta deixa de ser apenas:

> como fazer mais gradiente sobreviver pelo LM head?

E vira:

> como dar ao backbone um canal principal de aprendizagem que nao dependa tanto do LM head como gargalo discreto?

Essa e uma mudanca importante, porque reorganiza a prioridade experimental.

## O que o paper nao prova

Tambem e importante nao superestimar o paper.

### 1. Nao prova que o LM head explica tudo sozinho

O paper mostra uma rota plausivel e eficiente fora do next-token discreto. Ele nao prova que todo o gap observado no repo e exclusivamente causado pelo LM head.

### 2. Nao implica que vale pivotar para CALM completo agora

O ganho deles depende de um pacote inteiro:

- autoencoder de alta fidelidade
- espaco latente robusto
- objetivo likelihood-free
- head gerativo proprio
- avaliacao e sampling adaptados ao regime continuo

Isso e um novo paradigma de treino, nao uma pequena variante de `train.py`.

### 3. O proprio paper admite que `K=1` piora

Esse ponto e crucial. Sair do CE token-level cria um problema novo e dificil. Portanto, a leitura correta nao e "basta abandonar o discreto".

A leitura correta e:

> a linha latente parece conceitualmente certa, mas a interface latente precisa ser muito melhor desenhada do que as tentativas auxiliares brutas atuais.

## Conclusao estrategica

Se eu tivesse que resumir a implicacao do paper para o repo em uma frase:

> O paper aumenta a confianca de que a melhor linha do projeto e construir um bypass latente robusto e bem interfaceado, nao apenas insistir em variantes do mesmo head discreto.

Isso leva a cinco decisoes praticas:

1. **Tratar bypass latente como linha principal, nao como loss auxiliar improvisada.**
2. **Parar de usar embedding bruto como alvo privilegiado.**
3. **Parar de usar hidden futuro bruto como alvo final sem uma projecao ou codec.**
4. **Introduzir um alvo latente aprendido e estavel, idealmente com compressao de janela curta de tokens, nao so do proximo token isolado.**
5. **Medir se o ganho vem de aliviar o gargalo V -> D ou so de regularizacao extra, usando os probes de perturbacao e freeze que ja estao planejados em experiments.md.**

## Solucoes que eu considero mais fortes agora

Abaixo esta a lista das linhas que eu testaria com maior confianca, em ordem pratica.

### Linha A: codec latente pequeno para janela curta futura

Essa e a linha que mais diretamente traduz a intuicao do CALM para o repo sem abandonar o baseline discreto.

Ideia:

- treinar um codec pequeno que comprime uma janela curta futura de `2` a `4` tokens em um vetor `z`
- congelar esse codec
- prever `z` a partir de um hidden intermediario ou final
- manter a CE normal no `lm_head`
- usar a loss latente como segundo canal de treino para o backbone

Por que tenho confianca:

- ataca diretamente a dependencia exclusiva do target discreto
- evita `embedding` bruto como alvo
- evita `h_{t+1}` cru como alvo
- se alinha ao melhor insight do paper: aumentar a largura semantica do alvo por passo
- preserva comparabilidade com a baseline atual

Risco principal:

- se o codec for mal treinado, ele vira uma interface ruidosa e volta a brigar com a CE

Mitigacao:

- codec pequeno, objetivo simples e congelamento cedo
- loss com `stop-grad` no target
- schedule de lambda decrescente

### Linha B: projected latent target em vez de hidden futuro cru

Essa linha e uma versao mais barata e incremental da anterior.

Ideia:

- manter a intuicao do `EXP-029.2`
- mas projetar o alvo futuro para um subespaco aprendido menor e mais estavel
- supervisionar o preditor nesse subespaco, nao nas coordenadas completas de `h_{12}(t+1)`

Por que tenho confianca:

- e a correcao mais direta para o problema ja visto no `029`
- preserva o sinal util observado early/mid
- reduz a chance de o alvo carregar detalhe irrelevante que conflita com CE tarde

Risco principal:

- a projecao aprendida ainda pode colapsar ou ficar muito alinhada ao ruido do hidden alvo

Mitigacao:

- normalizacao do target
- projector estreito
- stop-grad
- ablar dimensao do target

### Linha C: target latente de janela futura em vez de proximo token isolado

Essa linha e muito promissora porque muda a unidade supervisionada.

Ideia:

- em vez de prever apenas um alvo associado a `t+1`, prever um resumo latente de `t+1:t+k`
- esse resumo pode vir de um codec congelado ou de um agregador projetado

Por que tenho confianca:

- aproxima mais a ideia do CALM sem exigir generacao continua completa
- deve reduzir ruido de um alvo de passo unico
- deve enfatizar estrutura de curto alcance semanticamente mais rica

Risco principal:

- se o resumo ficar informacionalmente fraco, ele vira regularizacao vaga e nao ajuda next-token

Mitigacao:

- manter `k` curto (`2` ou `4`)
- usar dimensao latente moderada
- comparar contra target de um unico token com mesmo budget

## Experimento que eu faria primeiro

**Um EXP-026 ou EXP-029 novo, inspirado diretamente no paper, mas sem abandonar o baseline discreto:**

- **Treinar um codec pequeno que comprime uma janela curta de 2 a 4 tokens futuros em um vetor latente z.**
- **Congelar esse codec.**
- **Fazer um preditor pequeno a partir de um hidden intermediario ou final para prever esse z futuro.**
- **Manter a CE normal no LM head.**
- **Usar loss latente com target projetado e stop-grad, nao hidden cru.**
- **Testar schedule de lambda decrescente, porque seu proprio historico ja sugere ajuda early/mid e conflito late.**

Essa seria minha aposta principal porque ela preserva tudo o que o repo precisa para comparacao causal e, ao mesmo tempo, absorve o melhor insight do CALM sem exigir um pivot completo para likelihood-free LM.

## Variantes de experimento que eu testaria com muita confianca

## EXP-026.1: Future-Window Learned Codec

**Tipo:** bypass latente principal, ainda com CE padrao.

**Mecanismo:**

- treinar um pequeno encoder `E(x_{t+1:t+k}) -> z_t`
- opcionalmente treinar um decoder raso so para validar que `z_t` retém informacao suficiente
- congelar `E`
- usar um predictor `P(h_l(t)) -> z_t`
- loss auxiliar `cosine` ou `mse` em target normalizado com `stop-grad`
- `lambda` com decay ao longo do treino

**Por que e forte:**

- e a translacao mais limpa do insight do paper para o setup atual
- troca alvo token-level cru por alvo semantico curto e estavel
- separa melhor "ajuda estrutural" de "regularizacao arbitraria"

**Ablacoes minimas:**

- `k=2` vs `k=4`
- `l=64` vs `l=128`
- source layer intermediaria vs final
- lambda constante vs decrescente

## EXP-026.2: Projected Codec Target

**Tipo:** variante mais controlada do 026.1.

**Mecanismo:**

- mesmo codec de janela futura
- antes da loss, passar `z_t` por um projector alvo pequeno `T(z_t)` congelado apos pretreino ou treinado com EMA
- prever `T(z_t)` em vez de `z_t` bruto

**Por que e forte:**

- adiciona uma interface mais limpa entre codec e backbone
- reduz risco de o espaco bruto do codec ainda ser mal condicionado para o predictor

## EXP-029.3R: Next-Window Projected Latent Prediction

**Tipo:** sucessor direto da familia 029.

**Mecanismo:**

- manter predictor saindo de `h_6` ou `h_12`
- trocar o target `h_{12}(t+1)` por um target projetado de janela futura
- opcionalmente usar media/projecao de `h_{12}(t+1:t+k)` so para um controle barato antes do codec completo

**Por que e forte:**

- e a menor mudanca com maior chance de corrigir o diagnostico do `029.2`
- testa se o problema era a brutalidade do target, nao a ideia de future latent supervision

## EXP-029.4: Dual-Stage Lambda Schedule

**Tipo:** controle de schedule, nao de arquitetura.

**Mecanismo:**

- qualquer variante latente forte acima
- `lambda` alto no early training
- queda monotona ou em dois estagios no mid/late

**Por que e forte:**

- conversa diretamente com o fato observado de ajuda early/mid e conflito late
- e barato, informativo e deve virar default em qualquer familia de bypass que mostre ganho parcial

## EXP-026.3: Frozen Codec vs Joint Codec

**Tipo:** experimento causal para interface.

**Mecanismo:**

- comparar codec congelado versus codec co-treinado

**Por que e forte:**

- responde se o ganho vem de um alvo estavel ou apenas de mais flexibilidade/mais parametros
- eu apostaria que **codec congelado** vai ser mais limpo no inicio

## O que eu deixaria em segundo plano

### 1. Variantes que apenas melhoram conditioning do head

Elas ainda sao uteis como controle, mas os resultados atuais e o paper juntos diminuem a confianca de que essa seja a linha principal.

### 2. Embedding bruto como target supervisionado

Isso agora parece claramente um baseline corretivo, nao uma aposta principal.

### 3. Hidden futuro bruto como target final

Tambem parece mais um controle diagnostico do que uma interface final promissora.

## Como medir se funcionou de verdade

O erro mais facil agora seria declarar vitoria se uma loss auxiliar melhorar CE cedo. Isso nao basta.

A pergunta certa e:

> o ganho veio de aliviar o gargalo estrutural `V -> D`, ou so de adicionar uma regularizacao extra qualquer?

Para responder isso, eu usaria explicitamente os probes ja planejados em `experiments.md`.

### Medicoes obrigatorias

1. **Perturbacao do LM head por step**
   - aplicar `shuffle` ou `noise` nos steps `25, 50, 100, 150, 200`
   - comparar baseline versus variante latente
   - se a variante latente depender menos do head para recuperar, isso e forte evidencia de alivio real do gargalo

2. **Freeze do LM head por step**
   - congelar `lm_head.weight` nos mesmos steps
   - continuar treino normal
   - se a variante latente sofrer menos que a baseline, ela realmente comprou independência estrutural

3. **Curva de recuperacao pos-choque**
   - medir perda imediata
   - `steps_to_recover`
   - area sob excesso de loss
   - isso e melhor do que olhar apenas loss final

4. **Comparacao matched-step em CE, nao em loss total**
   - qualquer familia com loss auxiliar deve ser comparada por `ce`
   - senao a leitura fica contaminada pelo termo extra

5. **Metricas do canal continuam uteis, mas como secundarias**
   - `surv`
   - `eff`
   - `rr`
   - `top10e`
   - `head_delta`
   - `head_drift0`
   - `conf`, `margin`, `ent_ratio`

### Regra de decisao que eu usaria

Uma variante latente so merece virar linha principal se entregar pelo menos duas das tres coisas abaixo:

- melhora de `ce` sustentada, nao so crossover temporario
- menor sensibilidade a perturbacao/freeze do `lm_head`
- melhor recuperacao apos choque no head

Se ela so melhorar CE cedo, mas continuar igualmente dependente do head, provavelmente estamos vendo regularizacao util, nao resolucao do gargalo.

## Ordem recomendada de implementacao

1. `EXP-029.4` ou equivalente: repetir melhor variante latente com `lambda` decrescente
2. `EXP-029.3R`: projected future-window target sem codec completo
3. `EXP-026.1`: learned codec de janela curta congelado
4. `EXP-026.2`: projected codec target

## Status de implementacao atual

- `EXP-029.2.7` implementado em `experiments/train_exp029_2_7_projected_future_window_warmdown.py`
- `EXP-026.1` implementado em `experiments/train_exp026_1_future_window_codec.py`
- `EXP-026.4` implementado em `experiments/train_exp026_4_token_patch_codec_warmdown.py`

Observacao: o `EXP-026.1` implementado e a versao pratica viavel no pipeline atual, com codec aprendido online e estabilizado por reconstrucao, nao um codec totalmente pretreinado e congelado fora do loop principal. 5. freeze/perturb sweeps nas melhores variantes

Observacao adicional: o `EXP-026.4` novo aproxima melhor a leitura do CALM sem pivot completo. Ele ancora o target em patches reais de tokens futuros, faz um pequeno pretrain do codec dentro do proprio treino, congela o codec depois dessa fase e so entao usa o latente congelado como alvo auxiliar com warmdown.

## Minha aposta atual

Se eu tivesse que escolher uma linha principal hoje, seria esta:

- manter baseline discreto intacto
- adicionar **bypass latente principal** com target de **janela curta futura**
- usar **codec pequeno, congelado, com target projetado e stop-grad**
- usar **lambda decrescente**
- validar com **freeze/perturb probes**

Essa combinacao e, para mim, a leitura mais forte e mais acionavel do que o paper sugere para o problema do repo.
