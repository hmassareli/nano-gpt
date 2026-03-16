O Problema: O LM Head Esmaga o Gradiente
O que acontece no forward (pra frente)
O modelo lê tokens e vai construindo uma "representação interna" — um vetor com 512 números que resume tudo que ele entendeu até ali. Pensa nisso como um resumo denso do texto.

No final, o LM head precisa transformar esse resumo de 512 números em uma votação entre 8192 tokens possíveis ("a próxima palavra é mais provavelmente 'gato' ou 'casa' ou...?"). Ele faz isso multiplicando por uma tabelona (uma matriz) que tem uma linha pra cada token do vocabulário.

Resumindo: 512 números → 8192 votos. Expansão. Sem problema.

O que acontece no backward (volta do gradiente)
Quando o modelo erra a previsão, calculamos o erro (loss). Aí precisamos propagar esse erro de volta pro modelo inteiro, pra que cada camada aprenda a se corrigir. Esse sinal de volta é o gradiente.

O gradiente começa rico: "você deveria ter dado mais voto pro token 3847 e menos pro 291, e o 5000 tava ok...". São 8192 instruções de correção, uma por token do vocabulário.

Mas pra voltar pro backbone (as 12 camadas de transformer), esse gradiente precisa passar pelo LM head ao contrário. E aí acontece o oposto do forward: 8192 instruções → 512 números. Compressão.

É como se você tivesse 8192 críticas detalhadas sobre um texto, mas só pudesse passar um bilhete de 512 palavras pro autor. Muita informação se perde.

Quanto se perde?
Imagina que o gradiente com 8192 instruções é uma casa com 8192 cômodos. Mas o corredor de volta pro backbone só tem 512 portas. O máximo de cômodos que conseguem passar é 512. Os outros 7680 ficam trancados — pra sempre.

Ou seja: só 512 de 8192 "direções de correção" sobrevivem. As outras ~7680 são descartadas. Isso é ~94% do sinal de aprendizado jogado fora.

O paper mediu na prática e encontrou 95–99% de perda.

Por que a situação pode ser ainda pior
Mesmo das 512 portas que existem, nem todas passam a mesma quantidade de informação. A tabelona do head pode ter portas favoritas — algumas deixam passar muito gradiente, outras quase nada.

Tecnicamente, isso se mede pelos valores singulares da matriz — pensa neles como a "largura" de cada porta. Se 10 portas são enormes e as outras 502 são estreitinhas, na prática só ~10 direções realmente funcionam. Isso é o rank efetivo (quantas portas realmente importam).

O cenário ideal: todas as 512 portas com a mesma largura. O cenário real: poucas portas dominam.

Como cada experimento ataca isso
EXP-002: Spectral Init
O quê: No início do treino, reorganiza as 512 portas pra que todas tenham a mesma largura.

Como: Usa a tabela de embeddings (que traduz tokens → vetores) como referência. Constrói o head como o "inverso perfeito" dessa tabela, garantindo portas uniformes.

Limitação: É só uma configuração inicial. O treino pode bagunçar as portas de volta.

Custo: Zero — só roda 1 vez antes do treino começar.

EXP-003: Conditioning Regularization
O quê: A cada passo de treino, adiciona uma "multa" se as portas ficarem desiguais.

Como: Verifica "quão longe as portas estão de serem todas iguais" e adiciona essa diferença ao erro. Assim o optimizer é forçado a manter as portas uniformes.

Diferença vs 002: Age continuamente, não só na init.

Custo: Uma multiplicação de matrizes extra por step.

EXP-004: Soft Weight Tying
O quê: Puxa o head pra ficar parecido com a tabela de embeddings.

Por que funciona: A tabela de embeddings aprende boas representações naturalmente (tokens similares ficam próximos). Se o head fica parecido com ela, herda essa boa estrutura. É como copiar a organização de uma biblioteca que já funciona bem, em vez de organizar do zero.

Custo: Calcula a diferença entre head e embedding a cada step.

EXP-005: Contrastive Loss
O quê: Em vez de melhorar as portas, constrói uma escada de incêndio que contorna o corredor apertado.

Como: Além do loss normal (que passa pelas 512 portas), adiciona um segundo loss que age direto nos 512 números internos, sem passar pelo head. É como dar feedback direto ao autor, sem precisar do bilhete.

Diferença: Não melhora a compressão — simplesmente a evita.

Custo: Cálculo extra de similaridade entre representações.

EXP-006: Factored Head
O quê: Troca o corredor de 512 portas por dois corredores em sequência — 512→2048 e depois 2048→8192 — com uma "curva" (não-linearidade) no meio.

Por que funciona: A curva (GELU) faz com que cada input use portas diferentes. É como se o corredor mudasse de formato dependendo de quem está passando. Na soma de todos os inputs, é como ter muito mais que 512 portas efetivas.

Custo: Mais parâmetros e mais cálculo no head.

Resumo
Exp	Analogia	Quando age
002	Iguala as portas no início	Só na init
003	Multa se as portas ficarem desiguais	Todo step
004	Copia organização da embedding	Todo step
005	Escada de incêndio (bypass)	Todo step
006	Corredor que muda de formato	Todo step
