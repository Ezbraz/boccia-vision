# Boccia Vision — preparação do experimento piloto v1

## Objetivo e alcance

Criar **6 conjuntos YOLO de detecção**: duas rodadas de avaliação (A e B) × três cenários de composição (Real, Sintético e Híbrido 50/50). Os resultados são **exploratórios**; a versão final precisa de mais partidas independentes.

**Entradas existentes, não modificadas:**

```text
data/roboflow/
  youtube/v1/{images,labels,data.yaml}     # 121 imagens: hk_court1_trecho
  local/v1/{images,labels,data.yaml}       # 120 imagens: p1 + p2, mesma partida
  synthetic/v1/{images,labels,data.yaml}   # 121 imagens: BocciaVision_01 e _03
```

**Agrupamento:** `p1` e `p2` constituem a **mesma partida local**; `hk_court1_trecho` é outra partida; `BocciaVision_01` e `BocciaVision_03` são dois ambientes virtuais distintos. Os nomes esperados estão fixados no código **somente para a v1**. O script falha se encontrar um grupo novo, para não misturá-lo silenciosamente.

## Protocolo

| Rodada | Origem real elegível para treino/validação | Teste externo real |
|---|---|---|
| A | Local: p1 + p2 | YouTube: 121 imagens |
| B | YouTube | Local: p1 + p2, 120 imagens |

Em **cada rodada**, o script reserva uma cauda temporal de 20% da partida de treino como **validação interna**, retirando do treino até 5 índices de frame imediatamente anteriores ao começo dessa cauda. A validação é a **mesma para os três cenários**, inclusive o modelo sintético. Assim, a escolha do checkpoint pode utilizar dados reais embora o treinamento sintético não os utilize. A validação ainda é correlacionada com a partida de treino e **não é independente por partida**. O teste externo permanece separado por partida.

**Orçamento por cenário (conjunto de treino, após reservar validação):**

- **Real:** 60 imagens reais.
- **Sintético:** 60 imagens sintéticas, 30 por ambiente virtual.
- **Híbrido 50/50:** 30 reais + 30 sintéticas (15 de cada ambiente).

O Híbrido usa subconjuntos das seleções já feitas para Real e Sintético. Essas imagens podem aparecer no treino de **modelos diferentes** para permitir comparações pareadas; elas nunca aparecem em treino e teste do **mesmo modelo**. Não há ampliação artificial offline no preparador. Imagens sintéticas excedentes não entram nos conjuntos principais do piloto.

> **Importante:** as imagens de teste das rodadas A e B são para avaliação, não para selecionar épocas, hiperparâmetros, augmentations ou outros ajustes. Se as rodadas forem usadas repetidamente para decidir o projeto, deixam de ser testes inéditos; coletar partidas inteiramente novas para o teste final.

## Uso no Windows / PyCharm

Copie `scripts/prepare_pilot_v1.py` para o diretório `scripts/` do repositório. O script requer PyYAML (normalmente instalado com Ultralytics):

```powershell
python -m pip install pyyaml
```

Na **raiz do repositório**, primeiro confira o plano sem criar dados:

```powershell
python scripts/prepare_pilot_v1.py --dry-run
```

Depois execute a geração real:

```powershell
python scripts/prepare_pilot_v1.py
```

A saída fica em:

```text
data/processed/v1/pilot_seed42/color/
  split_manifest.json
  rodada_a/
    real/{train,valid,test,data.yaml}
    synthetic/{train,valid,test,data.yaml}
    hybrid/{train,valid,test,data.yaml}
  rodada_b/
    real/{train,valid,test,data.yaml}
    synthetic/{train,valid,test,data.yaml}
    hybrid/{train,valid,test,data.yaml}
```

As pastas `train`, `valid` e `test` contêm, cada uma, `images/` e `labels/`. O `data.yaml` de cada cenário usa caminhos absolutos até o dataset local para funcionar com o Ultralytics no PyCharm. O `.gitignore` atual já ignora `data/processed/`.

**Modo classe genérica (arquivo separado, mesmos splits):**

```powershell
python scripts/prepare_pilot_v1.py --class-mode generic
```

**Semente de seleção:** `--seed 42`, fixa para comparação de modelos. Essa **não** é a semente de treinamento. O script não sobrescreve uma pasta de saída existente; para um novo protocolo use `--output` com outro caminho ou outra semente.

## Primeira validação e treinamento (apenas depois de revisar dry-run)

Primeiro treino ilustrativo, mantendo a estrutura atual de `train_yolo.py`:

```powershell
python scripts/train_yolo.py --data data/processed/v1/pilot_seed42/color/rodada_a/real/data.yaml --nome piloto_v1_a_real_seed42 --seed 42
```

O script de treinamento existente usa as **augmentations padrão do Ultralytics**. Gerar os dados sem augmentation offline **não** equivale a desativar augmentation durante o treino. Para comparar `com/sem augmentation`, o código de treinamento precisará de argumentos explícitos; não trate essa comparação como já implementada.

**Não execute os 36 treinamentos definitivos neste momento.** Primeiro audite as seis saídas, faça testes de carregamento e um pequeno experimento da v1.

## Reprodutibilidade

O `split_manifest.json` registra versão, semente, composições, grupos, frames de origem, caminhos relativos, hashes SHA-256 e número de anotações. Inclui também as exclusões por proximidade temporal. Os IDs de classe são normalizados **pelos nomes no YAML** (a ordem no Roboflow pode variar); `--class-mode generic` preserva as caixas e transforma todas as classes em `ball`. Labels vazias permanecem válidas.

Para versionar o manifesto, **depois de validar o resultado**, copie o JSON para `experiments/manifests/`:

```powershell
New-Item -ItemType Directory -Force .\experiments\manifests | Out-Null
Copy-Item .\data\processed\v1\pilot_seed42\color\split_manifest.json .\experiments\manifests\pilot_v1_seed42_color.json
```

Não envie imagens, labels, pesos ou diretórios de dados para o Git.

## Limitações explícitas

1. Existem apenas **duas partidas reais**. O teste externo por partida é adequado para investigar transferência entre origens, **mas insuficiente para alegar generalização ampla**.
2. A validação temporal interna não é independente por partida; a semelhança entre p1 e p2 também deve ser considerada.
3. Os dois ambientes virtuais são independentes enquanto cenários, mas foram produzidos pelo mesmo processo de renderização.
4. O teste por frames de uma única partida contém observações correlacionadas. Não interpretar cada frame como uma repetição independente.
5. O detector sintético usa validação real comum para escolher o checkpoint; isso deve ser descrito ao apresentar os resultados.
6. O preparador **não** detecta automaticamente duplicatas *visualmente semelhantes*. Detecta conteúdos binários idênticos e conflitos de nomes/frames; inspeção humana continua necessária.

## Evolução para a v2 definitiva

Capturar novas partidas/sessões reais independentes e novas famílias de cenas virtuais; congelar um conjunto de teste real totalmente novo **antes** dos experimentos; dividir por grupos independentes treino/val/test; calibrar o orçamento e a política de augmentation; só então executar a matriz definitiva e avaliar diferenças entre seeds. **Não reutilizar automaticamente as proporções ou os grupos fixos deste piloto na v2.**
