# Metodologia experimental — Boccia Vision

**Versão:** 22/11/2026  
**Propósito:** referência científica e técnica para a implementação do projeto de TCC.  
**Status dos itens:** **DECIDIDO**, **PROPOSTO**, **PENDENTE**. Nada neste documento implica que os experimentos já foram executados.

## 1. Questão de pesquisa e objetivos

Investigar como a origem e a composição dos dados de treinamento influenciam a detecção de bolas na bocha paralímpica, especialmente diante de variações de câmeras, distância, oclusão, iluminação, quadras e eventos de jogo.

O objetivo principal é comparar um detector **YOLO26s** treinado com dados **reais**, **sintéticos** e **híbridos**, mensurando desempenho e generalização em cenas independentes. Objetivos secundários: estudar o efeito de data augmentation, comparar classificação por cor versus uma classe genérica, verificar robustez por domínio de origem e tornar as avaliações reprodutíveis.

O TCC mais amplo também prevê homografia para vista superior, tracking, distâncias entre bolas e identificação da bola vencedora mais próxima do jack. **Essas são etapas separadas da matriz principal de detecção**; não são resultados já demonstrados.

## 2. Fontes de imagem e papel experimental

| Origem | Código | Descrição | Papel |
|---|---|---|---|
| YouTube | Y | Frames de partidas publicados em vídeo | Parte do dataset Real; avaliação por origem |
| Local | L | Gravações autorais em quadra | Parte do dataset Real; avaliação por origem |
| Sintético | S | Cenas em SketchUp + V-Ray | Dataset Sintético e mistura híbrida |
| Real | R | Combinação de Y e L | Cenário principal |
| Híbrido balanceado | H-bal | Combinação controlada de Y, L e S | Cenário principal |
| Híbrido completo | H-full | Total de dados elegíveis de Y, L, S | Estudo complementar |

**DECIDIDO:** Y e L formam um só cenário **Real** para treino principal, mas suas origens são preservadas nos manifests e métricas. O híbrido **balanceado** é utilizado nos experimentos centrais; o **completo** fica como análise complementar. A política exata de balanceamento, respeitando grupos independentes, continua **PENDENTE**.

### Produção de dados sintéticos

Usar o modelo de quadra do SketchUp + V-Ray para variar cores/texturas do piso, iluminação, câmera, distância, posicionamento, oclusão e presença de bolas fora da quadra, quando compatível com o caso de uso. As variações devem ser rastreadas por **família de cena**. **PROPOSTO:** automatizar anotações com máscaras/IDs de objeto, convertendo-as em bounding boxes YOLO via Python/OpenCV, mas somente após verificar a qualidade das anotações geradas.

### Política de anotação

Anotar consistentemente bolas detectáveis, inclusive parcialmente ocluídas, respeitando critérios documentados para objetos muito pequenos, ambíguos ou fora da quadra. Evitar que a regra de inclusão mude entre origens. Uma imagem sem objetos pode ser um negativo válido. Inspecionar o impacto de redimensionamento em bolas pequenas, particularmente em 640 px.

## 3. Exportação e padronização

**DECIDIDO:** exportações em **YOLO26 (Object Detection)** no Roboflow, preservando arquivos de imagem, labels `.txt` e `data.yaml` com resolução original quando possível. As tags de proveniência são `youtube`, `local`, `synthetic`. Não aplicar augmentation no Roboflow para não perder o controle do protocolo experimental; aplicar transformações exclusivamente no treinamento quando o regime exigir.

Estrutura das entradas locais planejada para sincronização com o Drive (não versionadas no Git):

```text
data/roboflow/
  youtube/v1/
  local/v1/
  synthetic/v1/
```

Na versão `v1` presente no Google Drive, cada origem contém `images/`, `labels/` e `data.yaml` diretamente em sua pasta. As versões `v2` preservarão a separação por origem. Outras exportações podem trazer `train/valid/test`, mas esses splits **não devem ser assumidos como os splits científicos finais**.

### Esquemas de classes

- **Específico por cor:** `blue_ball`, `red_ball`, `white_ball` (branca/jack), observando a nomenclatura validada dos datasets.
- **Genérico:** `ball`, unindo as três classes sem alterar as bounding boxes.

O script legado lista IDs com nomes na ordem azul, vermelha, branca; **validar os nomes e IDs em cada `data.yaml` antes de mesclar**. Reindexar pelas classes semânticas e rejeitar nomes desconhecidos até haver decisão explícita.

## 4. Divisões, independência e prevenção de vazamento

**DECIDIDO:** divisão por grupos, não por frames aleatórios. Para vídeos reais, o grupo deve representar vídeo, partida, sessão ou outro conjunto de frames altamente correlacionados. Para imagens sintéticas, utilizar família de cena/renderização. Variações de um mesmo frame/objeto não atravessam treino, validação e teste.

- Manter dados de teste **congelados e não utilizados** em treinamento, augmentation, balanceamento ou escolha de hiperparâmetros.
- Preservar os mesmos testes e regras de divisão nas comparações entre cenários; as três sementes de treino **não devem refazer o split**.
- Registrar uma tabela/manifesto com `image_id`, `source`, `group_id`, `split`, `dataset_version` e arquivo de anotação; procurar duplicatas e imagens quase idênticas.
- **PENDENTE:** definir percentuais finais e um conjunto real independente, considerando a quantidade efetiva de partidas, sessões e famílias de renderização.

**Estado do código:** `scripts/reorganizar_dataset.py` tem atualmente proporção padrão 70/15/15 e entrada fixa `data/roboflow/v1`. Esses valores e caminhos são **implementação legada, não protocolo final aprovado**. Se a diversidade de grupos for insuficiente para separar conjuntos de forma independente, registrar a limitação e ampliar o acervo ou rever o desenho experimental, sem criar falsa independência.

## 5. Matriz principal de treinamentos

| Fator | Valores |
|---|---|
| Treino | Real (Y+L), Sintético (S), Híbrido balanceado (Y+L+S) |
| Augmentation | Sem / Com (somente no treino) |
| Classificação | Por cor / Genérica |
| Semente do treino | 42 / 123 / 2026 |
| Modelo | YOLO26s (checkpoint `yolo26s.pt`) |

**Total principal: 3 × 2 × 2 × 3 = 36 treinamentos.** A proposta anterior de 48 execuções incluía YouTube e local como dois cenários de treinamento separados; foi substituída pela categoria agregada **Real**.

### Comparações complementares

1. **Híbrido completo**, sem limite de balanceamento, como estudo adicional (quantidade de execuções e composição final **PENDENTES**).
2. **Generalização entre domínios:** avaliar modelos em dados independentes Y, L e S e, se disponível, em um teste real externo congelado. É necessário identificar explicitamente quando o domínio de teste não é totalmente independente.
3. **Desempenho em situações difíceis:** bolas pequenas/distantes, oclusão, mudanças de piso, câmera e luz, quando houver metadados confiáveis.
4. **Piloto reduzido:** cerca de **120 imagens por origem** para testar integridade, preparação, treinamento e avaliação, **não** para defender conclusões finais.

## 6. Treinamento e comparação justa

**DECIDIDO:** usar a família YOLO26, modelo YOLO26s, com Ultralytics, mantendo igualdade de protocolo entre os cenários. **PENDENTE:** definir `imgsz`, `epochs`, `batch`, `patience`, política de augmentation específica, taxa de aprendizagem, orçamento de hardware e demais hiperparâmetros.

Controlar checkpoint inicial, conjuntos de teste, configurações, épocas máximas, early stopping, resolução, batch e métrica. Variar **somente o fator estudado** e não selecionar modelos pelo teste. Registrar versão Python/Ultralytics/PyTorch, GPU, seed, parâmetros, revisão Git e versão do dataset.

**Estado verificado em 22/11/2026:** `scripts/train_yolo.py` já usa **`yolo26s.pt` por padrão**. Seu `--data` padrão ainda aponta para `data/roboflow/v1/data.yaml` (layout legado); usar `--data` explicitamente com o YAML de um conjunto processado e validado enquanto não houver refatoração.

## 7. Métricas e análise quantitativa

**PROPOSTO:** registrar precisão, recall, F1, `mAP@0.50`, `mAP@0.50:0.95`, métricas por classe e por domínio de teste, matrizes de confusão e exemplos de falsos positivos/negativos. Apresentar média, desvio-padrão e amplitude entre as **três sementes**, com o mesmo split.

Comparar modelo por cor com modelo genérico em uma **base semântica comum** quando necessário: por exemplo, agregar as previsões por cor como `ball` para cálculo compatível, além de reportar separadamente as métricas por classe. Não comparar diretamente métricas de tarefas com espaços de classes distintos sem explicar a conversão.

A análise deve distinguir qualidade **in-domain** de generalização **cross-domain**, e reconhecer diferenças de tamanho/composição, ruído de anotação e número de grupos independentes. Relatar limitações, não atribuir causalidade sem controle apropriado.

## 8. Reprodutibilidade e organização dos artefatos

**PROPOSTO:** manter configurações, manifestos e métricas em `experiments` (versionáveis) e dados/pesos grandes em `data/processed/` e `runs/` (ignorados pelo Git):

```text
experiments/
  configs/
  manifests/
  results/
data/
  processed/
runs/
```

Campos mínimos: `experiment_id`, `dataset_variant`, `augmentation`, `class_mode`, `train_seed`, `split_id`, `dataset_version`, `checkpoint`, `imgsz`, `epochs`, `batch`, `patience`, `versions`, `hardware`, `git_commit`, `metrics`. Não versionar credenciais ou caminhos sensíveis.

## 9. Estado dos arquivos e backlog técnico

O repositório `Ezbraz/boccia-vision` na branch `develop` foi inspecionado em 22/11/2026:

| Arquivo | Estado verificado |
|---|---|
| `scripts/train_yolo.py` | Treinamento Ultralytics; modelo padrão `yolo26s.pt`, mas `--data` ainda aponta para o layout legado |
| `scripts/reorganizar_dataset.py` | Reorganiza **uma** exportação `data/roboflow/v1`; ainda sem composição Y/L/S e híbridos |
| `scripts/predict.py` | Inferência com pesos locais |
| `requirements.txt` | `ultralytics`, `opencv-python`, sem pinagem de versões |
| ``, `tests`, `experiments/`, `src/boccia_vision/` | Estruturas iniciais disponíveis |
| `.gitignore` | Ignora dados Roboflow, processados, vídeos, pesos e `runs` |

**Próximas implementações propostas, ainda não concluídas:**

1. Validar em computador da equipe a autenticação e a sincronização via `download_datasets.py` das exportações do Google Drive para `data/roboflow/{youtube,local,synthetic}/v1`. `setup_project.py` já está implementado para inicializar as pastas locais.
2. Inventariar imagens, grupos, labels, classes, duplicatas e diferenças entre origens.
3. Corrigir padronização e validar classes por nome semântico.
4. Planejar e congelar splits independentes, documentando o teste externo real.
5. Implementar construção dos datasets Real, Sintético, Híbrido balanceado e Híbrido completo, preservando grupos e manifestos.
6. Atualizar o caminho padrão de dados no treinamento, controlar regimes de augmentation e registrar parâmetros por experimento (modelo padrão já é YOLO26s).
7. Rodar piloto, validar avaliação cruzada e depois a matriz principal de 36 execuções.

## 10. Questões em aberto

- Quantos vídeos, partidas, sessões e famílias sintéticas independentes existem por origem?
- Qual será o split final e o teste real independente?
- Como balancear origens sem quebrar independência entre grupos?
- Qual resolução (640/960/1280) e quais parâmetros serão usados no protocolo definitivo?
- Qual política de augmentation respeita as características físicas das cenas?
- A exportação de máscaras de objetos no V-Ray será confiável para anotações automáticas?
- Quais avaliações complementares caberão no tempo e no orçamento computacional?

**Princípio central:** rastreabilidade, prevenção de vazamento e comparações científicas controladas, distinguindo decisões, propostas e resultados medidos.
