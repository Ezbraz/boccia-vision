# Boccia Vision — Guia de primeira execução

Este guia explica **como preparar o ambiente e executar o fluxo disponível na branch `develop`** do projeto Boccia Vision, desde o clone até a primeira inferência com YOLO.

> **Estado do projeto:** o modelo decidido é o **YOLO26s**, e `train_yolo.py` já usa `yolo26s.pt` por padrão. O caminho padrão de `--data` ainda é legado (`data/roboflow/v1/data.yaml`), portanto informe explicitamente o YAML de um dataset processado. A nova integração com Drive e o processador multi-origem ainda não estão implementados; este guia distingue o fluxo legado do fluxo planejado. Homografia, tracking e medição também não estão implementados.

## 0. Pré-requisitos e dados necessários

Antes de começar, tenha:

- **Git** instalado.
- **Python 3.11** instalado (versão recomendada para padronizar o ambiente entre colegas).
- Acesso à internet para instalar dependências e baixar os pesos pré-treinados na primeira execução.
- **Um dataset YOLO já anotado**, fornecido separadamente pela equipe (veja a etapa 3).
- Opcional: GPU compatível para acelerar o treinamento. O modelo também pode executar na CPU, mais lentamente.

**Importante:** os diretórios `data/` com imagens/anotações e `runs/` com pesos/resultados locais **não estão versionados no Git**, de acordo com o `.gitignore` atual. **Clonar o GitHub, sozinho, não fornece os dados necessários para treinar ou avaliar um modelo.** Combine com a equipe como receber o dataset, respeitando as permissões de uso das imagens.

## 1. Clonar o repositório e selecionar a branch

Execute no terminal:

```bash
git clone -b develop https://github.com/Ezbraz/boccia-vision.git
cd boccia-vision
```

Para conferir a branch atual:

```bash
git branch --show-current
```

O resultado esperado é `develop`.

> **Não execute `organizar_repositorio.py` na primeira instalação.** Esse utilitário foi criado para uma migração histórica de diretórios e não é necessário para começar a usar a estrutura atual.

## 2. Criar o ambiente virtual e instalar as dependências

Crie o ambiente virtual na raiz do projeto:

```bash
python -m venv .venv
```

Ative o ambiente conforme seu sistema operacional:

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS (bash/zsh):**

```bash
source .venv/bin/activate
```

Com o ambiente ativado, instale os pacotes:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -U ultralytics
```

A última linha garante a atualização do Ultralytics para uma versão com suporte à família YOLO26. **As versões ainda não estão fixadas em `requirements.txt`**; para os experimentos definitivos, a equipe deverá registrar as versões efetivamente utilizadas.

Confira a instalação:

```bash
python -c "import ultralytics, cv2; print('Ultralytics:', ultralytics.__version__); print('OpenCV:', cv2.__version__)"
```

Para conferir se o PyTorch reconhece uma GPU CUDA (opcional):

```bash
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
```

> Se o computador tiver GPU NVIDIA e o comando retornar `False`, consulte a instalação de PyTorch adequada ao seu sistema/CUDA. Não é necessário ter GPU para testar o ambiente, mas o treinamento em CPU pode ser demorado.

## 3. Localização dos datasets: organização nova e formato legado

**Organização aprovada para os dados do Drive:** `data/roboflow/{youtube,local,synthetic}/v1/`, cada origem com `images/`, `labels/` e `data.yaml`. O script atual de divisão **ainda não aceita essa estrutura**.

O script **legado** de reorganização procura uma única exportação YOLO em:

```text
data/
└── roboflow/
    └── v1/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
```

**Não misture ou mova os três datasets novos para esse caminho legado.** A estrutura acima serve somente para compreender ou reproduzir o fluxo antigo. Na organização aprovada, cada imagem deve ter um `.txt` com o mesmo nome-base dentro da respectiva pasta `labels/` (um arquivo vazio pode representar uma imagem sem objetos).

No esquema de três classes atualmente usado pelo script:

```yaml
train: train/images
val: valid/images
test: test/images
names: ['blue_ball', 'red_ball', 'white_ball']
```

**Atenção:** confira os índices das classes no `data.yaml` exportado e nas anotações. **Não substitua os nomes/índices automaticamente** se o dataset tiver outro esquema de classes. O `reorganizar_dataset.py` atual possui essa lista de três classes fixa no código e precisará ser adaptado para o experimento de classe genérica `bola`.

## 4. Dividir o dataset com uma semente registrada (fluxo legado)

**Atenção:** o comando abaixo só se aplica ao formato legado descrito acima, com pelo menos três grupos/vídeos distintos. **Não execute este divisor sobre os três datasets do Drive**, pois eles usam uma organização diferente e ainda não temos grupos independentes suficientes para o split por vídeo.

Somente para reproduzir um experimento antigo compatível, execute na raiz:

```bash
python scripts/reorganizar_dataset.py --modo video --seed 42
```

O script:

1. Reúne as imagens e anotações presentes em `train/`, `valid/` e `test/` do dataset de origem.
2. Tenta manter frames de um mesmo vídeo no mesmo subconjunto.
3. Produz uma nova divisão, por padrão **70% treino / 15% validação / 15% teste** (as proporções reais podem diferir devido ao agrupamento por vídeo).
4. Gera a pasta `data/processed/video_seed42/`, incluindo `data.yaml`, `seed.txt` e `split_manifest.json`.

**Limitações importantes antes de usar:**

- O modo `video` exige **pelo menos três grupos/vídeos distintos**.
- O reconhecimento de vídeo depende principalmente de nomes no padrão `..._mp4-0001...`. Imagens com outros nomes podem ser tratadas como grupos separados; é preciso revisar esse comportamento antes dos experimentos científicos.
- O script **não sobrescreve** uma pasta de saída que já exista. Se `data/processed/video_seed42/` já existir, revise a pasta antes de escolher outra saída ou remover dados antigos.
- **Não utilize `--modo aleatorio` para misturar frames correlacionados entre treino e teste** numa avaliação científica.
- Esta etapa é a execução disponível hoje; **a metodologia definitiva de separação e os testes independentes ainda serão ajustados pela equipe**.

Se você **já recebeu da equipe um dataset processado e validado**, com seu próprio `data.yaml`, **pule esta etapa** e use o caminho desse YAML diretamente na etapa 5.

## 5. Treinar o modelo YOLO26s

Com um dataset **já processado e validado** presente (por exemplo, um conjunto legado compatível), execute:

```bash
python scripts/train_yolo.py --data data/processed/video_seed42/data.yaml --modelo yolo26s.pt --epocas 80 --imgsz 640 --batch 8 --seed 42 --nome primeiro_teste_yolo26s
```

O que significam os principais argumentos:

| Argumento | Significado |
| --- | --- |
| `--data` | Caminho do `data.yaml` do dataset escolhido |
| `--modelo yolo26s.pt` | Usa a variante Small do YOLO26, definida para o TCC |
| `--epocas 80` | Limite máximo de épocas solicitado ao treinamento |
| `--imgsz 640` | Resolução de entrada |
| `--batch 8` | Quantidade de imagens por lote |
| `--seed 42` | Semente do treinamento |
| `--nome primeiro_teste_yolo26s` | Nome da execução nos resultados |

Na primeira execução, o Ultralytics pode baixar automaticamente o arquivo `yolo26s.pt`. O resultado será criado em uma pasta semelhante a:

```text
runs/detect/primeiro_teste_yolo26s/
└── weights/
    ├── best.pt
    └── last.pt
```

O script usa `deterministic=True` e `workers=0`. **Ele ainda não configura explicitamente as políticas de data augmentation**, portanto os padrões da versão instalada do Ultralytics podem ser aplicados. **Não interprete este treinamento isolado como um dos experimentos controlados finais** sem antes revisar esse aspecto.

Se houver erro de memória de GPU, considere reduzir o parâmetro `--batch`, por exemplo para `4` ou `2`, documentando a mudança.

## 6. Fazer uma predição com o modelo treinado

Escolha uma imagem que esteja disponível no computador (substitua o caminho de exemplo pelo arquivo real):

```bash
python scripts/predict.py --imagem data/processed/video_seed42/test/images/EXEMPLO.jpg --pesos runs/detect/primeiro_teste_yolo26s/weights/best.pt
```

**Substitua `EXEMPLO.jpg` pelo nome de uma imagem existente.**

O script salva as imagens com as detecções em `runs/predict/` (em uma subpasta nomeada pelo Ultralytics). Os arquivos dessa pasta são locais e não são enviados automaticamente para o GitHub.

## 7. Onde ficam as coisas?

| Diretório/arquivo | Responsabilidade |
| --- | --- |
| `scripts/reorganizar_dataset.py` | Reunir e dividir o dataset com semente |
| `scripts/train_yolo.py` | Treinar o detector |
| `scripts/predict.py` | Executar inferência em uma imagem |
| `data/roboflow/{youtube,local,synthetic}/v1/` | Datasets de origem organizados por fonte (novo padrão; ainda não processado automaticamente) |
| `data/processed/` | Datasets reorganizados localmente |
| `runs/detect/` | Saídas dos treinamentos e pesos |
| `runs/predict/` | Imagens geradas por inferência |
| `src/boccia_vision/` | Módulos futuros de detecção, homografia, tracking e medição |
| `experiments/` | Local planejado para configurações e resultados experimentais |
| `docs/` | Documentação do projeto |
| `tests/` | Testes automatizados (ainda não implementados) |

## 8. Problemas comuns

**`FileNotFoundError: Dataset não encontrado`**  
Confirme se o caminho fornecido em `--data` aponta para o `data.yaml` de um dataset processado e validado. O caminho padrão de `train_yolo.py` ainda usa o layout legado; os datasets originais separados por origem não podem ser treinados diretamente sem o processamento necessário.

**`Divisao por video exige ao menos 3 videos/grupos de origem`**  
O dataset não possui ao menos três grupos reconhecidos pelo script. Confira a quantidade de vídeos e o padrão de nomeação dos arquivos; não substitua por divisão aleatória sem avaliar o risco de vazamento entre conjuntos.

**`A pasta ... ja existe. Nao sera sobrescrita`**  
A saída já foi gerada anteriormente. Verifique os arquivos existentes; para executar uma nova divisão, informe outro diretório em `--saida`.

**`ModuleNotFoundError: No module named 'ultralytics'`**  
Ative o ambiente virtual criado na etapa 2 e reinstale as dependências nele.

**Falha ao baixar `yolo26s.pt`**  
Confira sua conexão e a versão do Ultralytics instalada.

**Treinamento muito lento ou erro de memória**  
Confira a disponibilidade de GPU e o valor de `--batch`. Registre quaisquer ajustes nos parâmetros para manter rastreabilidade.

## 9. Limites desta primeira execução e próximos passos do projeto

O fluxo legado disponível hoje (apenas com dataset compatível) é:

```text
Clonar → Instalar → Receber dataset legado → Dividir → Treinar YOLO26s → Predizer
```

O fluxo planejado para os datasets atuais do Drive é:

```text
Clonar → Preparar pastas → Sincronizar Drive → Validar/processar Real, Sintético e Híbrido → Treinar → Avaliar
```

O `setup_project.py` e o `download_datasets.py` já foram implementados. Primeiro execute `python scripts/setup_project.py` (ou a prévia com `--dry-run`); depois configure o rclone conforme `docs/README_DOWNLOAD_DRIVE.md` e execute `python scripts/download_datasets.py --version v1`. A construção dos conjuntos Real, Sintético e Híbrido ainda depende da refatoração de `reorganizar_dataset.py`.

Ainda não estão implementados na branch `develop`:

- Automação da matriz de experimentos comparativos.
- Avaliações cruzadas entre as origens dos datasets.
- Controle explícito das condições **com/sem augmentation**.
- Alternância automática entre o esquema de **uma classe** e **três classes**.
- Pipeline funcional de homografia, rastreamento, distâncias e determinação de resultados.

Essas funcionalidades serão adicionadas conforme a metodologia do TCC evoluir. Até lá, siga as etapas acima como um **teste funcional de instalação, treinamento e inferência**, não como execução do protocolo científico completo.

## Referências úteis

- [Repositório Boccia Vision — branch develop](https://github.com/Ezbraz/boccia-vision/tree/develop)
- [Instalação oficial do Ultralytics](https://docs.ultralytics.com/quickstart/)
- [Modelos YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [Guia oficial de treinamento](https://docs.ultralytics.com/modes/train/)
