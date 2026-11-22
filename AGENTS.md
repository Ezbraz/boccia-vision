# AGENTS.md — Boccia Vision

Instruções permanentes para o Codex e outros agentes que trabalharem neste repositório.

## Contexto e fonte de verdade

- Projeto de TCC: **Boccia Vision**, visão computacional para bocha paralímpica.
- Leia `METHODOLOGY.md` antes de mudar preparação de dados, treinamento ou avaliação.
- Arquitetura **escolhida para o projeto: YOLO26s** (`yolo26s.pt`), pela biblioteca Ultralytics.
- Repositório identificado no GitHub como `Ezbraz/boccia-vision`, branch de trabalho `develop` (link antigo `erixalv/boccia-vision` redireciona).
- Distinguir rigorosamente **implementado**, **decidido**, **proposto** e **pendente**. Não apresentar funcionalidades ou experimentos planejados como concluídos.
- Se o código divergir da metodologia, relatar a divergência antes de alterar o comportamento.

## Decisões científicas vigentes (22/11/2026)

- **Três cenários principais:** **Real** (YouTube + local), **Sintético** (SketchUp + V-Ray), **Híbrido balanceado** (YouTube + local + sintético).
- **Híbrido completo** (todos os dados elegíveis, sem a mesma restrição de balanceamento) é um estudo **complementar**.
- **Duas configurações de classes:** bolas diferenciadas por cor (`blue_ball`, `red_ball`, `white_ball`), ou classe única (`ball`). Confirmar nomes e IDs nos YAMLs, nunca confiar na ordem presumida.
- **Dois regimes:** com e sem data augmentation, restrito ao treino.
- **Três sementes de treinamento:** `42`, `123`, `2026`.
- **Total principal: 3 datasets × 2 regimes × 2 esquemas de classes × 3 sementes = 36 treinamentos.** O plano anterior de 48 foi substituído pela unificação das fontes YouTube e local no cenário Real.
- Piloto com cerca de 120 imagens por origem é uma prova de viabilidade, **não** evidência final.

## Origem dos dados, armazenamento e splits

- Roboflow gerencia imagens/anotações; tags de origem: `youtube`, `local`, `synthetic`.
- Exportação adotada: **YOLO26 detecção**, preservando imagens, labels `.txt` e `data.yaml`. Preferir imagens originais, sem augmentation no Roboflow.
- Entradas por origem: `data/roboflow/youtube/v1/`, `data/roboflow/local/v1/`, `data/roboflow/synthetic/v1/`; as versões `v2/` serão usadas no estudo definitivo (pastas ignoradas pelo Git).
- No Google Drive, cada origem em `v1/` contém `images/`, `labels/` e `data.yaml`. Outras exportações podem vir com `train/`, `valid/` e `test/`; **não usar os splits de origem sem revalidar**.
- **Separar por grupos independentes:** vídeo/partida/sessão nos reais; família de cena/renderização nos sintéticos. Variantes e frames correlacionados devem ficar no mesmo subconjunto.
- **Nunca contaminar teste:** sem imagens do teste em treino, balanceamento, augmentation ou seleção de hiperparâmetros. Manter grupos e splits congelados nas comparações e nas três sementes de treinamento.
- Reconstruir IDs das classes pelo **nome semântico** de cada YAML; verificar objetos, bounding boxes normalizadas e integridade imagem/label. Para a versão genérica, trocar a classe, **não** as caixas.
- Registrar `group_id`, `source`, splits, versões e seeds em manifests. Não sobrescrever datasets nem saídas existentes por padrão.
- Não versionar vídeos, imagens de dataset, pesos, `runs/`, `data/processed/` nem segredos de integração Google Drive.

## Estado atual dos scripts (inspeção da branch develop em 22/11/2026)

- `scripts/train_yolo.py` já usa **`yolo26s.pt` como padrão** para `--modelo`. O parâmetro `--data`, porém, ainda aponta por padrão para o caminho legado `data/roboflow/v1/data.yaml`; informe explicitamente `--data` com o YAML de um dataset processado e validado.
- `scripts/reorganizar_dataset.py` ainda lê **uma exportação** em `data/roboflow/v1` e tem proporção default `70/15/15`. **Não implementa** o pipeline multi-origem e os híbridos; requer refatoração e testes.
- `scripts/predict.py` executa inferência com pesos locais.
- `requirements.txt` contém `ultralytics` e `opencv-python`, sem versões fixadas.
- `src/boccia_vision/` para código reutilizável, `scripts/` para CLIs, `tests/` para testes, `experiments/` para configurações/manifests/resultados textuais, `docs/` para documentação.
- `scripts/setup_project.py` está implementado e prepara os diretórios locais de `v1` e `v2` sem sobrescrever dados. `scripts/download_datasets.py` realiza o download de exportações via rclone, exigindo configuração local de acesso ao Drive. A integração real ainda deve ser validada em máquina da equipe.

## Como trabalhar neste repositório

1. Ler código relevante e metodologia, identificar suposições e explicar as mudanças.
2. Preferir funções pequenas/testáveis, `pathlib`, `argparse`, compatibilidade Windows/PyCharm.
3. Validar caminhos, IDs de classes, caixas YOLO em `[0,1]`, pares imagem/label, nomes repetidos, grupos e duplicatas. Labels vazios podem representar negativos válidos.
4. Criar testes automatizados para processamento de dados, conversão de classe e prevenção de vazamento.
5. Para comparação justa, manter checkpoint, splits, hardware, hiperparâmetros, critérios e protocolo de avaliação equivalentes; variar somente fatores planejados.
6. Registrar seeds, versão do dataset, commit Git, versões do ambiente, parâmetros, métricas e artefatos, sem inventar resultados.
7. Antes de mudar metodologia, apresentar a alteração como proposta; atualizar `METHODOLOGY.md` quando aprovada.
8. Assinalar como **PENDENTE** qualquer split final, proporção híbrida, protocolo de teste ou hiperparâmetro que ainda não tenha sido definido.
9. Google Drive é fonte de distribuição das exportações; integração deve ser parametrizada e não guardar credenciais no Git.

## Escopo mais amplo

Homografia da quadra (visão superior), tracking de bolas, medição de distâncias ao jack e identificação da bola vencedora integram a visão geral do TCC, mas não são fatores da matriz principal de detecção nem devem ser descritos como já concluídos.
