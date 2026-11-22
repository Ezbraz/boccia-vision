#!/usr/bin/env python3
"""Corrige referências desatualizadas do Boccia Vision sem alterar a metodologia.

Uso, na raiz do clone do repositório:
    python corrigir_documentacao.py           # mostra prévia
    python corrigir_documentacao.py --aplicar # altera os arquivos, após validar todos

Não altera datasets, imagens, scripts de treino nem a pasta .git.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path


REPLACEMENTS: dict[str, list[tuple[str, str]]] = {
    "AGENTS.md": [
        ("`docs/METHODOLOGY.md`", "`METHODOLOGY.md`"),
        (
            "- `scripts/train_yolo.py` aceita `--modelo`, mas seu **default atual é `yolo11n.pt`**; para usar o modelo decidido, passar `--modelo yolo26s.pt` até atualizar e validar o script.",
            "- `scripts/train_yolo.py` já usa **`yolo26s.pt` como padrão** para `--modelo`. O parâmetro `--data`, porém, ainda aponta por padrão para o caminho legado `data/roboflow/v1/data.yaml`; informe explicitamente `--data` com o YAML de um dataset processado e validado.",
        ),
        (
            "- Entradas planejadas: `data/roboflow/youtube/v1/`, `data/roboflow/local/v1/`, `data/roboflow/synthetic/v1/` (pastas ignoradas pelo Git).",
            "- Entradas por origem: `data/roboflow/youtube/v1/`, `data/roboflow/local/v1/`, `data/roboflow/synthetic/v1/`; as versões `v2/` serão usadas no estudo definitivo (pastas ignoradas pelo Git).",
        ),
        (
            "- Exportações podem conter `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels` e `data.yaml`; **não usar os splits de origem sem revalidar**.",
            "- No Google Drive, cada origem em `v1/` contém `images/`, `labels/` e `data.yaml`. Outras exportações podem vir com `train/`, `valid/` e `test/`; **não usar os splits de origem sem revalidar**.",
        ),
    ],
    "METHODOLOGY.md": [
        (
            "Estrutura planejada das entradas locais (não versionadas):",
            "Estrutura das entradas locais planejada para sincronização com o Drive (não versionadas no Git):",
        ),
        (
            "Cada versão exportada pode trazer `train/valid/test` com subpastas `images` e `labels`, mais `data.yaml`. Esses splits de exportação **não devem ser assumidos como os splits científicos finais**.",
            "Na versão `v1` presente no Google Drive, cada origem contém `images/`, `labels/` e `data.yaml` diretamente em sua pasta. As versões `v2` preservarão a separação por origem. Outras exportações podem trazer `train/valid/test`, mas esses splits **não devem ser assumidos como os splits científicos finais**.",
        ),
        (
            "**Divergência observada em 22/11/2026:** `scripts/train_yolo.py` ainda usa o **default `yolo11n.pt`**, embora aceite `--modelo`. Até sua atualização, usar explicitamente `--modelo yolo26s.pt`; não afirmar que o código já está padronizado em YOLO26s.",
            "**Estado verificado em 22/11/2026:** `scripts/train_yolo.py` já usa **`yolo26s.pt` por padrão**. Seu `--data` padrão ainda aponta para `data/roboflow/v1/data.yaml` (layout legado); usar `--data` explicitamente com o YAML de um conjunto processado e validado enquanto não houver refatoração.",
        ),
        (
            "| `scripts/train_yolo.py` | Treinamento Ultralytics; default atual `yolo11n.pt` |",
            "| `scripts/train_yolo.py` | Treinamento Ultralytics; modelo padrão `yolo26s.pt`, mas `--data` ainda aponta para o layout legado |",
        ),
        (
            "6. Atualizar o treinamento para YOLO26s e registrar parâmetros por experimento.",
            "6. Atualizar o caminho padrão de dados no treinamento, controlar regimes de augmentation e registrar parâmetros por experimento (modelo padrão já é YOLO26s).",
        ),
    ],
    "docs/README_PRIMEIRA_EXECUCAO.md": [
        (
            "> **Estado do projeto:** este documento descreve os scripts existentes na `develop` na revisão de 22/11/2026. O modelo definido para os testes e experimentos definitivos é o **YOLO26s**. No código atual, o `train_yolo.py` ainda tem `yolo11n.pt` como padrão; por isso, neste guia passamos **`--modelo yolo26s.pt` explicitamente**. Ainda não há um pipeline completo de homografia, tracking e medição.",
            "> **Estado do projeto:** o modelo decidido é o **YOLO26s**, e `train_yolo.py` já usa `yolo26s.pt` por padrão. O caminho padrão de `--data` ainda é legado (`data/roboflow/v1/data.yaml`), portanto informe explicitamente o YAML de um dataset processado. A nova integração com Drive e o processador multi-origem ainda não estão implementados; este guia distingue o fluxo legado do fluxo planejado. Homografia, tracking e medição também não estão implementados.",
        ),
        (
            "## 3. Colocar o dataset no local esperado\n\nO script de reorganização atual procura um dataset YOLO exportado do Roboflow em:",
            "## 3. Localização dos datasets: organização nova e formato legado\n\n**Organização aprovada para os dados do Drive:** `data/roboflow/{youtube,local,synthetic}/v1/`, cada origem com `images/`, `labels/` e `data.yaml`. O script atual de divisão **ainda não aceita essa estrutura**.\n\nO script **legado** de reorganização procura uma única exportação YOLO em:",
        ),
        (
            "Crie as pastas, se necessário, e extraia ali o dataset **já anotado**, preservando os nomes de imagens e labels. Cada imagem deve ter um arquivo `.txt` com o mesmo nome-base na pasta de labels correspondente, inclusive se for uma imagem sem objetos (nesse caso, o arquivo pode estar vazio).",
            "**Não misture ou mova os três datasets novos para esse caminho legado.** A estrutura acima serve somente para compreender ou reproduzir o fluxo antigo. Na organização aprovada, cada imagem deve ter um `.txt` com o mesmo nome-base dentro da respectiva pasta `labels/` (um arquivo vazio pode representar uma imagem sem objetos).",
        ),
        (
            "## 4. Reorganizar o dataset com uma semente registrada\n\nNa **raiz** do repositório, execute:",
            "## 4. Dividir o dataset com uma semente registrada (fluxo legado)\n\n**Atenção:** o comando abaixo só se aplica ao formato legado descrito acima, com pelo menos três grupos/vídeos distintos. **Não execute este divisor sobre os três datasets do Drive**, pois eles usam uma organização diferente e ainda não temos grupos independentes suficientes para o split por vídeo.\n\nSomente para reproduzir um experimento antigo compatível, execute na raiz:",
        ),
        (
            "Com o dataset processado presente, execute:",
            "Com um dataset **já processado e validado** presente (por exemplo, um conjunto legado compatível), execute:",
        ),
        (
            "python scripts/predict.py --imagem data/roboflow/v1/test/images/EXEMPLO.jpg --pesos runs/detect/primeiro_teste_yolo26s/weights/best.pt",
            "python scripts/predict.py --imagem data/processed/video_seed42/test/images/EXEMPLO.jpg --pesos runs/detect/primeiro_teste_yolo26s/weights/best.pt",
        ),
        (
            "| `data/roboflow/v1/` | Dataset anotado de origem, fornecido separadamente |",
            "| `data/roboflow/{youtube,local,synthetic}/v1/` | Datasets de origem organizados por fonte (novo padrão; ainda não processado automaticamente) |",
        ),
        (
            "Verifique se o dataset foi recebido, extraído em `data/roboflow/v1/` e se o caminho fornecido em `--data` aponta para um `data.yaml` existente.",
            "Confirme se o caminho fornecido em `--data` aponta para o `data.yaml` de um dataset processado e validado. O caminho padrão de `train_yolo.py` ainda usa o layout legado; os datasets originais separados por origem não podem ser treinados diretamente sem o processamento necessário.",
        ),
        (
            "O fluxo disponível hoje é:\n\n```text\nClonar → Instalar → Receber dataset → Dividir → Treinar YOLO26s → Predizer\n```",
            "O fluxo legado disponível hoje (apenas com dataset compatível) é:\n\n```text\nClonar → Instalar → Receber dataset legado → Dividir → Treinar YOLO26s → Predizer\n```\n\nO fluxo planejado para os datasets atuais do Drive é:\n\n```text\nClonar → Preparar pastas → Sincronizar Drive → Validar/processar Real, Sintético e Híbrido → Treinar → Avaliar\n```\n\nEsse novo fluxo ainda depende dos scripts `setup_project.py`, `download_datasets.py` e da refatoração de `reorganizar_dataset.py`.",
        ),
    ],
    "scripts/README.md": [
        (
            "# Scripts executáveis\n\n- `train_yolo.py`: treinar o detector.\n- `predict.py`: inferência em uma imagem.\n- `reorganizar_dataset.py`: divisão por vídeo ou por imagem (se você adicionou o script anterior).\n- `extrair_frames.py`: implementar quando iniciar a extração dos vídeos.\n",
            "# Scripts executáveis\n\n- `train_yolo.py`: treina o detector; modelo padrão `yolo26s.pt`. O argumento `--data` ainda tem um caminho legado e deve ser informado explicitamente.\n- `predict.py`: realiza inferência em uma imagem.\n- `reorganizar_dataset.py`: script **legado** para divisão de uma única exportação em `data/roboflow/v1/`. Não atende ainda aos conjuntos Real, Sintético e Híbrido.\n- `reset_primeiro_teste.py`: auxilia a remover artefatos locais de um teste antigo, com prévia e confirmação.\n\n## Scripts planejados (ainda ausentes na revisão)\n\n- `setup_project.py`: inicialização segura da estrutura local.\n- `download_datasets.py`: sincronização das fontes do Google Drive.\n- `extrair_frames.py`: extração de frames de vídeo.\n",
        ),
    ],
}


def corrigir(conteudo: str, trocas: list[tuple[str, str]], nome: str) -> str:
    novo = conteudo
    for antigo, atualizado in trocas:
        total = novo.count(antigo)
        if total == 1:
            novo = novo.replace(antigo, atualizado)
        elif total == 0 and atualizado in novo:
            # Reexecução: troca já aplicada.
            continue
        elif nome == "AGENTS.md" and antigo == "`docs/METHODOLOGY.md`" and total > 1:
            novo = novo.replace(antigo, atualizado)
        else:
            raise ValueError(f"{nome}: trecho esperado encontrado {total} vezes: {antigo[:90]!r}")
    return novo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", type=Path, default=Path.cwd(), help="Raiz do clone Git")
    parser.add_argument("--aplicar", action="store_true", help="Aplica as correções após validar todos os arquivos")
    args = parser.parse_args()
    raiz = args.raiz.expanduser().resolve()
    if not (raiz / ".git").exists() or not (raiz / "scripts/train_yolo.py").is_file():
        parser.error(f"Pasta não parece ser o clone do projeto: {raiz}")

    pendentes: dict[Path, str] = {}
    diffs = []
    try:
        for relativo, trocas in REPLACEMENTS.items():
            arquivo = raiz / relativo
            if arquivo.is_symlink() or not arquivo.is_file():
                raise ValueError(f"Arquivo ausente ou link simbólico: {relativo}")
            conteudo = arquivo.read_text(encoding="utf-8-sig")
            atualizado = corrigir(conteudo, trocas, relativo)
            if atualizado != conteudo:
                pendentes[arquivo] = atualizado
                diffs.extend(difflib.unified_diff(
                    conteudo.splitlines(keepends=True), atualizado.splitlines(keepends=True),
                    fromfile=f"a/{relativo}", tofile=f"b/{relativo}",
                ))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Erro: {exc}\nNenhum arquivo foi alterado.\n")

    print(f"Arquivos a atualizar: {len(pendentes)}")
    if not pendentes:
        print("Nada a fazer: correções já aplicadas.")
        return
    if not args.aplicar:
        print("".join(diffs))
        print("\nPRÉVIA: nada foi alterado. Execute com --aplicar para corrigir.")
        return

    for arquivo, atualizado in pendentes.items():
        arquivo.write_text(atualizado, encoding="utf-8")
        print(f"Atualizado: {arquivo.relative_to(raiz).as_posix()}")
    print("Concluído. Confira 'git diff' antes de criar o commit.")


if __name__ == "__main__":
    main()
