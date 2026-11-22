#!/usr/bin/env python3
"""Cria a estrutura de diretórios locais do Boccia Vision, sem sobrescrever dados.

Uso a partir da raiz do repositório:
    python scripts/setup_project.py --dry-run
    python scripts/setup_project.py
    python scripts/setup_project.py --versoes v1

Não baixa arquivos do Drive, não cria YAMLs e não gera datasets processados.
"""

from __future__ import annotations

import argparse
from pathlib import Path

FONTES = ("youtube", "local", "synthetic")
VERSOES = ("v1", "v2")


def diretorios_planejados(versoes: tuple[str, ...]) -> list[Path]:
    """Define pastas de entrada, processamento e execução do projeto."""
    pastas = [
        Path("data/videos/youtube"),
        Path("data/videos/local"),
        Path("runs/detect"),
        Path("runs/predict"),
    ]

    for fonte in FONTES:
        for versao in versoes:
            origem = Path("data/roboflow") / fonte / versao
            pastas.extend((origem / "images", origem / "labels"))

    # Datasets derivados serão criados pelo futuro script de processamento.
    for versao in versoes:
        pastas.append(Path("data/processed") / versao)

    return sorted(set(pastas), key=lambda p: p.as_posix())


def validar_projeto(raiz: Path, pastas: list[Path]) -> None:
    """Identifica conflitos antes de qualquer alteração."""
    if not raiz.is_dir() or not (raiz / "scripts").is_dir() or not (raiz / "README.md").is_file():
        raise ValueError(f"Não parece ser a raiz do repositório Boccia Vision: {raiz}")

    for relativa in pastas:
        caminho = raiz
        for parte in relativa.parts:
            caminho = caminho / parte
            if caminho.is_symlink():
                raise ValueError(f"Link simbólico não permitido: {caminho}")
            if caminho.exists() and not caminho.is_dir():
                raise FileExistsError(f"Caminho necessário existe como arquivo: {caminho}")


def inicializar(raiz: Path, versoes: tuple[str, ...], dry_run: bool = False) -> tuple[int, int]:
    pastas = diretorios_planejados(versoes)
    validar_projeto(raiz, pastas)

    pendentes = [p for p in pastas if not (raiz / p).is_dir()]
    existentes = len(pastas) - len(pendentes)

    print(f"Projeto: {raiz}")
    print(f"Versões: {', '.join(versoes)}")
    print(f"Diretórios-alvo: {len(pastas)} | existentes: {existentes} | novos: {len(pendentes)}")
    print("\nPRÉVIA (sem alterações):" if dry_run else "\nCriando apenas os diretórios ausentes:")

    for relativa in pendentes:
        if not dry_run:
            (raiz / relativa).mkdir(parents=True, exist_ok=True)
        print(f"  {'CRIARIA' if dry_run else 'CRIADO'}  {relativa.as_posix()}/")

    if not pendentes:
        print("  Todas as pastas planejadas já existem.")
    if not dry_run:
        print("\nConcluído: arquivos existentes preservados.")
        print("Próximo passo: baixar imagens, labels e data.yaml do Google Drive.")
    return len(pendentes), existentes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raiz",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Raiz do repositório (padrão: pasta acima de scripts/).",
    )
    parser.add_argument(
        "--versoes", nargs="+", choices=VERSOES, default=list(VERSOES),
        help="Versões a inicializar (padrão: v1 v2).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Mostra as pastas sem criá-las.")
    args = parser.parse_args()
    versoes = tuple(dict.fromkeys(args.versoes))
    try:
        inicializar(args.raiz.expanduser().resolve(), versoes, args.dry_run)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
