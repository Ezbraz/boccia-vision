#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove APENAS os artefatos locais do primeiro experimento do Boccia Vision.

Coloque este arquivo em scripts/ ou na raiz do repositório.

Prévia (não exclui nada):
    python scripts/reset_primeiro_teste.py

Executar, após conferir a prévia:
    python scripts/reset_primeiro_teste.py --aplicar

Se estiver fora do repositório:
    python reset_primeiro_teste.py --raiz /caminho/para/boccia-vision

Este script NÃO apaga os datasets originais, labels originais, código ou histórico Git.
Não apaga predições em runs/predict, pois podem corresponder a outros testes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SEED_PRIMEIRO_TESTE = 17111
NOME_TREINO = "treino_01"
CONFIRMACAO = "RESETAR PRIMEIRO TESTE"


def encontrar_raiz(caminho: Path, argumento: Path | None) -> Path:
    if argumento is not None:
        raiz = argumento.expanduser().resolve()
    else:
        candidatos = [caminho.resolve().parent, caminho.resolve().parent.parent]
        raiz = next((p for p in candidatos if (p / "scripts" / "train_yolo.py").is_file()), None)
        if raiz is None:
            raise ValueError("Não encontrei o repositório. Use --raiz CAMINHO_DO_REPOSITORIO.")

    if not (raiz / "scripts" / "train_yolo.py").is_file():
        raise ValueError(f"Pasta inválida (falta scripts/train_yolo.py): {raiz}")
    return raiz


def nome_de_run_do_primeiro_teste(nome: str) -> bool:
    """Inclui treino_01 e incrementos automáticos: treino_012, treino_013 etc."""
    if nome == NOME_TREINO:
        return True
    if not nome.startswith(NOME_TREINO):
        return False
    sufixo = nome[len(NOME_TREINO):]
    return sufixo.isdigit() and int(sufixo) >= 2


def localizar_artefatos(raiz: Path) -> list[Path]:
    candidatos = [raiz / "data" / "processed" / f"video_seed{SEED_PRIMEIRO_TESTE}"]
    runs = raiz / "runs" / "detect"
    if runs.is_dir() and not runs.is_symlink():
        candidatos.extend(
            p for p in runs.iterdir() if nome_de_run_do_primeiro_teste(p.name)
        )
    return sorted((p for p in candidatos if p.exists() or p.is_symlink()), key=str)


def verificar_alvo(raiz: Path, alvo: Path) -> None:
    raiz_real = raiz.resolve()
    if alvo.is_symlink():
        raise ValueError(f"Recusado: alvo é link simbólico: {alvo}")
    if not alvo.is_dir():
        raise ValueError(f"Recusado: alvo não é diretório: {alvo}")
    if raiz_real not in alvo.resolve().parents:
        raise ValueError(f"Recusado: alvo fora do repositório: {alvo}")
    # Não seguir symlinks nas pastas intermediárias.
    for pasta in (alvo.parent, alvo.parent.parent, alvo.parent.parent.parent):
        if pasta == raiz_real:
            break
        if pasta.is_symlink():
            raise ValueError(f"Recusado: pasta ancestral é link simbólico: {pasta}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", type=Path, help="Raiz do repositório (opcional)")
    parser.add_argument("--aplicar", action="store_true", help="Efetua a exclusão após confirmação digitada")
    args = parser.parse_args()

    raiz = encontrar_raiz(Path(__file__), args.raiz)
    alvos = localizar_artefatos(raiz)
    print(f"Repositório: {raiz}")

    if not alvos:
        print("Nenhum artefato do primeiro teste foi encontrado. Nada foi alterado.")
        return

    print("\nArtefatos do primeiro teste encontrados:")
    for alvo in alvos:
        verificar_alvo(raiz, alvo)
        print(f"  [EXCLUIR] {alvo.relative_to(raiz).as_posix()}/")

    print("\nPRESERVADOS: data/roboflow/, data/videos/, src/, scripts/, Git e runs/predict/.")

    if not args.aplicar:
        print("\nPRÉVIA APENAS. Nada foi excluído.")
        print("Para executar: python scripts/reset_primeiro_teste.py --aplicar")
        return

    print("\nAVISO: a exclusão das pastas acima é permanente.")
    resposta = input(f'Digite exatamente "{CONFIRMACAO}" para confirmar: ')
    if resposta != CONFIRMACAO:
        print("Cancelado. Nenhuma pasta foi removida.")
        return

    for alvo in alvos:
        # Revalidação imediatamente antes de cada exclusão.
        verificar_alvo(raiz, alvo)
        shutil.rmtree(alvo)
        print(f"  [REMOVIDO] {alvo.relative_to(raiz).as_posix()}/")

    print("\nReset concluído. Dados originais e código preservados.")


if __name__ == "__main__":
    main()
