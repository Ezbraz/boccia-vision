#!/usr/bin/env python3
"""Baixa exportações do Google Drive para data/roboflow/ sem sobrescrever arquivos.

Requisitos:
  1. Instalar rclone: https://rclone.org/install/
  2. Executar `rclone config` e criar um remote Google Drive chamado `boccia_drive`.
     Use permissões de leitura (drive.readonly). A autenticação fica somente na
     máquina de cada integrante, NUNCA no Git.
  3. A conta precisa ter acesso à pasta `data` compartilhada do projeto.

Exemplos na raiz do repositório:
  python scripts/download_datasets.py --version v1 --dry-run
  python scripts/download_datasets.py --version v1
  python scripts/download_datasets.py --version v1 --source local
  python scripts/download_datasets.py --version v2 --remote drive_da_equipe

A pasta raiz do Drive é configurada por ID (opção --drive-folder-id), independentemente
onde ela esteja na sua conta. Na raiz do Drive devem existir roboflow/ e videos/.
Este script nunca usa `rclone sync` nem transfere dados para o Drive.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ID da pasta `data/` fornecida pela equipe (não contém credenciais).
DRIVE_DATA_FOLDER_ID = "12im3QZE-9AKgwfJ4McS32XXAsibe1jyi"
ORIGENS = ("youtube", "local", "synthetic")
VERSOES = ("v1", "v2")
EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DownloadError(RuntimeError):
    """Problema de configuração, cópia ou integridade."""


def executar(comando: list[str]) -> str:
    """Chama o rclone sem shell e interrompe ao primeiro erro."""
    resultado = subprocess.run(comando, text=True, encoding="utf-8",
                               errors="replace", stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False)
    if resultado.returncode:
        saida = (resultado.stderr or resultado.stdout).strip()
        raise DownloadError(
            f"Falha no rclone (código {resultado.returncode}):\n{saida[:2500]}"
        )
    return resultado.stdout


def itens_remotos(remote: str, pasta: str, drive_folder_id: str) -> dict[str, bool]:
    """Lista filhos diretos e marca quais são diretórios."""
    resposta = executar([
        "rclone", "lsf", f"{remote}:{pasta}",
        "--drive-root-folder-id", drive_folder_id,
    ])
    itens: dict[str, bool] = {}
    for linha in resposta.splitlines():
        nome = linha.strip()
        if nome:
            itens[nome.rstrip("/")] = nome.endswith("/")
    return itens


def encontrar_pasta(itens: dict[str, bool], nome: str, local: str) -> str:
    """Compatibiliza nomes como Youtube/ e youtube/ sem presumir capitalização."""
    candidatos = [n for n, eh_pasta in itens.items() if eh_pasta and n.casefold() == nome.casefold()]
    if len(candidatos) != 1:
        raise DownloadError(
            f"Esperada uma pasta '{nome}' em {local}; encontradas: {candidatos or 'nenhuma'}."
        )
    return candidatos[0]


def validar_entrada_local(destino: Path, raiz: Path) -> None:
    """Evita gravar sobre arquivos e seguir links simbólicos locais."""
    if not raiz.is_dir() or not (raiz / "README.md").is_file() or not (raiz / "scripts").is_dir():
        raise DownloadError(f"Raiz do projeto inválida: {raiz}")
    for alvo in (destino / "images", destino / "labels", destino / "data.yaml"):
        atual = raiz
        for parte in alvo.relative_to(raiz).parts:
            atual = atual / parte
            if atual.is_symlink():
                raise DownloadError(f"Link simbólico não permitido: {atual}")
            if atual.exists():
                deve_ser_arquivo = atual == destino / "data.yaml"
                if deve_ser_arquivo and not atual.is_file():
                    raise DownloadError(f"O YAML existente não é um arquivo: {atual}")
                if not deve_ser_arquivo and not atual.is_dir():
                    raise DownloadError(f"Pasta esperada existe como arquivo: {atual}")


def conferir_pares(destino: Path) -> tuple[int, int]:
    """Confere correspondência de nomes; labels vazios são negativos válidos."""
    images_dir = destino / "images"
    labels_dir = destino / "labels"
    imagens = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSOES_IMAGEM]
    labels = [p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    ids_imagens = {p.stem for p in imagens}
    ids_labels = {p.stem for p in labels}
    sem_label = sorted(ids_imagens - ids_labels)
    sem_imagem = sorted(ids_labels - ids_imagens)
    if len(ids_imagens) != len(imagens) or len(ids_labels) != len(labels):
        raise DownloadError(f"Nomes-base duplicados em {destino}")
    if sem_label or sem_imagem:
        raise DownloadError(
            f"Pares inconsistentes em {destino}: "
            f"imagens sem label={len(sem_label)} ({sem_label[:3]}), "
            f"labels sem imagem={len(sem_imagem)} ({sem_imagem[:3]})."
        )
    if not imagens:
        raise DownloadError(f"Nenhuma imagem encontrada em {images_dir}")
    if not (destino / "data.yaml").is_file():
        raise DownloadError(f"Arquivo data.yaml ausente: {destino}")
    return len(imagens), len(labels)


def validar_pares_remotos(remote: str, caminho_versao: str,
                          drive_folder_id: str) -> int:
    """Confere imagens e labels na origem antes de qualquer download."""
    imagens = itens_remotos(remote, f"{caminho_versao}/images", drive_folder_id)
    labels = itens_remotos(remote, f"{caminho_versao}/labels", drive_folder_id)
    nomes_imagens = [Path(n).stem for n, pasta in imagens.items()
                     if not pasta and Path(n).suffix.lower() in EXTENSOES_IMAGEM]
    nomes_labels = [Path(n).stem for n, pasta in labels.items()
                    if not pasta and Path(n).suffix.lower() == ".txt"]
    if len(set(nomes_imagens)) != len(nomes_imagens):
        raise DownloadError(f"Nomes de imagem duplicados em {caminho_versao}")
    if len(set(nomes_labels)) != len(nomes_labels):
        raise DownloadError(f"Nomes de label duplicados em {caminho_versao}")
    sem_label = sorted(set(nomes_imagens) - set(nomes_labels))
    sem_imagem = sorted(set(nomes_labels) - set(nomes_imagens))
    if not nomes_imagens or sem_label or sem_imagem:
        raise DownloadError(
            f"Dataset remoto inconsistente em {caminho_versao}: "
            f"imagens={len(nomes_imagens)} labels={len(nomes_labels)} "
            f"sem_label={sem_label[:3]} sem_imagem={sem_imagem[:3]}"
        )
    return len(nomes_imagens)


def preparar_fontes(remote: str, drive_folder_id: str, versao: str,
                    fontes: tuple[str, ...]) -> list[tuple[str, str, int]]:
    """Valida todas as fontes do Drive antes de copiar qualquer arquivo."""
    base = itens_remotos(remote, "", drive_folder_id)
    nome_roboflow = encontrar_pasta(base, "roboflow", "raiz do Drive")
    origem_remota = itens_remotos(remote, nome_roboflow, drive_folder_id)
    plano = []
    for fonte in fontes:
        nome_fonte = encontrar_pasta(origem_remota, fonte, f"{nome_roboflow}/")
        caminho_fonte = f"{nome_roboflow}/{nome_fonte}"
        versoes = itens_remotos(remote, caminho_fonte, drive_folder_id)
        nome_versao = encontrar_pasta(versoes, versao, caminho_fonte)
        caminho_versao = f"{caminho_fonte}/{nome_versao}"
        arquivos = itens_remotos(remote, caminho_versao, drive_folder_id)
        for pasta in ("images", "labels"):
            encontrar_pasta(arquivos, pasta, caminho_versao)
        if arquivos.get("data.yaml") is not False:
            raise DownloadError(f"Falta data.yaml em {caminho_versao}")
        esperado = validar_pares_remotos(remote, caminho_versao, drive_folder_id)
        plano.append((fonte, caminho_versao, esperado))
    return plano


def baixar(raiz: Path, versao: str, fontes: tuple[str, ...], remote: str,
           drive_folder_id: str, dry_run: bool) -> None:
    if not shutil.which("rclone"):
        raise DownloadError("rclone não encontrado no PATH. Instale-o e execute `rclone config`.")
    remotes = executar(["rclone", "listremotes"]).splitlines()
    if f"{remote}:" not in remotes:
        raise DownloadError(f"Remote '{remote}' não configurado. Execute `rclone config`.")

    plano = preparar_fontes(remote, drive_folder_id, versao, fontes)
    for fonte, _, _ in plano:
        validar_entrada_local(raiz / "data" / "roboflow" / fonte / versao, raiz)

    print(f"Fonte: Google Drive / data/ (ID {drive_folder_id})")
    print(f"Versão: {versao} | Fontes: {', '.join(fontes)}")
    for fonte, remoto, esperado in plano:
        destino = raiz / "data" / "roboflow" / fonte / versao
        print(f"\n{remote}:{remoto}/  ->  {destino} ({esperado} imagens esperadas)")
        if dry_run:
            continue
        destino.mkdir(parents=True, exist_ok=True)
        for subpasta in ("images", "labels"):
            executar([
                "rclone", "copy", f"{remote}:{remoto}/{subpasta}",
                str(destino / subpasta), "--ignore-existing",
                "--drive-root-folder-id", drive_folder_id,
            ])
        executar([
            "rclone", "copyto", f"{remote}:{remoto}/data.yaml",
            str(destino / "data.yaml"), "--ignore-existing",
            "--drive-root-folder-id", drive_folder_id,
        ])
        # Detecta tanto transferências incompletas quanto arquivos existentes divergentes.
        executar([
            "rclone", "check", f"{remote}:{remoto}", str(destino),
            "--one-way", "--drive-root-folder-id", drive_folder_id,
        ])
        quantidade_imagens, quantidade_labels = conferir_pares(destino)
        if quantidade_imagens != esperado or quantidade_labels != esperado:
            raise DownloadError(
                f"Quantidade local divergente em {destino}: "
                f"esperado {esperado}, encontrado {quantidade_imagens} imagens e {quantidade_labels} labels"
            )
        print(f"  OK: {quantidade_imagens} imagens, {quantidade_labels} labels, data.yaml verificado")

    if dry_run:
        print("\nPrévia concluída. Nenhum arquivo local foi criado ou alterado.")
    else:
        print("\nDownload concluído. Nenhum arquivo existente foi sobrescrito ou excluído.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=VERSOES, default="v1")
    parser.add_argument("--source", choices=("all", *ORIGENS), default="all")
    parser.add_argument("--remote", default="boccia_drive", help="Nome do remote do rclone")
    parser.add_argument("--drive-folder-id", default=DRIVE_DATA_FOLDER_ID,
                        help="ID da pasta data/ no Drive, não é uma credencial")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Raiz local do repositório")
    parser.add_argument("--dry-run", action="store_true", help="Verifica a estrutura sem copiar")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.remote):
        parser.error("--remote deve conter apenas letras, números, hífen e sublinhado")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.drive_folder_id):
        parser.error("--drive-folder-id inválido")
    fontes = ORIGENS if args.source == "all" else (args.source,)
    raiz = args.root.expanduser().resolve()
    try:
        baixar(raiz, args.version, fontes, args.remote,
               args.drive_folder_id, args.dry_run)
    except (DownloadError, OSError) as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
