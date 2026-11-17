"""Reune um dataset YOLO exportado do Roboflow e refaz o split.

Coloque este arquivo na raiz de boccia-vision (junto de train/, valid/, test/).
Exemplos:
    python reorganizar_dataset.py
    python reorganizar_dataset.py --modo aleatorio --seed 42
    python reorganizar_dataset.py --modo video --seed 2026 --train 0.7 --val 0.15 --test 0.15

Modo 'video': nenhum video-fonte fica em dois subconjuntos.
Modo 'aleatorio': imagens diferentes do mesmo video podem ir para subconjuntos
    diferentes; versoes aumentadas de um mesmo frame permanecem juntas.
"""

import argparse
import itertools
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

CONJUNTOS = ("train", "valid", "test")
EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASSES = ["blue_ball", "red_ball", "white_ball"]  # Conforme data.yaml fornecido


def identificar_grupo(imagem: Path, modo: str) -> str:
    # Remove o hash que o Roboflow acrescenta aos nomes exportados.
    frame = re.sub(r"\.rf\.[0-9a-f]+$", "", imagem.stem, flags=re.I)
    if modo == "video":
        # Exemplos: court1_mp4-0001_jpg, hk_court1_trecho_mp4-0089_jpg
        match = re.match(r"^(.*?)_mp4-\d+", frame, flags=re.I)
        if match:
            return match.group(1)
        # Para nomes diferentes do padrão, cada frame é um grupo isolado.
    return frame


def coletar_pares(raiz: Path, modo: str):
    pares = []
    for origem in CONJUNTOS:
        pasta_imagens = raiz / origem / "images"
        pasta_labels = raiz / origem / "labels"
        if not pasta_imagens.is_dir() or not pasta_labels.is_dir():
            raise FileNotFoundError(f"Pastas ausentes: {pasta_imagens} / {pasta_labels}")
        for imagem in sorted(pasta_imagens.iterdir()):
            if not imagem.is_file() or imagem.suffix.lower() not in EXTENSOES:
                continue
            label = pasta_labels / (imagem.stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Anotacao ausente para {imagem}: {label}")
            pares.append({"imagem": imagem, "label": label,
                          "origem": origem, "grupo": identificar_grupo(imagem, modo)})
    if not pares:
        raise ValueError("Nenhuma imagem encontrada nas pastas de origem.")
    return pares


def distribuir(pares, modo: str, seed: int, proporcoes: dict):
    rng = random.Random(seed)
    grupos = defaultdict(list)
    for par in pares:
        grupos[par["grupo"]].append(par)
    chaves = sorted(grupos)
    rng.shuffle(chaves)
    resultado = {c: [] for c in CONJUNTOS}

    if modo == "aleatorio":
        # Em datasets sem augmentation, 1 grupo equivale a 1 imagem.
        n = len(chaves)
        limites = (round(n * proporcoes["train"]),
                   round(n * (proporcoes["train"] + proporcoes["valid"])))
        for idx, grupo in enumerate(chaves):
            destino = "train" if idx < limites[0] else ("valid" if idx < limites[1] else "test")
            resultado[destino].extend(grupos[grupo])
    else:
        if len(chaves) < 3:
            raise ValueError("Divisao por video exige ao menos 3 videos/grupos de origem.")
        # Encontrar a atribuicao mais proxima das proporcoes, sem dividir videos.
        # Para muitos videos, utilizar uma alternativa gulosa escalavel.
        tamanhos = [len(grupos[c]) for c in chaves]
        metas = {c: len(pares) * proporcoes[c] for c in CONJUNTOS}
        if len(chaves) <= 10:
            melhor = None
            for distribuicao in itertools.product(CONJUNTOS, repeat=len(chaves)):
                if len(set(distribuicao)) != 3:
                    continue
                contagens = {c: 0 for c in CONJUNTOS}
                for destino, tamanho in zip(distribuicao, tamanhos):
                    contagens[destino] += tamanho
                erro = sum((contagens[c] - metas[c]) ** 2 for c in CONJUNTOS)
                if melhor is None or erro < melhor[0]:
                    melhor = (erro, distribuicao)
            destinos = melhor[1]
        else:
            ordem = sorted(range(len(chaves)), key=lambda i: -tamanhos[i])
            contagens = {c: 0 for c in CONJUNTOS}
            destinos = [None] * len(chaves)
            for i, indice in enumerate(ordem):
                vazios = [c for c in CONJUNTOS if contagens[c] == 0]
                faltam = len(ordem) - i
                candidatos = vazios if faltam == len(vazios) else CONJUNTOS
                escolhido = min(candidatos, key=lambda c: sum(
                    (contagens[z] + (tamanhos[indice] if z == c else 0) - metas[z]) ** 2
                    for z in CONJUNTOS))
                destinos[indice] = escolhido
                contagens[escolhido] += tamanhos[indice]
        for grupo, destino in zip(chaves, destinos):
            resultado[destino].extend(grupos[grupo])

    for destino in CONJUNTOS:
        resultado[destino].sort(key=lambda p: (p["imagem"].name, p["origem"]))
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Reorganizar um dataset YOLO com seed registrada")
    parser.add_argument("--modo", choices=("video", "aleatorio"), default="video")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--saida", type=Path, default=None)
    args = parser.parse_args()

    proporcoes = {"train": args.train, "valid": args.val, "test": args.test}
    if any(v <= 0 for v in proporcoes.values()) or abs(sum(proporcoes.values()) - 1) > 1e-9:
        parser.error("As proporcoes devem ser positivas e somar 1 (ex.: 0.70, 0.15, 0.15).")

    raiz = Path(__file__).resolve().parents[1] / "data" / "roboflow" / "v1"
    saida = (args.saida or raiz.parents[1] / "processed" / f"{args.modo}_seed{args.seed}").resolve()
    if saida.exists():
        raise FileExistsError(f"A pasta {saida} ja existe. Nao sera sobrescrita; escolha --saida.")

    pares = coletar_pares(raiz, args.modo)
    divisao = distribuir(pares, args.modo, args.seed, proporcoes)
    manifesto = {
        "seed": args.seed,
        "modo": args.modo,
        "proporcoes_solicitadas": proporcoes,
        "classes": CLASSES,
        "total_imagens": len(pares),
        "quantidade_por_conjunto": {c: len(divisao[c]) for c in CONJUNTOS},
        "arquivos": {c: [] for c in CONJUNTOS},
    }
    saida.mkdir(parents=True)
    usados = set()

    for destino in CONJUNTOS:
        imagens_saida = saida / destino / "images"
        labels_saida = saida / destino / "labels"
        imagens_saida.mkdir(parents=True)
        labels_saida.mkdir(parents=True)
        for par in divisao[destino]:
            nome = par["imagem"].name
            # Caso nomes se repitam em pastas de origem diferentes, desambiguar.
            if nome.lower() in usados:
                nome = f'{par["origem"]}__{nome}'
            if nome.lower() in usados:
                raise ValueError(f"Nome duplicado mesmo apos desambiguacao: {nome}")
            usados.add(nome.lower())
            shutil.copy2(par["imagem"], imagens_saida / nome)
            shutil.copy2(par["label"], labels_saida / (Path(nome).stem + ".txt"))
            manifesto["arquivos"][destino].append({
                "imagem": nome,
                "grupo": par["grupo"],
                "origem_imagem": par["imagem"].relative_to(raiz).as_posix(),
                "origem_label": par["label"].relative_to(raiz).as_posix(),
            })

    # Caminho absoluto para que funcione ao chamar o treino em outra pasta do PyCharm.
    yaml_texto = (
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"names: {json.dumps(CLASSES, ensure_ascii=False)}\n"
    )
    (saida / "data.yaml").write_text(yaml_texto, encoding="utf-8")
    (saida / "seed.txt").write_text(str(args.seed) + "\n", encoding="utf-8")
    (saida / "split_manifest.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Dataset salvo em: {saida}")
    print(f"Semente: {args.seed} | Modo: {args.modo}")
    for conjunto in CONJUNTOS:
        numero = len(divisao[conjunto])
        print(f"  {conjunto:5s}: {numero:3d} imagens ({numero / len(pares):.1%})")
    print("Seed e divisao registradas em seed.txt e split_manifest.json")


if __name__ == "__main__":
    main()
