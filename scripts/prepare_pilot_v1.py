#!/usr/bin/env python3
"""Prepara o piloto v1 do Boccia Vision, sem alterar exportacoes Roboflow.

Uso (raiz do projeto):
    python scripts/prepare_pilot_v1.py --dry-run
    python scripts/prepare_pilot_v1.py
    python scripts/prepare_pilot_v1.py --class-mode generic --seed 42

Protocolo piloto: dois folds por partida real, validacao temporal interna,
conjunto de teste externo por partida, tres composicoes e orcamento igual.
A validacao interna nao e independente por partida. Nunca ajustar parametros
usando os conjuntos de teste externos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml  # type: ignore
except ImportError as exc:
    raise SystemExit("PyYAML ausente. Instale com: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("youtube", "local", "synthetic")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CANONICAL_CLASSES = ("blue_ball", "red_ball", "white_ball")
# Contrato EXPLICITO da v1: p1 e p2 sao trechos da mesma partida.
GROUPS = {
    "youtube": {"hk_court1_trecho": "real_youtube_match_01"},
    "local": {"p1": "real_local_match_01", "p2": "real_local_match_01"},
    "synthetic": {
        "BocciaVision_01": "synthetic_scene_01",
        "BocciaVision_03": "synthetic_scene_03",
    },
}
FOLDS = (
    ("rodada_a", "real_local_match_01", "real_youtube_match_01"),
    ("rodada_b", "real_youtube_match_01", "real_local_match_01"),
)
SCENARIOS = ("real", "synthetic", "hybrid")
NAME_RE = re.compile(r"^(?P<clip>.+?)_mp4-(?P<frame>\d+)(?:_\w+)?(?:\.rf\.[0-9a-f]+)?$", re.I)


class PreparationError(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    source: str
    image: Path
    label: Path
    clip: str
    frame: int
    group: str
    image_sha256: str
    label_sha256: str
    annotation_count: int
    remap: dict[int, int]

    @property
    def key(self) -> str:
        return f"{self.source}/{self.image.name}"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_class_mapping(yaml_path: Path) -> dict[int, int]:
    try:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreparationError(f"YAML invalido: {yaml_path}: {exc}") from exc
    if not isinstance(config, dict) or "names" not in config:
        raise PreparationError(f"YAML sem 'names': {yaml_path}")
    names = config["names"]
    if isinstance(names, list):
        indexed = dict(enumerate(names))
    elif isinstance(names, dict):
        try:
            indexed = {int(k): v for k, v in names.items()}
        except (ValueError, TypeError) as exc:
            raise PreparationError(f"IDs de classe invalidos: {yaml_path}") from exc
    else:
        raise PreparationError(f"'names' nao e lista ou mapa: {yaml_path}")
    if sorted(indexed) != [0, 1, 2]:
        raise PreparationError(f"Esperadas 3 classes com IDs 0, 1, 2 em {yaml_path}: {indexed}")
    normalized = {i: str(name).strip().casefold() for i, name in indexed.items()}
    if set(normalized.values()) != set(CANONICAL_CLASSES):
        raise PreparationError(f"Classes inesperadas em {yaml_path}: {indexed}")
    if "nc" in config and int(config["nc"]) != 3:
        raise PreparationError(f"nc != 3 em {yaml_path}")
    return {i: CANONICAL_CLASSES.index(name) for i, name in normalized.items()}


def parse_annotations(path: Path, remap: dict[int, int]) -> tuple[int, tuple[str, ...]]:
    fixed = []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        values = line.split()
        if len(values) != 5:
            raise PreparationError(f"Label YOLO invalido ({path}, linha {n}): esperado 5 valores")
        try:
            raw_id = int(values[0])
            box = [float(v) for v in values[1:]]
        except ValueError as exc:
            raise PreparationError(f"Numero invalido em {path}:{n}") from exc
        if raw_id not in remap or any(not math.isfinite(v) for v in box):
            raise PreparationError(f"Classe ou coordenada invalida em {path}:{n}")
        x, y, w, h = box
        eps = 1e-3
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and
                -eps <= x - w / 2 <= 1 + eps and -eps <= x + w / 2 <= 1 + eps and
                -eps <= y - h / 2 <= 1 + eps and -eps <= y + h / 2 <= 1 + eps):
            raise PreparationError(f"Bounding box fora da imagem em {path}:{n}: {box}")
        fixed.append(f"{remap[raw_id]} {' '.join(values[1:])}")
    return len(fixed), tuple(fixed)


def load_samples(root: Path, version: str) -> list[Sample]:
    samples: list[Sample] = []
    for source in SOURCES:
        base = root / "data" / "roboflow" / source / version
        imgs, labels = base / "images", base / "labels"
        if not imgs.is_dir() or not labels.is_dir():
            raise PreparationError(f"Pastas images/labels ausentes em {base}")
        remap = read_class_mapping(base / "data.yaml")
        image_files = sorted((p for p in imgs.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.name)
        label_files = sorted((p for p in labels.iterdir() if p.is_file() and p.suffix.lower() == ".txt"), key=lambda p: p.name)
        if not image_files:
            raise PreparationError(f"Nenhuma imagem em {imgs}")
        if len({p.stem.casefold() for p in image_files}) != len(image_files):
            raise PreparationError(f"Imagens com nomes-base duplicados em {imgs}")
        if len({p.stem.casefold() for p in label_files}) != len(label_files):
            raise PreparationError(f"Labels com nomes-base duplicados em {labels}")
        expected = {p.stem.casefold() for p in image_files}
        actual = {p.stem.casefold() for p in label_files}
        if expected != actual:
            raise PreparationError(f"Pares divergentes em {base}: sem label {sorted(expected-actual)[:3]}, sem imagem {sorted(actual-expected)[:3]}")
        label_by_stem = {p.stem.casefold(): p for p in label_files}
        seen_raw = set()
        for image in image_files:
            match = NAME_RE.fullmatch(image.stem)
            if match is None:
                raise PreparationError(f"Nome sem padrao de video: {image.name}")
            clip = match.group("clip")
            if clip not in GROUPS[source]:
                raise PreparationError(f"Grupo nao mapeado: origem={source}, clip={clip}; atualize GROUPS explicitamente")
            frame = int(match.group("frame"))
            # Variantes de um mesmo frame nao podem ser tratadas como exemplos independentes no piloto.
            raw_key = (clip, frame)
            if raw_key in seen_raw:
                raise PreparationError(f"Mais de uma versao do mesmo frame em {source}: {raw_key}")
            seen_raw.add(raw_key)
            label = label_by_stem[image.stem.casefold()]
            count, _ = parse_annotations(label, remap)
            samples.append(Sample(source, image, label, clip, frame, GROUPS[source][clip],
                                  digest(image), digest(label), count, remap))
    # Detecta duplicatas EXATAS entre grupos. Similares devem ser inspecionadas separadamente.
    hashes: dict[str, Sample] = {}
    for sample in samples:
        previous = hashes.get(sample.image_sha256)
        if previous is not None and previous.group != sample.group:
            raise PreparationError(f"Imagem identica em grupos diferentes: {previous.key} / {sample.key}")
        hashes[sample.image_sha256] = sample
    return samples


def seeded_pick(samples: Iterable[Sample], qty: int, seed: int, label: str) -> list[Sample]:
    candidates = sorted(samples, key=lambda s: s.key)
    if len(candidates) < qty:
        raise PreparationError(f"{label}: precisa de {qty} imagens, mas apenas {len(candidates)} disponiveis")
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "big"))
    return sorted(rng.sample(candidates, qty), key=lambda s: s.key)


def split_real_val(real_samples: list[Sample], fraction: float, gap: int) -> tuple[list[Sample], list[Sample], list[Sample]]:
    ordered = sorted(real_samples, key=lambda s: (s.clip, s.frame, s.key))
    # Valiacao temporal interna de cauda: para p1+p2, fica na cauda de p2.
    n_val = max(1, math.ceil(len(ordered) * fraction))
    validation = ordered[-n_val:]
    val_keys = {s.key for s in validation}
    # O gap e em numero de FRAMES presentes na exportacao, nao segundos de video.
    first_by_clip = defaultdict(lambda: float("inf"))
    for s in validation:
        first_by_clip[s.clip] = min(first_by_clip[s.clip], s.frame)
    train_eligible, purged = [], []
    for s in ordered:
        if s.key in val_keys:
            continue
        if s.clip in first_by_clip and first_by_clip[s.clip] - gap <= s.frame < first_by_clip[s.clip]:
            purged.append(s)
        else:
            train_eligible.append(s)
    return train_eligible, validation, purged


def choose_synthetic(synthetic: list[Sample], per_env: int, seed: int) -> list[Sample]:
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for s in synthetic:
        by_group[s.group].append(s)
    if set(by_group) != {"synthetic_scene_01", "synthetic_scene_03"}:
        raise PreparationError(f"Ambientes sinteticos inesperados: {sorted(by_group)}")
    out = []
    for group in sorted(by_group):
        out.extend(seeded_pick(by_group[group], per_env, seed, f"synth:{group}"))
    return sorted(out, key=lambda s: s.key)


def scenario_plan(samples: list[Sample], seed: int, budget: int, val_fraction: float, gap: int):
    if budget < 2 or budget % 4:
        raise PreparationError("--budget deve ser multiplo de 4 e no minimo 4 (hibrido 50/50 e equilibrio por ambiente)")
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group].append(sample)
    if set(groups) != {"real_youtube_match_01", "real_local_match_01", "synthetic_scene_01", "synthetic_scene_03"}:
        raise PreparationError(f"Grupos inesperados: {sorted(groups)}")
    synth_train = choose_synthetic([s for s in samples if s.source == "synthetic"], budget // 2, seed)
    hybrid_synth = choose_synthetic(synth_train, budget // 4, seed + 100_000)
    plans = {}
    for name, train_group, test_group in FOLDS:
        real_eligible, real_val, purged = split_real_val(groups[train_group], val_fraction, gap)
        real_train = seeded_pick(real_eligible, budget, seed, f"{name}:real")
        hybrid_real = seeded_pick(real_train, budget // 2, seed, f"{name}:hybrid-real")
        external_test = sorted(groups[test_group], key=lambda s: s.key)
        assert len({s.group for s in external_test}) == 1
        plans[name] = {
            "train_real_group": train_group,
            "test_real_group": test_group,
            "validation": real_val,
            "test": external_test,
            "purged": purged,
            "eligible_real": real_eligible,
            "scenarios": {
                "real": real_train,
                "synthetic": synth_train,
                "hybrid": sorted([*hybrid_real, *hybrid_synth], key=lambda s: s.key),
            },
        }
        for scenario, train in plans[name]["scenarios"].items():
            if len(train) != budget:
                raise PreparationError(f"Quantidade incorreta para {name}/{scenario}")
            train_keys = {s.key for s in train}
            val_keys = {s.key for s in real_val}
            test_keys = {s.key for s in external_test}
            if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
                raise PreparationError(f"Vazamento de arquivo entre splits: {name}/{scenario}")
            # Mesmo conteudo de imagem, ainda que exportado com nomes diferentes.
            train_hashes = {s.image_sha256 for s in train}
            val_hashes = {s.image_sha256 for s in real_val}
            test_hashes = {s.image_sha256 for s in external_test}
            if train_hashes & val_hashes or train_hashes & test_hashes or val_hashes & test_hashes:
                raise PreparationError(f"Imagem identica entre splits: {name}/{scenario}")
            if {s.group for s in train if s.source != "synthetic"} & {s.group for s in external_test}:
                raise PreparationError(f"Vazamento de partida real: {name}/{scenario}")
            # Garante que nenhum frame original foi reaproveitado por engano.
            if len(train_keys) != len(train):
                raise PreparationError(f"Duplicatas em treino: {name}/{scenario}")
    return plans


def manifest_record(s: Sample, root: Path) -> dict:
    return {
        "source": s.source,
        "group_id": s.group,
        "clip_id": s.clip,
        "frame": s.frame,
        "image": s.image.relative_to(root).as_posix(),
        "label": s.label.relative_to(root).as_posix(),
        "image_sha256": s.image_sha256,
        "label_sha256": s.label_sha256,
        "annotations": s.annotation_count,
    }


def write_item(s: Sample, part: str, dataset_dir: Path, class_mode: str) -> None:
    image_name = f"{s.source}__{s.image.name}"
    label_name = Path(image_name).stem + ".txt"
    images = dataset_dir / part / "images"
    labels = dataset_dir / part / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    dest_image, dest_label = images / image_name, labels / label_name
    if dest_image.exists() or dest_label.exists():
        raise PreparationError(f"Colisao no destino: {dest_image}")
    shutil.copy2(s.image, dest_image)
    _, mapped = parse_annotations(s.label, s.remap)
    if class_mode == "generic":
        mapped = tuple("0" + line[line.find(" "):] for line in mapped)
    dest_label.write_text("\n".join(mapped) + ("\n" if mapped else ""), encoding="utf-8")


def create_dataset(folder: Path, train_items: list[Sample], val: list[Sample],
                   test: list[Sample], class_mode: str) -> None:
    for split, selection in (("train", train_items), ("valid", val), ("test", test)):
        # Criar diretorios mesmo se estiverem vazios (nao devem estar neste piloto).
        (folder / split / "images").mkdir(parents=True, exist_ok=True)
        (folder / split / "labels").mkdir(parents=True, exist_ok=True)
        for sample in selection:
            write_item(sample, split, folder, class_mode)
    names = ["ball"] if class_mode == "generic" else list(CANONICAL_CLASSES)
    folder_absolute = folder.resolve().as_posix()
    yaml_text = (f"path: {json.dumps(folder_absolute, ensure_ascii=False)}\n"
                 "train: train/images\nval: valid/images\ntest: test/images\n"
                 f"nc: {len(names)}\nnames: {json.dumps(names, ensure_ascii=False)}\n")
    (folder / "data.yaml").write_text(yaml_text, encoding="utf-8")


def build_manifest(root: Path, version: str, seed: int, budget: int,
                   val_fraction: float, gap: int, class_mode: str, plans: dict, samples: list[Sample]) -> dict:
    folds = {}
    for fold_name, plan in plans.items():
        scenarios = {}
        for scenario, train in plan["scenarios"].items():
            scenarios[scenario] = {
                "counts": {"train": len(train), "valid": len(plan["validation"]), "test": len(plan["test"])},
                "train_by_source": dict(sorted(Counter(s.source for s in train).items())),
                "train": [manifest_record(s, root) for s in train],
                "valid": [manifest_record(s, root) for s in plan["validation"]],
                "test": [manifest_record(s, root) for s in plan["test"]],
            }
        folds[fold_name] = {
            "real_train_group": plan["train_real_group"],
            "external_test_group": plan["test_real_group"],
            "internal_validation_note": "Cauda temporal real + embargo em frames; mesma partida do treino, nao independente.",
            "real_train_eligible": len(plan["eligible_real"]),
            "real_train_excluded_gap": [manifest_record(s, root) for s in plan["purged"]],
            "scenarios": scenarios,
        }
    return {
        "protocol": "pilot_v1_two_real_match_holdouts",
        "dataset_version": version,
        "split_seed": seed,
        "train_budget_per_scenario": budget,
        "class_mode": class_mode,
        "validation_fraction_source_real": val_fraction,
        "temporal_embargo_frames": gap,
        "class_names": ["ball"] if class_mode == "generic" else list(CANONICAL_CLASSES),
        "total_images_input": len(samples),
        "input_by_source": dict(sorted(Counter(s.source for s in samples).items())),
        "independence_note": (
            "Somente os testes externos sao independentes por partida real. "
            "A validacao interna compartilha a partida do treino; sinteticos sao renders "
            "de dois ambientes diferentes, porem do mesmo processo de geracao. "
            "Use resultados das rodadas para analise exploratoria, nao para ajustar continuamente "
            "o modelo; reservar novas partidas para avaliacao final."
        ),
        "evaluation_note": (
            "Todos os cenarios usam a MESMA validacao real por rodada para selecao de checkpoint, "
            "inclusive o modelo treinado apenas com imagens sinteticas. "
            "Isto e selecao com dominio real, nao treinamento supervisionado com imagens reais."
        ),
        "augmentation_note": "Nenhum augmentation offline. Aplicar apenas no treino apos o split.",
        "folds": folds,
    }


def run(args: argparse.Namespace) -> Path:
    root = args.root.resolve()
    if args.version != "v1":
        raise PreparationError("Este preparador e EXCLUSIVO da v1. A v2 exigira novo plano por grupos.")
    if not (0 < args.val_fraction < 0.5):
        raise PreparationError("--val-fraction deve estar entre 0 e 0.5 (exclusivo)")
    if args.gap < 0:
        raise PreparationError("--gap deve ser >= 0")
    output = (args.output or root / "data" / "processed" / args.version /
              f"pilot_seed{args.seed}" / args.class_mode).resolve()
    input_root = (root / "data" / "roboflow").resolve()
    if output == input_root or input_root in output.parents or output in input_root.parents:
        raise PreparationError("O destino nao pode estar dentro da origem nem conter a origem")
    if output.exists():
        raise PreparationError(f"Destino ja existe, preservado sem sobrescrita: {output}")
    samples = load_samples(root, args.version)
    plans = scenario_plan(samples, args.seed, args.budget, args.val_fraction, args.gap)
    manifest = build_manifest(root, args.version, args.seed, args.budget,
                              args.val_fraction, args.gap, args.class_mode, plans, samples)
    print(f"Entrada: {input_root} | {len(samples)} imagens | seed={args.seed}")
    print(f"Saida: {output} | classes={args.class_mode} | orçamento={args.budget}")
    for fold, plan in plans.items():
        print(f"\n{fold}: real para treino={plan['train_real_group']} "
              f"| teste={plan['test_real_group']} "
              f"| valid={len(plan['validation'])} "
              f"| gap removido={len(plan['purged'])}")
        for scenario, selected in plan["scenarios"].items():
            counts = dict(sorted(Counter(s.source for s in selected).items()))
            print(f"  {scenario:9s} train={len(selected)} {counts} | "
                  f"valid={len(plan['validation'])} | test={len(plan['test'])}")
    if args.dry_run:
        print("\nDRY RUN: nenhuma pasta criada e nenhum arquivo modificado.")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".__boccia_pilot_", dir=output.parent))
    try:
        for fold, plan in plans.items():
            for scenario, selected in plan["scenarios"].items():
                # O YAML aponta para destino FINAL, nao a pasta temporaria.
                final_dir = output / fold / scenario
                tmp_dir = stage / fold / scenario
                create_dataset(tmp_dir, selected, plan["validation"], plan["test"], args.class_mode)
                yaml_file = tmp_dir / "data.yaml"
                yaml_file.write_text(yaml_file.read_text(encoding="utf-8").replace(
                    f"path: {json.dumps(tmp_dir.resolve().as_posix(), ensure_ascii=False)}",
                    f"path: {json.dumps(final_dir.resolve().as_posix(), ensure_ascii=False)}"), encoding="utf-8")
        (stage / "split_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise PreparationError(f"Destino surgiu durante a execucao: {output}")
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"\nConcluido: {output}")
    print(f"Manifesto: {output / 'split_manifest.json'}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Raiz do repositorio")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--class-mode", choices=("color", "generic"), default="color")
    parser.add_argument("--budget", type=int, default=60, help="Imagens de treino por cenario; multiplo de 4")
    parser.add_argument("--seed", type=int, default=42, help="Semente do split (fixa entre treinos)")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--gap", type=int, default=5, help="Distancia em indices de frame antes do bloco de validacao")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (PreparationError, OSError, UnicodeError) as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
