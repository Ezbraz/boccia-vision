"""Treina o YOLO em um dataset no formato Ultralytics.

Exemplos:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --data data/processed/video_seed17111/data.yaml
"""
from pathlib import Path
import argparse
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=RAIZ / "data/roboflow/v1/data.yaml")
    parser.add_argument("--modelo", default="yolo26s.pt")
    parser.add_argument("--epocas", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--nome", default="treino_01")
    parser.add_argument("--seed", type=int, default=17111)
    args = parser.parse_args()

    dados = args.data if args.data.is_absolute() else (RAIZ / args.data)
    if not dados.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {dados}")
    modelo = YOLO(args.modelo)
    modelo.train(
        seed=args.seed,
        deterministic=True,
        data=str(dados.resolve()),
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,  # mais simples no Windows
        project=str(RAIZ / "runs" / "detect"),
        name=args.nome,
    )


if __name__ == "__main__":
    main()
