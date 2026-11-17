"""Realiza inferência com pesos treinados localmente.

Exemplo:
    python scripts/predict.py --imagem caminho/para/foto.jpg
"""
from pathlib import Path
import argparse
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagem", type=Path, required=True)
    parser.add_argument(
        "--pesos", type=Path,
        default=RAIZ / "runs/detect/treino_01/weights/best.pt",
    )
    args = parser.parse_args()
    pesos = args.pesos if args.pesos.is_absolute() else RAIZ / args.pesos
    imagem = args.imagem if args.imagem.is_absolute() else RAIZ / args.imagem
    if not pesos.is_file() or not imagem.is_file():
        raise FileNotFoundError(f"Confira os caminhos: {pesos} e {imagem}")
    modelo = YOLO(str(pesos))
    modelo.predict(source=str(imagem), save=True, project=str(RAIZ / "runs" / "predict"))


if __name__ == "__main__":
    main()
