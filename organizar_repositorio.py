#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Organiza o projeto Boccia Vision sem apagar nem sobrescrever arquivos.

Uso (executar a partir da raiz do projeto ou informar --raiz):
    python organizar_repositorio.py             # SOMENTE pre-visualiza
    python organizar_repositorio.py --aplicar   # move e organiza
    python organizar_repositorio.py --aplicar --copiar  # copia os originais

Somente a opção --aplicar modifica arquivos. A pasta .git e o README.md
existente nunca são alterados. Em caso de colisão com arquivos existentes,
a operação é interrompida antes de qualquer movimentação.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

IMAGEM_EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
PASTAS_PROJETO = (
    "data/roboflow/v1",
    "data/processed",
    "data/videos",
    "scripts",
    "src/boccia_vision/detection",
    "src/boccia_vision/homography",
    "src/boccia_vision/tracking",
    "src/boccia_vision/measurement",
    "experiments",
    "runs",
    "tests",
    "docs",
)

GITIGNORE_BLOCO = """# === Boccia Vision: gerado pelo organizador ===
# Arquivos do ambiente de desenvolvimento
__pycache__/
*.py[cod]
.venv/
venv/
.idea/
.pytest_cache/

# Dados, vídeos e artefatos grandes
/data/roboflow/
/data/processed/
/data/videos/
/runs/
*.pt
*.onnx

# Temporários
*.log
.DS_Store
# === fim do bloco Boccia Vision ===
"""

TREINO_PY = '''"""Treina o YOLO em um dataset no formato Ultralytics.

Exemplos:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --data data/processed/video_seed42/data.yaml
"""
from pathlib import Path
import argparse
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=RAIZ / "data/roboflow/v1/data.yaml")
    parser.add_argument("--modelo", default="yolo11n.pt")
    parser.add_argument("--epocas", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--nome", default="treino_01")
    args = parser.parse_args()

    dados = args.data if args.data.is_absolute() else (RAIZ / args.data)
    if not dados.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {dados}")
    modelo = YOLO(args.modelo)
    modelo.train(
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
'''

PREDICT_PY = '''"""Realiza inferência com pesos treinados localmente.

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
'''

ARQUIVOS_NOVOS = {
    "scripts/train_yolo.py": TREINO_PY,
    "scripts/predict.py": PREDICT_PY,
    "scripts/README.md": (
        "# Scripts executáveis\n\n"
        "- `train_yolo.py`: treinar o detector.\n"
        "- `predict.py`: inferência em uma imagem.\n"
        "- `reorganizar_dataset.py`: divisão por vídeo ou por imagem (se você adicionou o script anterior).\n"
        "- `extrair_frames.py`: implementar quando iniciar a extração dos vídeos.\n"
    ),
    "experiments/README.md": (
        "# Experimentos\n\n"
        "Versione aqui configurações, métricas, sementes e manifestos de divisão.\n"
        "Não coloque imagens ou modelos grandes nesta pasta.\n"
    ),
    "docs/README.md": "# Documentação\n\nDescreva metodologia, coleta de dados e avaliação do TCC.\n",
    "tests/README.md": "# Testes\n\nTestes automatizados de detecção, homografia e medições.\n",
    "requirements.txt": "ultralytics\nopencv-python\n",
    "src/boccia_vision/__init__.py": '"""Código principal do projeto Boccia Vision."""\n',
    "src/boccia_vision/detection/__init__.py": '"""Detecção de objetos."""\n',
    "src/boccia_vision/homography/__init__.py": '"""Homografia e mudança de perspectiva."""\n',
    "src/boccia_vision/tracking/__init__.py": '"""Rastreamento de bolas entre frames."""\n',
    "src/boccia_vision/measurement/__init__.py": '"""Calibração de escala e cálculo de distâncias."""\n',
    "src/boccia_vision/pipeline.py": (
        '"""Ponto futuro de integração: detecção -> homografia -> tracking -> distâncias."""\n'
    ),
}


def validar_labels(raiz: Path) -> dict[str, int]:
    """Verifica cada anotação YOLO, sem modificar os dados."""
    contagens = {}
    for split in SPLITS:
        pasta = raiz / split
        if not pasta.exists():
            continue
        imagens = pasta / "images"
        labels = pasta / "labels"
        if not imagens.is_dir() or not labels.is_dir():
            raise ValueError(f"Estrutura incorreta em {pasta}: esperado images/ e labels/")
        arquivos_imagem = [p for p in imagens.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGEM_EXTENSOES]
        sem_label = [p.name for p in arquivos_imagem
                     if not (labels / f"{p.stem}.txt").is_file()]
        if sem_label:
            raise ValueError(
                f"Existem {len(sem_label)} imagens sem labels em {split}. "
                f"Exemplo: {sem_label[0]}"
            )
        contagens[split] = len(arquivos_imagem)
    return contagens


def preparar_movimentos(raiz: Path) -> list[tuple[Path, Path]]:
    destinos = [(raiz / split, raiz / "data" / "roboflow" / "v1" / split)
                for split in SPLITS]
    destinos += [
        (raiz / "data.yaml", raiz / "data/roboflow/v1/data.yaml"),
        (raiz / "README.roboflow.txt", raiz / "data/roboflow/v1/README.roboflow.txt"),
        (raiz / "reorganizar_dataset.py", raiz / "scripts/reorganizar_dataset.py"),
    ]
    conflitos = [(src, dst) for src, dst in destinos if src.exists() and dst.exists()]
    if conflitos:
        texto = "\n".join(f"  {a.relative_to(raiz)} -> {b.relative_to(raiz)}" for a, b in conflitos)
        raise FileExistsError(
            "Há arquivos tanto na origem quanto no destino; "
            "nada será sobrescrito. Resolva os conflitos:\n" + texto
        )
    return [(src, dst) for src, dst in destinos if src.exists()]


def corrigir_yaml(caminho: Path) -> bool:
    """Adapta caminhos à nova pasta sem perder metadados Roboflow."""
    if not caminho.is_file():
        return False
    original = caminho.read_text(encoding="utf-8-sig")
    conteudo = original
    # Não fixar paths absolutos, pois o YAML será usado em computadores diferentes.
    conteudo = re.sub(r"(?m)^path\s*:\s*[^\n]*\n", "", conteudo)
    substituicoes = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    }
    for chave, valor in substituicoes.items():
        linha = rf"(?m)^{chave}\s*:[^\n]*"
        if re.search(linha, conteudo):
            conteudo = re.sub(linha, f"{chave}: {valor}", conteudo)
        else:
            conteudo = f"{chave}: {valor}\n" + conteudo
    if conteudo != original:
        caminho.write_text(conteudo, encoding="utf-8")
        return True
    return False


def adaptar_script_antigo(caminho: Path) -> str:
    """Atualiza APENAS os padrões conhecidos do divisor já fornecido ao projeto."""
    if not caminho.exists():
        return "não encontrado; pode ser adicionado posteriormente"
    texto = caminho.read_text(encoding="utf-8")
    if 'raiz = Path(__file__).resolve().parents[1] / "data" / "roboflow" / "v1"' in texto:
        return "já adaptado"
    antigo_raiz = 'raiz = Path(__file__).resolve().parent'
    antigo_saida = 'saida = (args.saida or raiz / f"dataset_dividido_{args.modo}_seed{args.seed}").resolve()'
    if antigo_raiz not in texto or antigo_saida not in texto:
        return "formato diferente; ajuste manualmente a entrada e a saída"
    novo = texto.replace(
        antigo_raiz,
        'raiz = Path(__file__).resolve().parents[1] / "data" / "roboflow" / "v1"',
    ).replace(
        antigo_saida,
        'saida = (args.saida or raiz.parents[1] / "processed" / f"{args.modo}_seed{args.seed}").resolve()',
    )
    # Os datasets produzidos devem ser portáveis, inclusive após mover a pasta.
    linha_absoluta = '        f"path: {json.dumps(saida.as_posix(), ensure_ascii=False)}\\n"\n'
    novo = novo.replace(linha_absoluta, "")
    backup = caminho.with_name("reorganizar_dataset.antes_da_migracao.py")
    if backup.exists():
        raise FileExistsError(f"Backup já existe, ajuste manualmente: {backup}")
    backup.write_text(texto, encoding="utf-8")
    caminho.write_text(novo, encoding="utf-8")
    return "adaptado (backup preservado em scripts/reorganizar_dataset.antes_da_migracao.py)"


def configurar_gitignore(raiz: Path) -> bool:
    destino = raiz / ".gitignore"
    atual = destino.read_text(encoding="utf-8") if destino.exists() else ""
    if "# === Boccia Vision: gerado pelo organizador ===" in atual:
        return False
    separador = "" if not atual else "\n" if atual.endswith("\n") else "\n\n"
    destino.write_text(atual + separador + GITIGNORE_BLOCO, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", type=Path, default=Path(__file__).resolve().parent,
                        help="Raiz do repositório (padrão: pasta do script)")
    parser.add_argument("--aplicar", action="store_true",
                        help="Executa a migração; sem este argumento é só prévia")
    parser.add_argument("--copiar", action="store_true",
                        help="Copia os arquivos em vez de mover (apenas com --aplicar)")
    args = parser.parse_args()
    if args.copiar and not args.aplicar:
        parser.error("Use --copiar junto com --aplicar")

    raiz = args.raiz.expanduser().resolve()
    if not raiz.is_dir():
        parser.error(f"Pasta do projeto inexistente: {raiz}")
    if not any((raiz / p).exists() for p in ("README.md", ".git", "train", "data/roboflow/v1")):
        parser.error(f"A pasta não parece ser a raiz de boccia-vision: {raiz}")

    movimentos = preparar_movimentos(raiz)
    quantidades = validar_labels(raiz)
    print(f"Projeto: {raiz}")
    if quantidades:
        print("Anotações conferidas: " + ", ".join(f"{k}={v}" for k, v in quantidades.items())
              + f" (total={sum(quantidades.values())})")
    print("\nMovimentações planejadas:" if movimentos else "\nDados já organizados ou não encontrados.")
    for origem, destino in movimentos:
        print(f"  {origem.relative_to(raiz)} -> {destino.relative_to(raiz)}")
    print("\nEstrutura a criar: " + ", ".join(PASTAS_PROJETO))
    print("Arquivos já existentes serão preservados; README.md e .git não serão tocados.")
    if not args.aplicar:
        print("\nPRÉVIA APENAS. Para executar: python organizar_repositorio.py --aplicar")
        return

    # Executar as movimentações antes de criar os arquivos auxiliares.
    # Se uma cópia/movimentação falhar, devolvemos as entradas já movidas.
    feitos: list[tuple[Path, Path]] = []
    try:
        for origem, destino in movimentos:
            destino.parent.mkdir(parents=True, exist_ok=True)
            if args.copiar:
                if origem.is_dir():
                    shutil.copytree(origem, destino)
                else:
                    shutil.copy2(origem, destino)
            else:
                shutil.move(str(origem), str(destino))
            feitos.append((origem, destino))
    except Exception:
        if not args.copiar:
            for origem, destino in reversed(feitos):
                shutil.move(str(destino), str(origem))
        raise

    for pasta in PASTAS_PROJETO:
        (raiz / pasta).mkdir(parents=True, exist_ok=True)

    yaml = raiz / "data/roboflow/v1/data.yaml"
    yaml_atualizado = corrigir_yaml(yaml)

    novos = 0
    for nome, conteudo in ARQUIVOS_NOVOS.items():
        arquivo = raiz / nome
        if arquivo.exists():
            continue
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        arquivo.write_text(conteudo, encoding="utf-8")
        novos += 1

    situacao_divisor = adaptar_script_antigo(raiz / "scripts/reorganizar_dataset.py")
    alterou_gitignore = configurar_gitignore(raiz)

    print("\nORGANIZAÇÃO CONCLUÍDA")
    print(f"  Arquivos/pastas movidos ou copiados: {len(feitos)}")
    print(f"  Arquivos auxiliares novos: {novos}")
    print(f"  data.yaml atualizado: {'sim' if yaml_atualizado else 'não'}")
    print(f"  Divisor de dataset: {situacao_divisor}")
    print(f"  .gitignore atualizado: {'sim' if alterou_gitignore else 'já configurado'}")
    print("  README.md e pasta .git preservados.")
    print("\nAtenção: .gitignore NÃO remove do histórico arquivos que já estavam no Git.")
    print("Revise o resultado com 'git status' antes de fazer commit.")
    print("Instale as dependências com: python -m pip install -r requirements.txt")


if __name__ == "__main__":
    main()
