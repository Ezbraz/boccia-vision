# Scripts executáveis

- `train_yolo.py`: treina o detector; modelo padrão `yolo26s.pt`. O argumento `--data` ainda tem um caminho legado e deve ser informado explicitamente.
- `predict.py`: realiza inferência em uma imagem.
- `reorganizar_dataset.py`: script **legado** para divisão de uma única exportação em `data/roboflow/v1/`. Não atende ainda aos conjuntos Real, Sintético e Híbrido.
- `setup_project.py`: inicializa os diretórios locais de `v1` e `v2` sem substituir arquivos.
- `download_datasets.py`: baixa imagens, labels e YAMLs pelo Google Drive com rclone, sem sobrescrever dados locais; configurar o remote antes de usar.

## Scripts planejados

- `extrair_frames.py`: extração de frames de vídeo.
