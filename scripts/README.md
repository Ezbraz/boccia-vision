# Scripts executáveis

- `train_yolo.py`: treina o detector; modelo padrão `yolo26s.pt`. O argumento `--data` ainda tem um caminho legado e deve ser informado explicitamente.
- `predict.py`: realiza inferência em uma imagem.
- `reorganizar_dataset.py`: script **legado** para divisão de uma única exportação em `data/roboflow/v1/`. Não atende ainda aos conjuntos Real, Sintético e Híbrido.
- `reset_primeiro_teste.py`: auxilia a remover artefatos locais de um teste antigo, com prévia e confirmação.

## Scripts planejados (ainda ausentes na revisão)

- `setup_project.py`: inicialização segura da estrutura local.
- `download_datasets.py`: sincronização das fontes do Google Drive.
- `extrair_frames.py`: extração de frames de vídeo.
