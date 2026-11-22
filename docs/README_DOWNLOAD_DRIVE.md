# Boccia Vision — Download dos datasets do Google Drive

O script `scripts/download_datasets.py` usa **rclone** para copiar os datasets da pasta `data/` compartilhada no Google Drive para o clone local do repositório. Não altera o Google Drive e não sobrescreve arquivos locais existentes.

## 1. Instalar e configurar o rclone (uma vez por integrante)

1. Instale o rclone conforme o guia oficial: https://rclone.org/install/.
2. No terminal, execute `rclone config`.
3. Crie um novo remote chamado **`boccia_drive`**, tipo **Google Drive** (`drive`).
4. Escolha o escopo de **somente leitura** (`drive.readonly`), que permite listar e baixar, mas não permite editar ou excluir arquivos no Drive.
5. Faça login com uma conta que possua acesso à pasta compartilhada `data/` do Boccia Vision.
6. Não é necessário inserir o ID da pasta raiz na configuração do remote: o script informa o ID da pasta `data/` para o rclone durante cada operação.

**Atenção, 2026:** a documentação oficial do rclone informa que o ID de cliente OAuth compartilhado do projeto está sendo descontinuado em 2026. Se a autenticação com o cliente padrão não funcionar, será necessário configurar um **OAuth Client ID próprio** no Google Cloud, seguindo as instruções oficiais: https://rclone.org/drive/#making-your-own-client-id. Nunca adicione o ID secreto do cliente, tokens ou o arquivo de configuração do rclone ao GitHub.

A pasta `data/` já identificada pela equipe tem ID público de referência de pasta:

`12im3QZE-9AKgwfJ4McS32XXAsibe1jyi`

Esse ID não é uma credencial e sozinho não concede acesso a um Drive privado.

## 2. Validar a configuração

Na raiz do projeto, execute:

```bash
rclone listremotes
```

O resultado deve incluir `boccia_drive:`. Para verificar que o remote consegue ler a pasta da equipe, execute:

```bash
rclone lsf boccia_drive: --drive-root-folder-id 12im3QZE-9AKgwfJ4McS32XXAsibe1jyi
```

Espera-se listar pelo menos as pastas `roboflow/` e `videos/`.

## 3. Baixar os datasets

Na raiz do repositório, execute primeiro a prévia:

```bash
python scripts/setup_project.py --dry-run
python scripts/setup_project.py
python scripts/download_datasets.py --version v1 --dry-run
```

Depois, faça a cópia:

```bash
python scripts/download_datasets.py --version v1
```

Para baixar somente uma fonte:

```bash
python scripts/download_datasets.py --version v1 --source synthetic
```

Para baixar `v2` depois que ela existir no Drive:

```bash
python scripts/download_datasets.py --version v2
```

Se o remote tiver outro nome, use `--remote NOME`. Se a pasta principal de dados for migrada, use `--drive-folder-id NOVO_ID`.

## 4. Estrutura local gerada

```text
data/
  roboflow/
    youtube/v1/
      images/
      labels/
      data.yaml
    local/v1/
      images/
      labels/
      data.yaml
    synthetic/v1/
      images/
      labels/
      data.yaml
```

O script normaliza o nome local para `youtube/`, mesmo que a pasta do Drive ainda se chame `Youtube/`.

Na checagem anterior do Drive havia **121 imagens YouTube, 120 locais e 121 sintéticas**. O script **não reduz automaticamente** essas contagens para 120: a amostragem e o balanceamento acontecerão na etapa de processamento.

## 5. Integridade e segurança

- Faz pré-validação no Drive para confirmar `images/`, `labels/` e `data.yaml` em cada origem e versão.
- Exige correspondência de nomes entre imagens e labels. Anotações vazias são aceitas como potenciais imagens negativas.
- Copia somente arquivos ausentes (`rclone copy --ignore-existing` / `copyto --ignore-existing`).
- Executa `rclone check` após a cópia, verificando divergências entre arquivos de origem e destino.
- Não remove, não substitui e não envia arquivos ao Drive.
- Se detectar conflito ou arquivo incompleto, interrompe com erro para revisão manual.
- Verifica a quantidade esperada de imagens e labels após a cópia.

**Importante:** essa etapa valida correspondência e integridade de transporte, mas **não valida visualmente todas as anotações** nem implementa os splits, a conversão de classes ou os datasets híbridos. O `data.yaml` de origem preserva os caminhos fornecidos pelo Roboflow, que podem não apontar para as pastas finais do treinamento; o processador criará novos YAMLs apropriados.

O script foi testado com um **Drive simulado em ambiente local**, incluindo falhas intencionais, e teve sua sintaxe verificada. A autenticação real do rclone na conta da equipe e a transferência efetiva pelo Google Drive ainda precisam ser verificadas no computador de um integrante.

## 6. Privacidade

Recomenda-se definir compartilhamento restrito no Google Drive e conceder acesso somente aos integrantes. Prefira permissão de leitor para quem só precisa baixar datasets. Não registre credenciais ou tokens em arquivos versionados.
