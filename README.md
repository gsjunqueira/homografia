# Projeto de Fusão de Imagens Usando Homografia

Este projeto implementa um pipeline para fusão de imagens utilizando homografia, capturando um conjunto de dados com um smartphone. As imagens são capturadas rotacionando o aparelho em torno do próprio eixo, como se estivesse capturando uma imagem panorâmica. O processo envolve a detecção de pontos-chave, correspondência de descritores, cálculo da homografia e, finalmente, a fusão de imagens para criar uma imagem final composta.

## Estrutura do Projeto

- `homography.py`: Módulo que contém as funções para detectar pontos-chave, descrever as imagens, calcular a homografia e realizar a fusão das imagens.
- `main.py`: Script principal que executa o pipeline de homografia em um conjunto de imagens.

## Pré-requisitos

- Python 3.x
- Poetry

## Configuração do Ambiente

1. **Instale o Poetry**: Se ainda não o fez, siga as instruções de instalação no [site oficial do Poetry](https://python-poetry.org/docs/#installation).

2. **Clone o Repositório**: Se ainda não o fez, clone o repositório do projeto:

    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd <NOME_DO_DIRETORIO>
    ```

3. **Instale as Dependências**: No diretório do projeto, execute o comando para instalar todas as dependências especificadas no `pyproject.toml`:

    ```bash
    poetry install
    ```

4. **Ative o Ambiente Virtual**: Entre no ambiente virtual criado pelo Poetry:

    ```bash
    poetry shell
    ```

## Como Executar

1. **Captura de Imagens**: Capture um conjunto de imagens usando seu smartphone, rotacionando o dispositivo em torno do próprio eixo (como uma imagem panorâmica). Salve essas imagens em um diretório `datasets` dentro do diretório do projeto.

2. **Organize as Imagens**: Coloque as imagens capturadas no diretório `datasets`. Certifique-se de que as imagens estejam nomeadas em uma sequência lógica para a fusão (por exemplo, `IMG_3855.jpeg`, `IMG_3856.jpeg`, etc.).

3. **Executar o Pipeline**: Use o Poetry para executar o script principal `main.py`:

    ```bash
    poetry run python main.py
    ```

4. **Resultados**: As imagens intermediárias e o resultado final da fusão serão salvos no diretório `datasets` com os seguintes nomes:
    - `Homografia_1.jpg`
    - `Homografia_2.jpg`
    - `Homografia_3.jpeg`

## Explicação das Funções

### `detect_keypoint_describe(image)`

Detecta pontos-chave e calcula os descritores usando o algoritmo SIFT. Retorna os pontos-chave, descritores e imagens com os pontos destacados.

### `match_filter_describe(describe_1, describe_2, threshold=0.9)`

Filtra as correspondências entre dois conjuntos de descritores usando força bruta e o teste de razão.

### `calcula_homografia(k_pts_1, k_pts_2, corresp)`

Calcula a matriz de homografia entre dois conjuntos de pontos-chave correspondentes.

### `stitching_mascara(im1, im2, h)`

Realiza o mapeamento e a fusão de duas imagens usando a matriz de homografia.

### `dual_homografia(im_1, im_2, path, passo)`

Executa a homografia entre duas imagens, salvando os resultados intermediários e finais.

### `main()`

Executa todo o pipeline para um conjunto de imagens, aplicando a função `dual_homografia` em pares de imagens e salvando os resultados.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma _issue_ ou _pull request_ para melhorias.

## Licença

Este projeto é licenciado sob os termos da MIT License.
