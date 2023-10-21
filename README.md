# Projeto de Análise de Gestos e Libras com OpenCV

Este é um guia para executar o projeto de análise de gestos em Python, que utiliza diversas bibliotecas, como OpenCV, NumPy e Matplotlib. Além disso, é necessário baixar o arquivo de modelo de pose. Siga os passos abaixo para configurar e executar o projeto.

## Pré-requisitos

Antes de executar o projeto, certifique-se de ter os seguintes pré-requisitos instalados:

- Python (versão 3.6 ou superior)
- OpenCV (instale com `pip install opencv-python`)
- NumPy (instale com `pip install numpy`)
- Matplotlib (instale com `pip install matplotlib`)
- Arquivos de modelo de pose:
  - [Baixe o arquivos de modelo](https://drive.google.com/drive/folders/1Lz_mCHW_Phjq3k2ebKYaMsfYy6MnZuY5?usp=share_link)

Certifique-se de que todos esses pré-requisitos estejam instalados antes de continuar.

## executando o projeto

Para executar o projeto, siga os passos abaixo:
após baixar os arquivos do modelo de pose, coloque-os na pasta `pose` na raiz do projeto.
ficando assim "pose/hand/pose_deploy.prototxt" e "pose/hand/pose_iter_102000.caffemodel" com os arquivos de modelo de pose na pasta pose.

1. Abra o terminal na pasta raiz do projeto
2. Execute o comando `python camDetector.py` para executar o projeto com a câmera do computador e realizar gestos com a mão para indentificar a letra em libras.
3. Execute o comando `python imageDetector.py` para executar o projeto com uma imagem com gestos de mão para indentificar a letra em libras e posição da mão e dedos.