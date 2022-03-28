import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import cv2
print(cv2.__version__)

sys.path.append("/modulos/")
print(sys.path)

import modulos.extrator_POSICAO as posicao
import modulos.extrator_ALTURA as altura
import modulos.extrator_PROXIMIDADE as proximidade
import modulos.alfabeto as alfabeto


arquivo_proto = "pose/hand/pose_deploy.prototxt"
arquivo_pesos = "pose/hand/pose_iter_102000.caffemodel"
numero_pontos = 22
pares_poses = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
              [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
              [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]


letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q',
          'R', 'S', 'T', 'U', 'V', 'W']

modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)

window_name = 'imagem'

imagem = cv2.imread("imagens/hand/Libras/B.JPG")

cv2.imshow(window_name, imagem)

imagem_copia = np.copy(imagem)

imagem_largura = imagem.shape[1]
imagem_altura = imagem.shape[0]
proporcao = imagem_largura / imagem_altura

print(imagem_largura, imagem_altura, proporcao)


cor_pontoA, cor_pontoB, cor_linha = (14, 201, 255), (255, 0, 128), (192, 192, 192)
cor_txtponto = (10, 216, 245)

tamanho_fonte, tamanho_linha, tamanho_circulo, espessura = 5, 1, 4, 2

fonte = cv2.FONT_HERSHEY_SIMPLEX



entrada_altura = 256
entrada_largura = int(((proporcao * entrada_altura) * 8) // 8)


print(entrada_altura, entrada_largura)






entrada_blob = cv2.dnn.blobFromImage(imagem, 1.0 / 255,
                                     (entrada_largura, entrada_altura),
                                     (0, 0, 0), swapRB=False, crop=False)



modelo.setInput(entrada_blob)
saida = modelo.forward()

print(saida.shape)


pontos = []
limite = 0.1
for i in range(numero_pontos):
    mapa_confianca = saida[0, i, :, :]
    mapa_confianca = cv2.resize(mapa_confianca, (imagem_largura, imagem_altura))

    _, confianca, _, ponto = cv2.minMaxLoc(mapa_confianca)

    if confianca > limite:
        cv2.circle(imagem_copia, (int(ponto[0]), int(ponto[1])), 5, cor_pontoA,
                   thickness=espessura, lineType=cv2.FILLED)
        cv2.putText(imagem_copia, ' ' + (str(int(ponto[0]))) + ',' +
                    str(int(ponto[1])), (int(ponto[0]), int(ponto[1])),
                    fonte, 0.3, cor_txtponto, 0, lineType=cv2.LINE_AA)

        cv2.circle(imagem, (int(ponto[0]), int(ponto[1])), tamanho_circulo,
                   cor_pontoA,
                   thickness=espessura, lineType=cv2.FILLED)
        cv2.putText(imagem, ' ' + "{}".format(i), (int(ponto[0]),
                                                  int(ponto[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    cor_txtponto,
                    0, lineType=cv2.LINE_AA)

        pontos.append((int(ponto[0]), int(ponto[1])))

    else:
        pontos.append((0, 0))

print(len(pontos))


print(pontos)

cv2.imshow('imagem2',imagem_copia)





for par in pares_poses:
    parteA = par[0]
    parteB = par[1]

    if pontos[parteA] != (0, 0) and pontos[parteB] != (0, 0):
        cv2.line(imagem_copia, pontos[parteA], pontos[parteB], cor_linha,
                 tamanho_linha, lineType=cv2.LINE_AA)
        cv2.line(imagem, pontos[parteA], pontos[parteB], cor_linha, tamanho_linha,
                 lineType=cv2.LINE_AA)


cv2.imshow('imagem3',imagem_copia)


posicao.posicoes = []


# Dedo polegar
posicao.verificar_posicao_DEDOS(pontos[1:5], 'polegar', altura.verificar_altura_MAO(pontos))

# Dedo indicador
posicao.verificar_posicao_DEDOS(pontos[5:9], 'indicador', altura.verificar_altura_MAO(pontos))


# Dedo médio
posicao.verificar_posicao_DEDOS(pontos[9:13], 'medio', altura.verificar_altura_MAO(pontos))


# Dedo anelar
posicao.verificar_posicao_DEDOS(pontos[13:17], 'anelar', altura.verificar_altura_MAO(pontos))


# Dedo mínimo
posicao.verificar_posicao_DEDOS(pontos[17:21], 'minimo', altura.verificar_altura_MAO(pontos))

print(posicao.posicoes)


p = proximidade.verificar_proximidade_DEDOS(pontos)

print(p)


for i, a in enumerate(alfabeto.letras):
  if proximidade.verificar_proximidade_DEDOS(pontos) == alfabeto.letras[i]:
    cv2.putText(imagem, ' ' + letras[i], (50,50), fonte, 1, cor_txtponto,
                tamanho_fonte, lineType=cv2.LINE_AA)




plt.figure(figsize= [14,10])
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))

cv2.imshow('imagem4',imagem)


cv2.waitKey(0)

cv2.destroyAllWindows()