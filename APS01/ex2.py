#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())


def realca_caixa_vermelha(bgr): 
    """Não mude ou renomeie esta função
        deve receber uma imagem bgr
        e fazer alguma filtragem / seleção de modo a obter uma imagem
        de saída grayscale
        em que somente os pixels da caixa estão brancos e todo o restante está preto
        Dica: Use mais de um canal, por exemplo R e B
    """
    r, g, b = cv2.split(cv2.cvtColor((bgr.copy()), cv2.COLOR_BGR2RGB))
    mascara_r, mascara_g, mascara_b = np.zeros_like(r), np.zeros_like(g), np.zeros_like(b)
    mascara_r[r > 200], mascara_g[g < 120], mascara_b[b < 150] = 255, 255, 255
    return cv2.bitwise_and(mascara_b, mascara_g, mask=mascara_r)


if __name__ == "__main__":
    img = cv2.imread("img/cena_canto_sala.jpg")
    
    # Faz o processamento
    saida = realca_caixa_vermelha(img)
    cv2.imwrite( "saida_ex2.png", saida)


    # NOTE que a OpenCV terminal trabalha com BGR
    cv2.imshow('entrada', img)

    cv2.imshow('saida', saida)

    cv2.waitKey()
    cv2.destroyAllWindows()

