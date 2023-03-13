#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

def segmenta_linha_branca(bgr):
    """Não mude ou renomeie esta função
        deve receber uma imagem e segmentar as faixas brancas
    """

    res = np.zeros(bgr.shape[:-1], dtype=np.uint8)

    return res

def estimar_linha_nas_faixas(img, mask):
    """Não mude ou renomeie esta função
        deve receber uma imagem preta e branca e retorna dois pontos que formen APENAS uma linha em cada faixa. Desenhe cada uma dessas linhas na iamgem.
         formato: [[(x1,y1),(x2,y2)], [(x1,y1),(x2,y2)]]
    """
   
    return None

def calcular_equacao_das_retas(linhas):
    """Não mude ou renomeie esta função
        deve receber dois pontos que estejam em cada uma das
        faixas e retornar as equacões das duas retas, 
        onde y = h + m * x. Formato: [(m1,h1), (m2,h2)]
    """
    
    return []

def calcular_ponto_de_fuga(img, equacoes):
    """Não mude ou renomeie esta função
        deve receber duas equacoes de retas e retornar o ponto de encontro entre elas. Desenhe esse ponto na imagem.
    """

    return img.copy(), (img.shape[1]//2,img.shape[0]//2)

        
