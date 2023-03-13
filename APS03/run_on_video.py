#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import biblioteca

print("Baixe o arquivo a seguir para funcionar: ")
print("https://raw.githubusercontent.com/Insper/robot22.1/main/aula03/yellow.mp4")

cap = cv2.VideoCapture('yellow.mp4')

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    # frame = cv2.imread("frame0000.jpg")
    # ret = True
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")
        break
    else:
        mask = img.copy()

        # Imagem original
        cv2.imshow('img',img)
        # Mascara
        cv2.imshow('mask',mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

