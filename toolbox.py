# -*- coding: utf-8 -*-
import cv2
import numpy as np

def segment_fruit(image):
    
    # Limiarize cada um dos canais de forma independente
    ret, imR = cv2.threshold(image[:,:,0],0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret, imG = cv2.threshold(image[:,:,1],0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret, imB = cv2.threshold(image[:,:,2],0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Combine as segmentações utilizando o operador OR
    segmentation = imR | imG | imB

    # Preencha as regiões que não foram corretamente detectadas
    im_floodfill = segmentation.copy()
    h, w = segmentation.shape[:2]
    maskT = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, maskT, (0,0), 255);
    segmentation_filled = segmentation | ~im_floodfill

    # Desenhe apenas o maior objeto presente na imagem
    im2, contours, hierarchy = cv2.findContours(segmentation_filled.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    final_segmentation = np.zeros((h, w), np.uint8)
    cv2.drawContours(final_segmentation, [contours[0]], 0, 1, cv2.FILLED)
    
    # Retorne a imagem segmentada
    return final_segmentation > 0.5, contours[0]

def get_dominant_color(image, segmentation):
    
    mask = np.zeros(image.shape[:2], np.uint8) + segmentation
    mean_color = np.uint8(cv2.mean(image, mask=mask))[:3]
    
    return mean_color

def get_fruit_struct(image, contour):

    x,y,w,h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    
    return aspect_ratio


if __name__ == '__main__':

    print cv2.__version__
