import cv2
import numpy as np
import os

def remove_background(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(thresh)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            result = remove_background(image_path)
            output_path = os.path.join(folder_path, 'processed_' + filename)
            cv2.imwrite(output_path, result)

folder_path = './icons_black'
process_images_in_folder(folder_path)
