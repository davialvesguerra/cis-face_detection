import numpy as np
import cv2

def reshape_image_to_emotions_model(img_2D: np.array, format_2D: tuple):
  #trasform image in 48x48 format
  img = cv2.resize(img_2D, format_2D)
  #add a new dimension in image
  #like this: 1 x 48 x 48 x 1
  img = np.expand_dims(img, axis = (0, 3))

  return img

