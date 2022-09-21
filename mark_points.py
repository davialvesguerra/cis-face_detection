import numpy as np
import pandas as pd
import cv2 as cv
import argparse

class MarkPoints:

  def __init__(self, path_image=False):
    self.path_image = path_image
    self.data = []

  def get_image(self):
    if self.path_image:
      self.img = cv.imread(self.path_image)
      self.copy_img = cv.imread(self.path_image)

    else:
      self.img = np.zeros((512,512,3), np.uint8)
      self.copy_img = np.zeros((512,512,3), np.uint8)

  def draw_circle(self, event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
      cv.circle(self.img,(x,y),5,(255,0,0),-1)
      self.data.append([x,y])

  def set_first_window(self):
    cv.namedWindow('first_window')
    cv.setMouseCallback('first_window', self.draw_circle)

    while True:
        cv.imshow('first_window',self.img)
        if cv.waitKey(20) & 0xFF == 27:
            break

    self.img = np.copy(self.copy_img) 
    cv.destroyAllWindows()

  def set_second_window(self):
    cv.namedWindow('second_window')
    cv.setMouseCallback('second_window', self.draw_circle)

    while True:
        cv.imshow('second_window',self.img)
        if cv.waitKey(1) & 0xFF == 27:
            break
    
    self.img = np.copy(self.copy_img) 
    cv.destroyAllWindows()

  def create_dataframe_points(self):
    pd.DataFrame(self.data, ).to_csv('data_points.csv', header=False, index=False)

  def open_image(self):
    self.get_image()
    self.set_first_window()
    self.set_second_window()
    self.create_dataframe_points()
  
    
def parse_arguments():
  ap = argparse.ArgumentParser()
  ap.add_argument('-p','--image_path',required=True, help='Path of image file')

  return vars(ap.parse_args())

if __name__ == "__main__":
  args = parse_arguments()
  image_path = args['image_path']

  image = MarkPoints(image_path)
  image.open_image()


