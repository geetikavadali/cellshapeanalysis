import cv2
from google.colab.patches import cv2_imshow

pad_perc = 0.25 #size percentage of border wrt source image size

def add_border(src, pad_perc):
  top = int(pad_perc * src.shape[0]) 
  bottom = top
  left = int(pad_perc * src.shape[1]) 
  right = left
  
  dst = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
  return dst


def save_separated_cells(image):
  # loading image of collection of mef cells
  image = image
  # grayscaling image
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # thresholding white cells selecting only most prominent
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

  # generating contours
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  ROI_number = 0
  # iterating over contours
  for c in cnts:
      # forming bounding rectangle over roi's
      x,y,w,h = cv2.boundingRect(c)
      ROI = image[y:y+h, x:x+w]
      # generating separate images for different ROI's with added borders
      cv2.imwrite('{}.png'.format(ROI_number), add_border(ROI, pad_perc))
      ROI_number += 1
