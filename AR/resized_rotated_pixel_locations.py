import cv2
from scipy import ndimage
import math
import numpy as np

image = cv2.imread("sunglasses.png", -1)
# cv2.circle(image, (500,140), 2, color= (0,0,255) ,thickness = -5)
# cv2.imwrite('orj2.jpeg', image)
x0, y0 = 500, 140
# print(x0, y0)
angle = 35
xy = np.array([x0,y0])
im_rot = ndimage.rotate(image,angle)
cv2.imwrite('orj1.jpeg', im_rot)
org_center = (np.array(image.shape[:2][::-1])-1)/2
rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
org = xy-org_center
a = np.deg2rad(angle)
new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a), -org[0]*np.sin(a) + org[1]*np.cos(a) ])
x1,y1 = new+rot_center
# print(x1,y1)
x1 = int(round(x1,0))
y1 = int(round(y1,0))
# x1, y1 = int(x1,y1)
print(x1,y1)
cv2.circle(im_rot, (x1,y1), 2, color= (0,0,255) ,thickness = -5)
cv2.imwrite('orj2.jpg', im_rot)