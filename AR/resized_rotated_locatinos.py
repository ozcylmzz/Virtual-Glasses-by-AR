import cv2
from scipy import ndimage
import math

glasses = cv2.imread("sunglasses.png", -1)
# cv2.imwrite('orj1.jpeg', glasses)
# # cv2.waitKey(0)
# cv2.circle(glasses, (500,140), 2, color= (0,0,255) ,thickness = -5)
# cv2.imwrite('orj2.jpeg', glasses)
height = glasses.shape[0]
width = glasses.shape[1]
print(height,width)
x = 800,250
img = cv2.resize(glasses,x,interpolation= cv2.INTER_AREA)
nheight = img.shape[0]
nwidth = img.shape[1]
print(nheight,nwidth)

rx = 800 / 1000
ry = 250 / 372

new_x = int(500 * rx)
new_y = int(140 * ry)

cv2.circle(img, (new_x,new_y), 2, color= (0,0,255) ,thickness = -5)
cv2.imwrite('res1.jpeg', img)


angle = 45
img2 = ndimage.rotate(glasses, (angle))


print(img2.shape[0])
print(img2.shape[1])

# new_x1 = cos(theta)*old_x1 - sin(theta)*old_y1 ; new_y1 = sin(theta)*old_x1 + cos(theta)*old_y1
# xp = (x - center_x) * cos(angle) - (y - center_y) * sin(angle) + center_x
# yp = (x - center_x) * sin(angle) + (y - center_y) * cos(angle) + center_y


# new_x1 = int(math.cos(theta) * 500 - math.sin(theta) * 140)
# new_y1 = int(math.sin(theta) * 500 + math.cos(theta) * 140)

center_x = glasses.shape[0] / 2
center_y = glasses.shape[1] / 2

xp = int((500 - center_x) * math.cos(angle) - (140 - center_y) * math.sin(angle) + center_x)
yp = int((500 - center_x) * math.sin(angle) + (140 - center_y) * math.cos(angle) + center_y)

print("qwe")
print(xp,yp)

cv2.circle(img2, (xp,yp), 2, color= (0,0,255) ,thickness = -5)
cv2.imwrite('qq1.jpeg', img2)

cv2.waitKey(0)