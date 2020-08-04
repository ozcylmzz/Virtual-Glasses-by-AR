import tkinter
import tkinter.messagebox
from tkinter import *

import cv2

glasses = 0

def helloCallBack():
   tkinter.messagebox.showinfo( "Hello Python", "Hello World")
   # do something for this button

def LiveImageListener():
   tkinter.messagebox.showinfo( "Hello Python", "LiveImageListener")
   # do something for this button

def PhotoUploadListener():
   tkinter.messagebox.showinfo( "Hello Python", "PhotoUploadListener")
   # do something for this button

def AAListener():
   tkinter.messagebox.showinfo( "1. Gözlük", "1. Gözlük seçildi, görüntü değişiyor.")
   # do something for this button
   global glasses
   glasses = 1
   # glasses.set(1)
   # print(glasses)

def BBListener():
   tkinter.messagebox.showinfo( "2. Gözlük", "2. Gözlük seçildi, görüntü değişiyor.")
   # do something for this button
   global glasses
   glasses = 2
   # print(glasses)

def CCListener():
   tkinter.messagebox.showinfo( "3. Gözlük", "3. Gözlük seçildi, görüntü değişiyor.")
   # do something for this button

def DDListener():
   tkinter.messagebox.showinfo( "4. Gözlük", "4. Gözlük seçildi, görüntü değişiyor.")
   # do something for this button

def xxListener():
   tkinter.messagebox.showinfo( "1. Sap", "1. Sap seçildi, görüntü değişiyor.")
   # do something for this button

def yyListener():
   tkinter.messagebox.showinfo( "2. Sap", "2. Sap seçildi, görüntü değişiyor.")
   # do something for this button

def zzListener():
   tkinter.messagebox.showinfo( "3. Sap", "3. Sap seçildi, görüntü değişiyor.")
   # do something for this button

# glasses = 0

top = tkinter.Tk()

imgAA = tkinter.PhotoImage(file = r"g2.png")
imgBB = tkinter.PhotoImage(file = r"glasses.png")
imgCC = tkinter.PhotoImage(file = r"sunglasses2.png")
# imgDD = tkinter.PhotoImage(file = r"Faces\\face0.jpg")
imgxx = tkinter.PhotoImage(file = r"glasses_ear_left.png")
imgyy = tkinter.PhotoImage(file = r"glasses_ear_right.png")
# imgzz = tkinter.PhotoImage(file = r"Faces\\face0.jpg")

imgAA = imgAA.subsample(2, 2)
imgBB = imgBB.subsample(3, 3)
imgCC = imgCC.subsample(3, 3)
# imgDD = imgDD.subsample(3, 3)
imgxx = imgxx.subsample(3, 3)
imgyy = imgyy.subsample(3, 3)
# imgzz = imgzz.subsample(3, 3)

liveImage = tkinter.Button(top, text ="Canlı Görüntü", command = LiveImageListener)
UploadPhoto = tkinter.Button(top, text ="Fotoğraf yükleme", command = PhotoUploadListener)

AA = tkinter.Button(top, text ="1. Gözlük", image = imgAA, command = AAListener)
AA = tkinter.Button(top, text ="1. Gözlük", image = imgAA, command = AAListener)
BB = tkinter.Button(top, text ="2. Gözlük", image = imgBB, command = BBListener)
CC = tkinter.Button(top, text ="3. Gözlük", image = imgCC, command = CCListener)
# DD = tkinter.Button(top, text ="4. Gözlük", image = imgDD, command = DDListener)

xx = tkinter.Button(top, text ="1. Çerçeve", image = imgxx, command = xxListener)
yy = tkinter.Button(top, text ="2. Çerçeve", image = imgyy, command = yyListener)
# zz = tkinter.Button(top, text ="3. Çerçeve", image = imgzz, command = zzListener)

liveImage.pack()
UploadPhoto.pack()
AA.pack()
BB.pack()
CC.pack()
# DD.pack()

xx.pack()
yy.pack()
# zz.pack()


# imgAA = cv2.imread('Faces\\face4.jpg', 1)
# dim = (10, 10)
# imgAA = cv2.resize(imgAA, dim, interpolation=cv2.INTER_AREA)
# cv2.imshow("afsad", imgAA)

# l = Label(image = img)
# l.pack()

# Button(top, text = 'Click Me !', image = img, compound = LEFT).pack(side = TOP)

# filename = ImageTk.PhotoImage(Image.open('imagename.jpeg' ))
# background_label = tk.Label(self.root, image=filename)
# background_label.place(x=0, y=0, relwidth=1, relheight=1)



print(glasses)



top.mainloop()






# butonlara fotoğraf ekleme
# görünüme glasses.py output sonucunu ekleyecek alan araştırma
# butonların seçimine uygun şekilde arka planda veri güncellemesi