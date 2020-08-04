import dlib
import cv2
import numpy as np
from scipy import ndimage

glassesPath = "sunglasses2"

file = open("glasses.txt", "r")
for line in file:
    word = line.split()
    if (word[0] == glassesPath):
        glassesCenterX = int(word[1])
        glassesCenterY = int(word[2])

        earLetfX = int(word[3])
        earLetfY = int(word[4])

        earRightX = int(word[5])
        earRightY = int(word[6])

file.close()


# Dosyadaki verilerin alımı / gözlük sapındaki noktalar
file_ear = open("glasses_ear_left.txt", "r")
for line in file_ear:
    word = line.split()
    if (word[0] == glassesPath):
        ear_left_X_file = int(word[1])
        ear_left_Y_file = int(word[2])
        ear_left_file = int(word[1]), int(word[2])

        ear_eye_left_X_file = int(word[3])
        ear_eye_left_Y_file = int(word[4])
        ear_eye_left_file = int(word[3]), int(word[4])

file_ear.close()


file_ear = open("glasses_ear_right.txt", "r")
for line in file_ear:
    word = line.split()
    if (word[0] == glassesPath):
        ear_right_X_file = int(word[1])
        ear_right_Y_file = int(word[2])
        ear_right_file = int(word[1]), int(word[2])

        ear_eye_right_X_file = int(word[3])
        ear_eye_right_Y_file = int(word[4])
        ear_eye_right_file = int(word[3]), int(word[4])

file_ear.close()


file_path = glassesPath + "//" + glassesPath + ".txt"
f = open(file_path, "r")
mtr = []

for line in f:
    word = line.split()
    i = int(word[0])
    mtr.append( [int(word[1]), int(word[2]), int(word[3]), int(word[4])] )

f.close()


video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def resized_pixel_locations(sunglasses_img, x0, y0, resizeRate, angle):

    resizeRateX = resizeRate / sunglasses_img.shape[1]
    resizeRateY = resizeRate / sunglasses_img.shape[1]
    sunglasses_img = resize(sunglasses_img, resizeRate)
    x0 = int(x0 * resizeRateX)
    y0 = int(y0 * resizeRateY)

    sunglasses_img_ret = sunglasses_img
    # Fonksiyondan dönecek image tekrar rotate edilmemeli, bu sebepten "sunglasses_img_ret" değişkenine alındı.

    xy = np.array([x0, y0])

    im_rot = ndimage.rotate(sunglasses_img, angle)
    cv2.imwrite('orj1.jpeg', im_rot)
    org_center = (np.array(sunglasses_img.shape[:2][::-1]) - 1) / 2
    rot_center = (np.array(im_rot.shape[:2][::-1]) - 1) / 2.

    org = xy - org_center

    a = np.deg2rad(angle)
    new = np.array([org[0] * np.cos(a) + org[1] * np.sin(a), -org[0] * np.sin(a) + org[1] * np.cos(a)])

    x1, y1 = new + rot_center
    x1 = int(round(x1, 0))
    y1 = int(round(y1, 0))

    cv2.circle(im_rot, (x1, y1), 2, color=(0, 0, 255), thickness=-5)
    cv2.imwrite('orj2.jpeg', im_rot)

    return x1, y1, sunglasses_img_ret


#Resize an image to a certain width
def resize(img, width):

    # Resize fonksiyonu iki parametre alır;
    # img => Kullanılan gözlük fotoğrafının adı. (Farkı klasörde ise dosya yolu bilgisi)
    # width => Yüz tespiti ile bulunan noktalara göre hesaplanan yüz genişliği bilgisi
    #
    # Resize fonksiyonu bu bilgiler ile "cv2.resize()" fonksiyonunu kullanarak gözlük fotoğrafını yeniden boyutlandırır ve bu fotoğrafı döndürür.

    r = float(width) / img.shape[1]
    dim = (width, int(img.shape[0] * r))
    # dim = dim * 0,8 # Gözlük resize için yeniden
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


#Combine an image that has a transparency alpha channel
def blend_transparent(face_img, sunglasses_img):

    # blend_transparent() fonkssiyonu geözlük fotoğrafının opaklığının ayarlanması için kullanılmaktadır.
    # ".png" uzantılı dosyalarda RGB değerlerine ek olarak "ALPHA" değeri bulunur. Bu değer ilgili pixel in opaklığını göstermektedir.
    # Alpha değeri kullanılarak gözlük camlarındaki opaklık sağlanmış olur, bu şekilde oluşturulan son fotoğraf ana programa döndürülür.

    # print("11110000 " + sunglasses_img.shape)

    overlay_img = sunglasses_img[:,:,:3]
    overlay_mask = sunglasses_img[:,:,3:]

    background_mask = 255 - overlay_mask

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


#Find the angle between two points
def angle_between(point_1, point_2):
    angle_1 = np.arctan2(*point_1[::-1])
    angle_2 = np.arctan2(*point_2[::-1])
    return np.rad2deg((angle_1 - angle_2) % (2 * np.pi))


def calculate_angle(left, left1, right, right1, nose_mid, nose_end):
    angle_one = ((nose_mid[0] - left[0]) / 4 - (right[0] - nose_mid[0]) / 4)
    angle_two = ((nose_end[0] - left1[0]) / 4 - (right1[0] - nose_end[0]) / 4)
    return round((angle_one + angle_two) / 1.75, 4)


def get_pixel_locations(input_img, x0, y0, resizeRate, angle, nose, grrcx0, grrcy0):

    resizeRateX = resizeRate / input_img.shape[1]
    resizeRateY = resizeRate / input_img.shape[1]
    input_img = resize(input_img, resizeRate)
    x0 = int(x0 * resizeRateX)
    y0 = int(y0 * resizeRateY)

    xy = np.array([x0, y0])

    im_rot = ndimage.rotate(input_img, angle)
    cv2.imwrite('test----orj00.jpeg', im_rot)
    org_center = (np.array(input_img.shape[:2][::-1]) - 1) / 2
    rot_center = (np.array(im_rot.shape[:2][::-1]) - 1) / 2.

    org = xy - org_center

    a = np.deg2rad(angle)
    new = np.array([org[0] * np.cos(a) + org[1] * np.sin(a), -org[0] * np.sin(a) + org[1] * np.cos(a)])

    x1, y1 = new + rot_center
    x1 = int(round(x1, 0))
    y1 = int(round(y1, 0))

    cv2.circle(im_rot, (x1, y1), 2, color=(0, 0, 255), thickness=-5)
    cv2.imwrite('test----orj01.jpeg', im_rot)

    x1 = nose[0] - grrc_x1 + x1
    y1 = nose[1] - grrc_y1 + y1

    return x1, y1




#Start main program
while True:


    #  Kameradan fotoğraf alımı
    ret, img = video_capture.read()
    # Canlı resim yerine img input deneniyor
    # img = cv2.imread('Faces\\face11.jpg', 1)
    # ret = img

    img = resize(img, 700)
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        # detect faces
        #  Yüz tespiti

        dets = detector(gray, 1)
        #find face box bounding points
        for d in dets:

            x = d.left()
            y = d.top()
            w = d.right()
            h = d.bottom()

        dlib_rect = dlib.rectangle(x, y, w, h)

        ##############   Find facial landmarks   ##############
        #  Yüzdeki "landmark" ların tespiti
        detected_landmarks = predictor(gray, dlib_rect).parts()
        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])

            # Yüzde tespit edilen landmarklardan yola çıkılarak gözlüğün geleceği noktaların bulunması işlemi bu blokta yapılmaktadır.
            # Sol göz için;
            # 	x koordinatı olarak => 18 numaralı noktanın x koordinatı,
            # 	y koordinatı olarak => 37 numaralı noktanın x koordinatı,
            # Sağ göz için;
            # 	x koordinatı olarak => 27 numaralı noktanın x koordinatı,
            # 	y koordinatı olarak => 46 numaralı noktanın x koordinatı,
            # seçilmiştir.

            if idx == 0:  #
                left = pos
            elif idx == 1:  #
                left1 = pos
            elif idx == 15:  #
                right1 = pos
            elif idx == 16:  #
                right = pos
            elif idx == 17:
                eye_left_posx = pos
            elif idx == 26:
                eye_right_posx = pos
            elif idx == 36:
                eye_left_posy = pos
            elif idx == 45:
                eye_right_posy = pos
            elif idx == 27:
                nose = pos
            elif idx == 28:  #
                nose_mid = pos
            elif idx == 30:  #
                nose_end = pos

            # Bulunan her "landmark" noktasına bir halka çizdirilmesi
            # cv2.circle(img_copy, pos, 2, color=(0, 0, 255), thickness=-5)

        # ortaleftx = (-left[0] +left1[0]) / 2
        # ortarightx = (right[0] - right1[0]) / 2
        # ortalefty = (-left[1] + left1[1]) / 2
        # ortarighty = (right[1] - right1[1]) / 2

        head_degree = int(calculate_angle(left, left1, right, right1, nose_mid, nose_end))
        print("Head Degree *******")
        print(head_degree)
        if head_degree < 0:
            path = str(359 + head_degree) + ".png"
            head_degree4file = 359 + head_degree
        else:
            head_degree4file = head_degree
            path = format(str(head_degree).zfill(3)) + ".png"




        path = glassesPath + "/" + path
        glasses = cv2.imread(path,-1)
        eye_left = eye_left_posx[0], eye_left_posy[1]
        eye_right = eye_right_posx[0], eye_right_posy[1]

        # Gözlerin konumuna bağlı olarak yüz açısının tespit edilmesi işlemi
        degree = np.rad2deg(np.arctan2(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1]))
        print("2D Degree ++++")
        print(degree*-1 - 90)
        print("----------------")

        ##############   Resize and rotate glasses   ##############

        #Translate facial object based on input object.

        # Göz merkezinin belirlenmesi
        eye_center = (eye_left[1] + eye_right[1]) / 2

        #Sunglasses translation
        # Yüzde gözlüğün üst noktasının geleceği kısmın belirlenmesi
        glass_trans = int(.2 * (eye_center - y))


        # resize glasses to width of face and blend images
        # Yüz genişliğinin bulunması (Resize fonksiyonunda kullanılacaktır.)
        face_width = w - x
        resize_rate = ( (eye_left[0] - eye_right[0]) / (mtr[head_degree4file][0] - mtr[head_degree4file][2]) )
        # Alttaki iki satır tam olarak bu arada bulunmalı !
        grc_x0 = int(glassesCenterX * resize_rate)
        grc_y0 = int(glassesCenterY * resize_rate)
        resize_rate = int( round( resize_rate * glasses.shape[1], 0 ) )

        # Fonksiyon "resized_pixel_locations" fonksiyonunun içine alınmıştır.

        grrc_x1, grrc_y1, glasses_resize = resized_pixel_locations(glasses, glassesCenterX, glassesCenterY, resize_rate, degree + 90) # Buruna göre yerleştirme
        differenceX = nose[0] - (grrc_x1 + x)
        differenceY = nose[1] - (grrc_y1 + y + glass_trans)

        # Rotate glasses based on angle between eyes
        # Gözlük fotoğrafının yüz açısına göre döndürülmesi
        yG, xG, cG = glasses_resize.shape
        glasses_resize_rotated = ndimage.rotate(glasses_resize, (degree+90))
        glass_rec_rotated = ndimage.rotate(img[nose[1] - grrc_y1: nose[1] - grrc_y1 + yG, nose[0] - grrc_x1: nose[0] - grrc_x1 + xG], (degree + 90)) # Buruna göre yerleştirme

        #blending with rotation
        # Gözlük fotoğrafında opaklık ayarının yapılması (ilgili fonksiyon çağırılarak)
        h5, w5, s5 = glass_rec_rotated.shape
        rec_resize = img_copy[nose[1] - grrc_y1 : nose[1] - grrc_y1 + h5, nose[0] - grrc_x1 : nose[0] - grrc_x1 + w5] # Buruna göre yerleştirme
        blend_glass3 = blend_transparent(rec_resize , glasses_resize_rotated)


        # Ortaya çıkan yeniden boyutlandırılmış döndürülmüş gözlük fotoğrafının yüze yerleştirilmesi.
        img_copy[nose[1] - grrc_y1: nose[1] - grrc_y1 + h5, nose[0] - grrc_x1: nose[0] - grrc_x1 + w5] = blend_glass3 # Buruna göre yerleştirme

        # Bu aşamada önceki satırlardan ortaya çıkan koordinatlar kullanılmalı,
        # Bu koordinatlar yüzde 28 numara
        # Ve gözlük merkezi olmaalı


        if glassesPath == "yeni3" or glassesPath == "sunglasses2":
            ear_img_left = cv2.imread("glasses_ear_left.png", -1)
            ear_img_right = cv2.imread("glasses_ear_right.png", -1)
        else:
            ear_img_right = cv2.imread("glasses_ear_right_OLD.png", -1)
            ear_img_left = cv2.imread("glasses_ear_left_OLD.png", -1)

        if head_degree > 2:
            pixel4earX, pixel4earY = get_pixel_locations( glasses, mtr[head_degree4file][0], mtr[head_degree4file][1], resize_rate, degree + 90, nose, grrc_x1, grrc_y1 )
            # cv2.circle(img_copy, (pixel4earX, pixel4earY), 2, color=(0, 255, 0), thickness=-5)
            ear4degree = np.rad2deg(np.arctan2(left[0] - pixel4earX, left[1] - pixel4earY))
            # ear4degree = np.rad2deg(np.arctan2(ortaleftx - pixel4earX, ortalefty - pixel4earY))
            # Yüzdeki koordinatlar ve saptaki koordinatlardan yararlanılarak sapın resize edileceği oranın bulunması ve uygulanması
            earResizeRateX = (pixel4earX - left[0]) / (-ear_left_file[0] + ear_eye_left_file[0])
            # earResizeRateX = (pixel4earX - ortaleftx) / (-ear_left_file[0] + ear_eye_left_file[0])
            earResizeRate = int(round(earResizeRateX * ear_img_left.shape[1], 0))

            ear_eye_left_X, ear_eye_left_Y, ear_img_left = resized_pixel_locations(ear_img_left, ear_eye_left_X_file, ear_eye_left_Y_file, earResizeRate, ear4degree+90)
            ear_eye_left = (ear_eye_left_X, ear_eye_left_Y)

            ear_img_left = ndimage.rotate(ear_img_left, ear4degree + 90)

            eye_ear_img_left = img_copy[ pixel4earY - ear_eye_left[1]: pixel4earY - ear_eye_left[1] + ear_img_left.shape[0], pixel4earX - ear_eye_left[0]: pixel4earX - ear_eye_left[0] + ear_img_left.shape[1]]
            ear_img_left = blend_transparent(eye_ear_img_left, ear_img_left)
            img_copy[ pixel4earY - ear_eye_left[1]: pixel4earY - ear_eye_left[1] + ear_img_left.shape[0], pixel4earX - ear_eye_left[0]: pixel4earX - ear_eye_left[0] + ear_img_left.shape[1]] = ear_img_left


        elif head_degree < -2:
            pixel4earX, pixel4earY = get_pixel_locations( glasses, mtr[head_degree4file][2], mtr[head_degree4file][3], resize_rate, degree + 90, nose, grrc_x1, grrc_y1 )
            # cv2.circle(img_copy, (pixel4earX, pixel4earY), 2, color=(0, 255, 0), thickness=-5)
            ear4degree = np.rad2deg(np.arctan2(-right[0] + pixel4earX, -right[1] + pixel4earY))
            # ear4degree = np.rad2deg(np.arctan2(-ortarightx + pixel4earX, -ortarighty + pixel4earY))
            # Yüzdeki koordinatlar ve saptaki koordinatlardan yararlanılarak sapın resize edileceği oranın bulunması ve uygulanması
            earResizeRateX = (pixel4earX - right[0]) / (ear_right_file[0] - ear_eye_right_file[0])
            # earResizeRateX = (pixel4earX - ortarightx) / (ear_right_file[0] - ear_eye_right_file[0])
            earResizeRate = int(round(earResizeRateX * ear_img_right.shape[1], 0))

            ear_right_X, ear_right_Y, ear_img_right = resized_pixel_locations(ear_img_right, ear_right_X_file, ear_right_Y_file, earResizeRate, ear4degree+90)
            ear_right = (ear_right_X, ear_right_Y)

            ear_img_right = ndimage.rotate(ear_img_right, ear4degree + 90)

            eye_ear_img_right = img_copy[pixel4earY - ear_right[1]: pixel4earY - ear_right[1] + ear_img_right.shape[0], pixel4earX - ear_right[0]: pixel4earX - ear_right[0] + ear_img_right.shape[1]]
            ear_img_right = blend_transparent(eye_ear_img_right, ear_img_right)
            img_copy[pixel4earY - ear_right[1]: pixel4earY - ear_right[1] + ear_img_right.shape[0], pixel4earX - ear_right[0]: pixel4earX - ear_right[0] + ear_img_right.shape[1]] = ear_img_right


        cv2.imshow('Output', img_copy)

    except Exception as e:
        print(e)
        # print(e.__cause__)
        # print(e.__class__)
        # print(e.__context__)
        # print(e.__doc__)
        # print(e.__reduce__())
        # print(e.__dict__)
        # print(e.__suppress_context__)
        # print(e.__dir__())
        # print(e.__hash__())
        # print(e.__str__())
        # print(e.__traceback__)
        # print(e.__repr__())
        # print(e.__reduce_ex__())
        # print(e.__reduce__())
        # print(e.__str__())
        # print(e.__getattribute__())
        # print(e.with_traceback())
        cv2.imshow('Output', img_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break