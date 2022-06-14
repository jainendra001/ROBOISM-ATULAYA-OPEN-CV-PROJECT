import cv2 as cv
import cv2
import math
import numpy as np

image = cv.imread("C://Users//MY PC//Desktop//CVtask.jpg")
image = cv.resize(image, (877, 620))
imgo = image.copy()
# convert image into greyscale mode
color = {'green': [ 79, 209, 146 ], 'orange': [ 9, 127, 240 ], 'white': [ 210, 222, 228 ], 'black': [ 0, 0, 0 ]}
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# find threshold of the image
_, thrash = cv.threshold(gray_image, 240, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thrash, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print("initial=", len(contours))
for contour in contours:
    shape = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
    x_cor = shape.ravel() [ 0 ]
    y_cor = shape.ravel() [ 1 ]

    if len(shape) == 4:
        # shape cordinates
        x, y, w, h = cv.boundingRect(shape)

        # width:height
        aspectRatio = float(w) / h
        if aspectRatio >= 0.9 and aspectRatio <= 1.05:
            M = cv.moments(contour)
            if M [ 'm00' ] != 0.0:
                x = int(M [ 'm10' ] / M [ 'm00' ])
                y = int(M [ 'm01' ] / M [ 'm00' ])
            print(image [ y, x, : ])
            for i in color.keys():
                d = np.array(color [ i ])
                d.reshape((3,))
                # if (d == image[y, x, :]).any():
                # cv.putText(image, i, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            cv.circle(image, (x, y), 3, (220, 0, 0), 1)
            cv.drawContours(image, [ shape ], 0, (0, 0, 0), -1)

            # cv.putText(image, "Square", (x_cor, y_cor), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))

print(image.shape)
cv.imshow("Shape", image)
cv.waitKey(0)
cv.destroyAllWindows()

LMAO = cv.imread("C://Users//MY PC//Desktop//ATULIYA//LMAO.jpg")
HaHa = cv.imread("C://Users//MY PC//Desktop//ATULIYA//HaHa.jpg")
Ha = cv.imread("C://Users//MY PC//Desktop//ATULIYA//Ha.jpg")
XD = cv.imread("C://Users//MY PC//Desktop//ATULIYA//XD.jpg")

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
parameters = cv2.aruco.DetectorParameters_create()

lst = [ None ] * 5
c, id, r = cv2.aruco.detectMarkers(Ha, dictionary, parameters=parameters)
lst [ id [ 0 ] [ 0 ] ] = Ha
c, id, r = cv2.aruco.detectMarkers(HaHa, dictionary, parameters=parameters)
lst [ id [ 0 ] [ 0 ] ] = HaHa
c, id, r = cv2.aruco.detectMarkers(LMAO, dictionary, parameters=parameters)
lst [ id [ 0 ] [ 0 ] ] = LMAO
c, id, r = cv2.aruco.detectMarkers(XD, dictionary, parameters=parameters)
lst [ id [ 0 ] [ 0 ] ] = XD


# HERE I STARTED
def Centre(v1, v2):
    x = (v1 [ 0 ] + v2 [ 0 ]) / 2
    x = int(x)
    y = (v1 [ 1 ] + v2 [ 1 ]) / 2
    y = int(y)
    V = [ x, y ]
    return V


def DISTANCE(V1, V2):
    x = V1 [ 0 ] - V2 [ 0 ]
    y = V1 [ 1 ] - V2 [ 1 ]
    D = math.sqrt(x ** 2 + y ** 2)
    return D


def arangle(image):
    # cv.imshow("image",image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    c, i, r = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
    p = [ ]
    o = None
    for j in range(len(c)):
        for C in c [ j ]:
            p.append((C [ 0 ]).tolist())
            p.append((C [ 1 ]).tolist())
            p.append((C [ 2 ]).tolist())
            p.append((C [ 3 ]).tolist())
            Cx = C [ 0 ] [ 0 ] + C [ 2 ] [ 0 ]
            Cx = Cx / 2
            Cx = int(Cx)
            Cy = C [ 0 ] [ 1 ] + C [ 2 ] [ 1 ]
            Cy = Cy / 2
            Cy = int(Cy)
            mx = C [ 0 ] [ 0 ] + C [ 3 ] [ 0 ]
            mx = mx / 2
            my = C [ 0 ] [ 1 ] + C [ 3 ] [ 1 ]
            my = my / 2
            dy = Cy - my
            dx = Cx - mx
            o = math.degrees(math.atan((dy / dx)))
            if (dx < 0 and dy > 0):
                o = o + 180
            elif (dx < 0 and dy <= 0):
                o = o + 180
            elif (dy < 0 and dx > 0):
                o = 360 + o
    print(p [ 0 ])
    C = Centre(p [ 0 ], p [ 2 ])
    print(C)
    return o, C


for i in range(1, 5):
    a = lst [ i ]
    arangle(a)


def arucord(image, o, C):
    """"(height, width) = image.shape[ :2 ]
    M = cv.getRotationMatrix2D((C[ 0 ],C[ 1 ]), o, 1.0)
    rotated = cv.warpAffine(image, M, (width, height), borderValue=(255, 255, 255))
    return rotated"""
    n = approx.ravel()
    i = 0

    for j in n:
        if (i % 2 == 0):
            x = n[ i ]
            y = n[ i + 1 ]

            # String containing the co-ordinates.
            string = str(x) + " " + str(y)

            if (i == 0):


d = {(210, 222, 228): 4, (0, 0, 0): 3, (9, 127, 240): 2, (79, 209, 146): 1}

for i in range(1, 5):
    o, C = arangle(lst[ i ])
    lst[ i ] = arucord(lst[ i ], o, C)


    def resize(size, image):
        C, i, r = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
        C = C[0][0]
        p1 = list(C [ 0 ])
        p2 = list(C [ 1 ])
        sizear = DISTANCE(p1, p2)
        alp = size / sizear
        sh = image.shape
        sh = list(sh)
        a = int(sh [ 1 ] * alp)
        b = int(sh [ 0 ] * alp)
        resized = cv2.resize(image, (a, b))
        # cv2_imshow(resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return resized


    def resize(size, image):
        r = 100.0 / image.shape [ 1 ]
        dim = (100, int(image.shape [ 0 ] * r))
        # perform the actual resizing of the image and show it
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("resized", resized)
        cv2.waitKey(0)


def boundrect(img):
    c, id, r = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
    c = c [ 0 ] [ 0 ]
    v = [ list(c[0]), list(c [ 1 ]), list(c [ 2 ]), list(c [ 3 ]) ]
    xmin = v [ 0 ] [ 0 ]
    ymin = v [ 0 ] [ 1 ]
    xmax = v [ 0 ] [ 0 ]
    ymax = v [ 0 ] [ 1 ]
    for i in v:
        if (i [ 0 ] < xmin):
            xmin = i [ 0 ]
        if (i [ 0 ] > xmax):
            xmax = i [ 0 ]
        if (i [ 1 ] < ymin):
            ymin = i [ 1 ]
        if (i [ 1 ] > ymax):
            ymax = i [ 1 ]
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    an = img [ ymin:ymax, xmin:xmax ]
    imgcpy = img.copy()
    imgcpy1 = cv2.cvtColor(imgcpy, cv2.COLOR_BGR2GRAY)
    imgcpy1 = cv2.Canny(imgcpy, 30, 150)
    # cv2.imshow("cpy",imgcpy1)
    contoursaruco, hie = cv2.findContours(imgcpy1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for j in contoursaruco:
        imgcpy = cv2.drawContours(imgcpy, [ j ], 0, (0, 0, 0), -1)
    ancpy = imgcpy [ ymin:ymax, xmin:xmax ]
    an = cv2.bitwise_xor(ancpy, an)
    return an


img2 = imgo.copy()
for cnt in contours:
    if cv2.contourArea(cnt) > 1000 and cv2.contourArea(cnt) < 400000:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, peri * 0.0025, True)
        # approx = cv2.approxPolyDP(cnt,epsilon,True)
        if len(approx) == 4:
            p1 = approx [ 0 ] [ 0 ].tolist()
            p2 = approx [ 1 ] [ 0 ].tolist()
            sideo = DISTANCE(p1, p2)
            # print(sideo)
            # cv2.drawContours(img2,[approx],-1,(0,0,0),4)
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.putText(img2,f" {len(cnt)}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            # cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)
            # print(h,w)
            c = Centre([ y, x ], [ (y + h), (x + w) ])
            rgb = img2 [ c [ 0 ] ] [ c [ 1 ] ]
            # print(rgb)
            rgb = tuple(rgb)
            if rgb in d:
                ids = d [ rgb ]
            else:
                continue
            print("id", ids)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            drawn = cv2.drawContours(img2, [ box ], 0, (0, 0, 0), -1)
            cv2.imshow("cont", drawn)
            angle = rect [ 2 ] + 90
            angle = int(angle)
            ang, centre = arangle(lst [ ids ])
            lst [ ids ] = arucord(lst [ ids ], -angle, centre)
            lst [ ids ] = resize(sideo, lst [ ids ])
            toor = boundrect(lst [ ids ])
            cv2.imshow("toor", toor)
            cv2.waitKey(0)
            print("toor", toor.shape)
            # print(toor2.shape)
            shap = toor.shape
            h = shap [ 0 ]
            w = shap [ 0 ]
            print(h, w)
            toor = toor [ 0:h, 0:w ]
            print("toor", toor.shape)
            img2 [ y:y + h, x:x + w ] = cv2.bitwise_or(toor, img2 [ y:y + h, x:x + w ])
            # img2[y:y+h,x:x+w]=toor
            # cv2.imshow("lst",lst[ids])
            print("length=", len(lst))
            print("5", lst [ 0 ])

    ## END - draw rotated rectangle
cv2.imshow("result", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
