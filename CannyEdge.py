import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('E:\orch2-2-2.png')

plt.subplot(121)
plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

def nothing(x):
    pass

TrackbarsImg = np.zeros((300,512,3), np.uint8)
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)

cv2.createTrackbar('R','Trackbars',0,255,nothing)
cv2.createTrackbar('G','Trackbars',0,255,nothing)
cv2.createTrackbar('B','Trackbars',0,255,nothing)

switch = '0:OFF\n1:ON'
cv2.createTrackbar(switch, 'Trackbars',0,1,nothing)




while(1):
    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R', 'Trackbars')
    g = cv2.getTrackbarPos('G', 'Trackbars')
    b = cv2.getTrackbarPos('B', 'Trackbars')
    s = cv2.getTrackbarPos(switch, 'Trackbars')
    edges = cv2.Canny(img, r, g, L2gradient=True, apertureSize=3)

    cv2.imshow('Trackbars',cv2.resize(edges, None, fx=0.5, fy=0.5) )
    cv2.resizeWindow("Trackbars", 1000, 800)
    k = cv2.waitKey(1000) & 0xFF
    if k == 27:
        break



    if s == 0:
        TrackbarsImg[:] = 0
    else:
        TrackbarsImg[:] = [b,g,r]

cv2.imshow('Trackbars', TrackbarsImg)



contours=cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


cv2.drawContours(edges, contours[0], -1, (255,255,255), 10)

plt.subplot(122)
plt.imshow(edges)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()