import cv2
import numpy as np



img1 = cv2.imread('F:\Bernard\epreuves\orch2-2-2.jpg')
#img1 = cv2.resize(img1, (0,0), fx=0.9, fy=0.9)

mask = np.zeros(img1.shape[:2], dtype=np.uint8)
bgdmodel = np.zeros((1, 13*5), np.float64)   # Temporary array for the background model
fgdmodel = np.zeros((1, 13*5), np.float64)   # Temporary arrays for the foreground model
#a=np.zeros((50000,50000), dtype=np.uint8)
#exit()
cv2.grabCut(img1,
            mask,
            (1000,1000, 100,100),#img1.shape[1]-2000, img1.shape[0]-2000),
            bgdmodel,
            fgdmodel,
            1,
            cv2.GC_INIT_WITH_RECT)
