import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/hinh2.png")
cv2.imshow("Original", image)
blurred = cv2.medianBlur(image, 5)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()  # Cai dat ham detector cham (ham loc, lay cham)

# Filter by Inertia
params.filterByInertia = True  # Chon xai cong cu Filter Inertia
params.minInertiaRatio = 0.7  # Chon min do tron cua hinh la 0.7

# Create a detector with the parameters (Tao ct con detect)
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

# Binary Picture For Detect Black Pips Of Dice
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 131, 15)
cv2.imshow("binary", binary)
# Detect black blobs.
keypoints = detector.detect(binary)

tong_cham = len(keypoints)  # so luong cham den

tam = np.zeros(shape=(len(keypoints), 2))  # Tim tam cua nhung cham
for i in range(len(keypoints)):
    tam[i][0] = keypoints[i].pt[0]  # Toa do x
for i in range(len(keypoints)):
    tam[i][1] = keypoints[i].pt[1]  # Toa do y
tam = np.float32(tam)  # Chuyen float 32bit
# Su dung Kmean
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 7
ret, label, center = cv2.kmeans(tam, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

count = np.zeros((7, 1), dtype=int)  # Ma tran count luu so cham den cung 1 vung
font = cv2.FONT_HERSHEY_SIMPLEX
for i in label:  # Ma tran chua so vung cua tung xuc xuat
    count[i] = count[i] + 1  # count chu so cham trong tung vung
for i in range(len(count)):
    cv2.putText(image, str(count[i]), (center[i][0], center[i][1]), font, 0.5, (0, 0, 255), 1)

# Đêm số đốm trắng
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 7)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
erode = cv2.erode(binary, kernel, iterations=1)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
dilate = cv2.dilate(erode, kernel2, iterations=1)
dilate = cv2.dilate(dilate, kernel2, iterations=1)
binary = ~dilate
cv2.imshow("bin1", binary)

keypoints = detector.detect(binary)
tong_cham += len(keypoints)

tam = np.zeros(shape=(len(keypoints), 2))
for i in range(len(keypoints)):
    tam[i][0] = keypoints[i].pt[0]
for i in range(len(keypoints)):
    tam[i][1] = keypoints[i].pt[1]

tam = np.float32(tam)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 4
ret, label, center = cv2.kmeans(tam, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
count = np.zeros((4, 1), dtype=int)
font = cv2.FONT_HERSHEY_SIMPLEX
for i in label:
    count[i] = count[i] + 1
for i in range(len(count)):
    cv2.putText(image, str(count[i]), (center[i][0], center[i][1]), font, 0.5, (0, 0, 255), 1)

cv2.putText(image, 'Tong cham:' + str(tong_cham), (300, 50), font, 1, (0, 0, 255), 1)
cv2.imshow("ket qua", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
