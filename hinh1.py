import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/hinh1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(src = gray, ksize = (5, 5), sigmaX = 0)
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()  # Cai dat ham detector cham (ham loc, lay cham)

# Filter by Inertia
params.filterByArea = True
params.minArea = 3
params.maxArea = 128*128
params.filterByCircularity = False
params.filterByColor = False
params.filterByConvexity = False
params.filterByInertia = False  # Chon xai cong cu Filter Inertia
params.minInertiaRatio = 0.7  # Chon min do tron cua hinh la 0.7
params.minThreshold = int(0.5*255)
params.maxThreshold = int(0.95*255)
params.thresholdStep = 10

# Create a detector with the parameters (Tao ct con detect)
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

# Binary Picture For Detect Black Pips Of Dice
#binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 131, 15)
(_,binary) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", binary)
# Detect black blobs.
keypoints = detector.detect(binary)
print(keypoints)
tong_cham = len(keypoints)  # so luong cham den

tam = np.zeros(shape=(len(keypoints), 2))  # Tim tam cua nhung cham
for i in range(len(keypoints)):
    tam[i][0] = keypoints[i].pt[0]  # Toa do x
for i in range(len(keypoints)):
    tam[i][1] = keypoints[i].pt[1]  # Toa do y
tam = np.float32(tam)  # Chuyen float 32bit
# Su dung Kmean
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 6
ret, label, center = cv2.kmeans(tam, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
print(tam)
print(label)
count = np.zeros((6, 1), dtype=int)  # Ma tran count luu so cham den cung 1 vung
font = cv2.FONT_HERSHEY_SIMPLEX
for i in label:  # Ma tran chua so vung cua tung xuc xuat
    count[i] = count[i] + 1  # count chu so cham trong tung vung
for i in range(len(count)):
    cv2.putText(image, str(count[i]), (center[i][0], center[i][1]), font, 0.5, (0, 0, 255), 1)


cv2.putText(image, 'Tong cham:' + str(tong_cham), (300, 50), font, 1, (0, 0, 255), 1)
cv2.imshow("ket qua", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
