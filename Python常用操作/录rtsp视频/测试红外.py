import cv2

img = cv2.imread('D:\\Zengh\\vlcsnap-2023-10-13-16h11m11s423.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('D:\\Zengh\\gray.jpg', img)
