import numpy
import cv2

img1 = numpy.zeros((300,300), dtype="uint8")
img2 = numpy.zeros((300,300), dtype="uint8")

img1_circle = cv2.circle(img1 , (150,150), 150 , 255 , -1)
img2_rectangle = cv2.rectangle(img2 , (150,150) , (250,250) ,255 , -1)

bitwise_and = cv2.bitwise_and(img1 , img2)
bitwise_or  = cv2.bitwise_or(img1 , img2)
bitwise_not = cv2.bitwise_not(img1)

cv2.imshow("image" , bitwise_not)
cv2.imshow("image2" , img1_circle)
cv2.imshow("imag3e" , img2_rectangle)
cv2.imshow("image45" , bitwise_and)
cv2.imshow("imag5e" , bitwise_or)

cv2.waitKey(0)
cv2.destroyAllWindows()