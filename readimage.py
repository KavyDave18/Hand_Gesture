import numpy as np
import cv2

# TO READ IMAGE AND DISPLAY IT
# img = cv2.imread("images/image.png")
# cv2.imshow("Output",img)
# h , c ,v = img.shape
# print(f"Height: {h}, Width: {c}, Channels: {v}")
# cv2.waitKey(0)
# cv2.waitKey(1) & 0xFF == ord('q')

# TO READ AND DISPLAY VIDEO
# vid=cv2.VideoCapture("video/video.mp4")
# while True:
#     succesor , img = vid.read()
#     cv2.imshow("Video",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# TO MAKE IMAGE INTO GRAYSCALEIMG
# img = cv2.imread("images/image.png")
# imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Image",imgray)
# cv2.waitKey(0) & 0xFF == ord('s')

# BLUR IMAGE
# img = cv2.imread("images/image.png")
# blurimg = cv2.GaussianBlur(img,(17,17),300)
# cv2.imshow("Blurred img",blurimg)
# cv2.waitKey(0) & 0xFF == ord('s')

# FIND EDGER DITECTOR
# img = cv2.imread("images/image.png")
# edgimg = cv2.Canny(img,10,10)
# cv2.imshow("Edge Image",edgimg)
# cv2.waitKey(0) & 0xFF == ord('s')

#DILATE IMAGE (Make egdes thicker)
# img = cv2.imread("imgaes/image.png")
# imgDilate = cv2.dilate(edgimg,(1,1),iterations=3)
# cv2.imshow("Dilated Image",imgDilate)
# cv2.waitKey(0) & OxFF == ord('s')

#Erosion Image (Make edges thinner)
# Erosedimg = cv2.erode(edgimg,(7,7),iterations=1)
# cv2.imshow("Erosed Image",Erosedimg)
# cv2.waitKey(0) & 0xFF == ord('s')

#RESIZEIG IMAGE
# img = cv2.imread("images/image.png")
# resized_img = cv2.resize(img , (1000 ,1000))
# print(img.shape)
# print(resized_img.shape)
# cv2.imshow("resized",resized_img)
# cv2.waitKey(0) & 0xFF == ord('s')

#CROP IMAGE
# img = cv2.imread("images/image.png")
# crop_img = img[0:1000, 0:1000]  
# cv2.imshow("Cropped Image" , crop_img)
# cv2.waitKey(0) & 0xFF == ord('s')

# ROTATING IMAGE
# img = cv2.imread('images/image.png')
# h, w, c = img.shape
# print(f"height : {h}, width : {w}, channels : {c}")
# center = (w//2 , h//2)
# angle= 90
# M = cv2.getRotationMatrix2D(center , angle , 1.0)
# img_rotated = cv2.warpAffine(img , M , (w,h))
# cv2.imshow("Rotated Image" , img_rotated)
# cv2.waitKey(0)

#FLIP IMAGE
img = cv2.imread('images/person.png')
img_flip_h = cv2.flip(img , 1)
img_flip_v = cv2.flip(img , 0)
img_flip_both = cv2.flip(img , -1)
cv2.imshow('Original Image' , img)
cv2.imshow('Flippef Image Horizontal' , img_flip_h)
cv2.imshow('Flippef Image Vertical' , img_flip_v)
cv2.imshow('Flippef Image Both' , img_flip_both)
cv2.waitKey(0)

# DRAWING SHAPES ON IMAGE
# img = np.zeros((512,512,3),np.uint8)
# img[200:400,100:300]=255,0,0 #Blue Color
# cv2.imshow('Black Image',img)
# # cv2.waitKey(0) & 0xFF == ord('s')
# cv2.line(img,(0,0),(290,290),(0,255,255),3)
# cv2.line(img,(290,290),(300,300),(0,255,0),15)
# cv2.rectangle(img,(0,0),(200,150),(255,0,0),cv2.FILLED)
# cv2.circle(img,(400,400),30,(124,234,180),-1) #-1 means filled the circle
# cv2.putText(img,'OpenCV',(250,250),cv2.FONT_HERSHEY_COMPLEX,1,(135,55,200),2)
# cv2.imshow('Line',img)
# cv2.waitKey(0) & 0xFF == ord('s')

#WRAP PERSPECTIVE
# img = cv2.imread("images/image.png")
# width,height = 250,350;
# pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
# pts1 = np.float32([[456,677],[250,400],[562,766],[629,900]])
# matrix = cv2.getPerspectiveTransform(pts1,pts2)
# imgOutPut = cv2.warpPerspective(img,matrix,(width,height))
# cv2.imshow("WrapImage",imgOutPut)
# cv2.waitKey(0) & 0xFF == ord('s')

#JOINING IMAGES
# img = cv2.imread("images/image.png")
# img2 = cv2.imread("images/image2.png")
# ver = np.vstack((img,img2))
# hor = np.hstack((img,img2))
# cv2.imshow("VStack",ver)
# cv2.waitKey(0) & 0xFF == ord('s')

#COLOR DETACTION

# def empty(a):
#     pass

# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 640, 240)

# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

# while True:
#     img = cv2.imread("images/image.png")
#     if img is None:
#         print(" Image not found!")
#         break

#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
#     v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

#     lower = np.array([h_min, s_min, v_min])
#     upper = np.array([h_max, s_max, v_max])

#     mask = cv2.inRange(imgHSV, lower, upper)
#     result = cv2.bitwise_and(img, img, mask=mask)

#     cv2.imshow("Result", result)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key != 255:
#         print(f"H: {h_min}-{h_max}, S: {s_min}-{s_max}, V: {v_min}-{v_max}")

# cv2.destroyAllWindows()

#SHAPE DETACTION

# def getCounters(img):
#     contours,hirerchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         print(area)

# img = cv2.imread("images/shapes.png")
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray,(3,3),5)
# imgCanny = cv2.Canny(imgBlur,50,50)
# imgThres = cv2.threshold(imgCanny,50,255,cv2.THRESH_BINARY)[1]
# imgBlank = np.zeros_like(img)
# getCounters(imgThres)
# # cv2.imshow("Blank",imgBlank)   
# # cv2.imshow("Gray Image",imgGray)
# # cv2.imshow("Blured Image",imgBlur)
# # cv2.imshow("Canny",imgCanny0)
# cv2.imshow("Orignal",img)
# cv2.waitKey (0) & 0xFF == ord('s')
