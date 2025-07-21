import cv2

img = cv2.imread("images/triangle.png")
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray , 240 , 255 , cv2.THRESH_BINARY_INV)

contours , hierarchy = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img , contours , -1 , (0,0,0) , 5)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt , 0.01*cv2.arcLength(cnt , True), True)
    corners = len(approx)

    if corners == 3:
        shape = "Triangle"  
    elif corners == 4:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            shape = "Square"
        else:
            shape = "Rectangle"
    elif corners == 5:
        shape = "Pentagon"
    elif corners == 6:
        shape = "Hexagon"
    else:
        shape = "Unkown"

    cv2.drawContours(img , [approx] , 0 , (0,255,0),5)
    X = approx.ravel()[0]
    Y = approx.ravel()[1] - 10
    cv2.putText(img , shape , (X,Y) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0,0,255),2)

cv2.imshow("Contours", img)
cv2.waitKey(0)