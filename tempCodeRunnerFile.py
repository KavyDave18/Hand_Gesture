img = cv2.imread("images/image.png")
cv2.imshow("Output",img)
h , c ,v = img.shape
print(f"Height: {h}, Width: {c}, Channels: {v}")
cv2.waitKey(0)
cv2.waitKey(1) & 0xFF == ord('q')