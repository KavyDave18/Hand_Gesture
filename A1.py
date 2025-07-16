import cv2

s = input("Enter image Location:")
img = cv2.imread('images/'+s)
if img is not None:
    print("Image loaded successfully!")
    choise=int(input("Enter 1 for Grayscale, 2 for RGB "))
    if choise == 1:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    elif choise == 2:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    else:
        print("Invalid choice")
    
    choise2 = int(input("Enter 1 to save image . Enter 2 to show image: "))
    if choise2 == 1:
        save_path = input("Enter the path to save the image:")
        cv2.imwrite(save_path, img)
    elif choise2 == 2:
        cv2.imshow("Image" , img)
        cv2.waitKey(0)
        cv2.closeAllWindows()
    else:
        print("Invalid choice")
else:
    print("Image not found!")
    exit(1)