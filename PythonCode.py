import cv2 as cv


img = cv.imread('E:skary.webp')


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


#upper_body_cascade = cv.CascadeClassifier('C:\\Users\\Harshil Zahran\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_upperbody.xml')
#lower_body_cascade = cv.CascadeClassifier('C:\\Users\\Harshil Zahran\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_lowerbody.xml')
#face_cascade = cv.CascadeClassifier('C:\\Users\\Harshil Zahran\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
full_body_cascade = cv.CascadeClassifier('C:\\Users\Harshil Zahran\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_fullbody.xml')


#upper_body_rec = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)


#lower_body_rec = lower_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)


#face_rec = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


full_body_rec = full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)


#for (x, y, w, h) in upper_body_rec:
 #   cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


#for (x, y, w, h) in lower_body_rec:
#    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


#for (x, y, w, h) in face_rec:
#    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


for (x, y, w, h) in full_body_rec:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)


cv.imshow('Object Detection', img)


cv.waitKey(0)
cv.destroyAllWindows()
