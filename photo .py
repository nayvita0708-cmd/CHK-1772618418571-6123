import cv2

# Load face cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load image
img = cv2.imread("image.jpg")   # 🔹 Put your image name here

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangle around faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Show output image
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()