import cv2
import numpy as np

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,177,123))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.6:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (x,y,x2,y2) = box.astype("int")

            cv2.rectangle(frame,(x,y),(x2,y2),(0,255,0),2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()