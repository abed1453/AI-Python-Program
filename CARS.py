import cv2

cap = cv2.VideoCapture("video.mp4")

car_cascade = cv2.CascadeClassifier("cars.xml")

while cap.isOpened():
  ret, frame = cap.read()
  if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in cars:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
      cv2.putText(frame, "Car", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Car Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
      break
  else:
    break

cap.release()
cv2.destroyAllWindows()
