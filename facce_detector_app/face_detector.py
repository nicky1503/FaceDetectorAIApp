import cv2

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Load pre-trained on face frontals from opencv (haar cascade algorithm)

    trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Convert image to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangle
    for face in face_coordinates:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print((x, y), (x+w, y+h))

    # Display the image
    cv2.imshow("Face Detector", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


