import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
import cv2


mtcnn = MTCNN(keep_all=False)


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to PIL Image for MTCNN
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Detect faces
        boxes, _ = mtcnn.detect(img)

        # Draw rectangles around detected faces
        if boxes is not None:
            for box in boxes:
                # Draw rectangle on the frame
                frame = cv2.rectangle(frame,
                                      (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])),
                                      (255, 0, 0),
                                      2)

                # Crop the face using the bounding box
                cropped_face = img.crop(box)
                cropped_face = cv2.cvtColor(np.array(cropped_face), cv2.COLOR_RGB2BGR)
                cv2.imshow("Cropped", cropped_face)  # Display the cropped face

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()