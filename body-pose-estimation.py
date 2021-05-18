import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic



# Video Properties
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(1)  # Secondary Camera
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

mp_drawing.DrawingSpec((0,255,0), 2, 2)


with mp_holistic.Holistic(min_tracking_confidence=.5, min_detection_confidence=.5) as holistic:

    while True:
        sucess, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = holistic.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Face Landmarks
        mp_drawing.draw_landmarks(img, result.face_landmarks, mp_holistic.FACE_CONNECTIONS, mp_drawing.DrawingSpec(color=(36, 202, 249), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(36, 202, 249), thickness=1, circle_radius=1))

        # Right hand
        mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Left Hand
        mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Pose Detections
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)



        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

