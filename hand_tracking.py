import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize a white canvas to draw on
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Initialize Kalman Filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

# Previous point
prev_point = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # mirror image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Get index fingertip coordinates
        h, w, _ = frame.shape
        index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)

        # Kalman filter update
        measurement = np.array([[np.float32(x_index)], [np.float32(y_index)]])
        kalman.correct(measurement)
        predicted = kalman.predict()
        x_kalman, y_kalman = int(predicted[0]), int(predicted[1])

        # Draw if making the "drawing gesture" (index up, middle down)
        index_finger_up = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_finger_up = handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

        if index_finger_up and not middle_finger_up:
            if prev_point is not None:
                cv2.line(canvas, prev_point, (x_kalman, y_kalman), (0, 0, 255), 5)
            prev_point = (x_kalman, y_kalman)
        else:
            prev_point = None

        # "Clear canvas gesture" (make a fist: all fingers down)
        fingers_up = [
            handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            handLms.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < handLms.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
            handLms.landmark[mp_hands.HandLandmark.PINKY_TIP].y < handLms.landmark[mp_hands.HandLandmark.PINKY_PIP].y
        ]
        if not any(fingers_up):
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # clear canvas

    # Overlay canvas on frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Kalman Hand Drawing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


