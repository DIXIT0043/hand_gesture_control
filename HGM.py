import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

labels_dict = {0: 'Move', 1: 'Free', 2: 'Click'}

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    H, W, _ = frame.shape

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        first_hand_landmarks = results.multi_hand_landmarks[0]  # Get landmarks of the first detected hand only
        mp_drawing.draw_landmarks(
            frame,  # image to draw
            first_hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        data_aux = []
        x_ = []
        y_ = []

        for i in range(len(first_hand_landmarks.landmark)):
            x = first_hand_landmarks.landmark[i].x
            y = first_hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        for i in range(len(first_hand_landmarks.landmark)):
            x = first_hand_landmarks.landmark[i].x
            y = first_hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])

        predicted_label = labels_dict.get(int(prediction[0]), 'Unknown')  # Get the predicted label or 'Unknown' if not found

        # Control mouse based on predicted_label
        if predicted_label == 'Move':
            # Example: move mouse to a specific location on the screen
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            pyautogui.moveTo(x_center * pyautogui.size()[0] / W, y_center * pyautogui.size()[1] / H)
        elif predicted_label == 'Click':
            # Example: click the left mouse button
            pyautogui.click()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()