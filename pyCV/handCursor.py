import cv2
import mediapipe as mp
import pyautogui

# Init mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

# Store smoothed cursor position
smooth_x, smooth_y = screen_w // 2, screen_h // 2
alpha = 0.2  # smoothing factor (0 = no movement, 1 = instant movement)

while True:
    success, image = cap.read()
    if not success:
        break

    # Flip for natural feel
    image = cv2.flip(image, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Index finger tip = landmark #8
            x = hand_landmarks.landmark[8].x * screen_w
            y = hand_landmarks.landmark[8].y * screen_h

            # Apply exponential smoothing
            smooth_x = smooth_x * (1 - alpha) + x * alpha
            smooth_y = smooth_y * (1 - alpha) + y * alpha

            pyautogui.moveTo(smooth_x, smooth_y)  # move mouse smoothly

            # Optional: draw landmarks for debugging
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking Mouse", image)
    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
