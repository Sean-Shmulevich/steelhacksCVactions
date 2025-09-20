import cv2
import mediapipe as mp
import time

# Init mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

prev_x = None
prev_time = None

# Swipe detection state
swipe_active = False
swipe_direction = None
velocity_threshold = 1.0   # how fast must it move
idle_timeout = 0.3         # seconds of inactivity to consider swipe ended

last_move_time = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        x = hand.landmark[0].x  # wrist normalized [0,1]
        t = time.time()

        if prev_x is not None and prev_time is not None:
            dt = t - prev_time
            dx = x - prev_x
            velocity = dx / dt

            # Detect start of a swipe
            if not swipe_active:
                if velocity > velocity_threshold:
                    swipe_active = True
                    swipe_direction = "LEFT → RIGHT"
                    last_move_time = t
                elif velocity < -velocity_threshold:
                    swipe_active = True
                    swipe_direction = "RIGHT → LEFT"
                    last_move_time = t

            # If swipe is active, update timestamp
            elif swipe_active:
                if abs(velocity) > 0.2:  # still moving a bit
                    last_move_time = t

                # Check for inactivity to end swipe
                if t - last_move_time > idle_timeout:
                    print(f"Swipe detected: {swipe_direction}")
                    swipe_active = False
                    swipe_direction = None

        prev_x, prev_time = x, t

    cv2.imshow("Hand Swipe Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
