# pinch_sequence_detect_timeout.py

import time
import sys
import math
import cv2
import mediapipe as mp

from Quartz.CoreGraphics import (
    CGEventCreateScrollWheelEvent, CGEventPost, kCGHIDEventTap,
    kCGScrollEventUnitLine, CGWarpMouseCursorPosition
)
from Quartz import CGMainDisplayID, CGDisplayBounds

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

PINCH_ON_THR    = 0.38
PINCH_OFF_THR   = 0.48
PINCH_EMA_A     = 0.35
PINCH_DEBOUNCE  = 80   # ms

# Max time allowed between first tight pinch and second tight pinch
SEQ_TIMEOUT     = 2.5   # seconds

class EMA:
    def __init__(self, a=0.5, init=None): self.a,self.v=a,init
    def update(self, x):
        self.v = x if self.v is None else (self.a*x + (1-self.a)*self.v)
        return self.v

def palm_scale(lms):
    dx = lms[0].x - lms[9].x
    dy = lms[0].y - lms[9].y
    return max(1e-6, math.hypot(dx, dy))

class StablePinch:
    def __init__(self, on_thr=0.38, off_thr=0.48, ema_a=0.35, debounce_ms=80):
        assert on_thr < off_thr
        self.on_thr, self.off_thr = on_thr, off_thr
        self.ema_a = ema_a
        self.debounce = debounce_ms / 1000.0
        self.v = None
        self.state = False
        self.last_change = 0.0

    def update(self, lms, now):
        s = palm_scale(lms)
        d = math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y) / s
        self.v = d if self.v is None else (self.ema_a*d + (1-self.ema_a)*self.v)

        target = self.state
        if not self.state and self.v < self.on_thr:
            target = True
        elif self.state and self.v > self.off_thr:
            target = False

        if target != self.state and (now - self.last_change) >= self.debounce:
            self.state = target
            self.last_change = now

        return self.state, self.v

def init_system():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not found.")
    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    pinch = StablePinch(PINCH_ON_THR, PINCH_OFF_THR, PINCH_EMA_A, PINCH_DEBOUNCE)
    return cap, hands, pinch

# ---------- Main loop ----------
def main(camera=False):
    cap, hands, pinch = init_system()

    print("Pinch sequence tracker: Tight → Loose → Tight (with timeout)")

    # Sequence state
    seq_state = 0
    seq_start_time = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        now = time.time()

        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0].landmark
            pinched, pinch_val = pinch.update(lms, now)

            # Timeout check
            if seq_state > 0 and seq_start_time and (now - seq_start_time > SEQ_TIMEOUT):
                seq_state = 0
                seq_start_time = None

            # --- Sequence detector ---
            if seq_state == 0 and pinched:
                seq_state = 1
                seq_start_time = now
            elif seq_state == 1 and not pinched:
                seq_state = 2
            elif seq_state == 2 and pinched:
                print("Second tight pinch detected — SEQUENCE COMPLETE")
                seq_state = 0
                seq_start_time = None

            if camera:
                mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"Pinch:{pinched} val={pinch_val:.2f}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if pinched else (0,0,255), 2)
                cv2.imshow("Pinch Sequence", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("camera" in sys.argv)