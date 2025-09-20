# fist_control.py
# Controls:
#   • Make a fist to move the mouse cursor.
#   • Open your hand to stop moving the cursor.

import sys
import time
import math
import cv2
import mediapipe as mp

# Platform-specific mouse control (macOS Quartz in this case)
try:
    from Quartz.CoreGraphics import (
        CGEventPost, kCGHIDEventTap, CGWarpMouseCursorPosition
    )
    from Quartz import CGMainDisplayID, CGDisplayBounds
except ImportError:
    print("This script requires pyobjc-framework-Quartz for mouse control on macOS.")
    print("Install with: pip install pyobjc-framework-Quartz")
    sys.exit(1)

# ---- MediaPipe Hands Setup ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------- macOS Mouse Actions ----------
def get_screen_size():
    """Gets the main display's size in pixels."""
    b = CGDisplayBounds(CGMainDisplayID())
    return int(b.size.width), int(b.size.height)

def move_cursor_norm(nx, ny):
    """
    Moves the cursor to a normalized position (0,0 to 1,1) on the screen.
    """
    sw, sh = get_screen_size()
    # Clamp values to prevent moving cursor off-screen
    nx = min(max(nx, 0.01), 0.99)
    ny = min(max(ny, 0.01), 0.99)
    x = int(nx * sw)
    y = int(ny * sh)
    CGWarpMouseCursorPosition((x, y))

# ---------- Helpers & Tuning ----------
# Landmark indices for fingertips (excluding thumb)
FINGER_TIPS = [8, 12, 16, 20] # Index, Middle, Ring, Pinky

# Fist thresholds are based on the average distance of fingertips to the wrist,
# normalized by "palm scale" (wrist to middle finger base).
# These values may need tuning for your specific hand size and camera setup.
FIST_ON_THR     = 0.38  # Lower value = tighter fist needed
FIST_OFF_THR    = 0.48  # Higher value = more open hand needed to release
FIST_EMA_A      = 0.35  # Smoothing factor for fist detection
FIST_DEBOUNCE   = 80    # ms to wait before changing state

# Cursor smoothing
CURSOR_EASE     = 0.35  # 0..1 (higher = snappier cursor)

class EMA:
    """Exponential Moving Average for smoothing values."""
    def __init__(self, a=0.5, init=None):
        self.a = a
        self.v = init

    def update(self, x):
        if self.v is None:
            self.v = x
        else:
            self.v = self.a * x + (1 - self.a) * self.v
        return self.v

def palm_scale(lms):
    """
    Calculates a rotation-robust hand scale using the distance
    between the wrist (0) and the base of the middle finger (9).
    """
    dx = lms[0].x - lms[9].x
    dy = lms[0].y - lms[9].y
    return max(1e-6, math.hypot(dx, dy))

class StableFist:
    """Smoothed, hysteretic, and debounced fist detector."""
    def __init__(self, on_thr=0.45, off_thr=0.55, ema_a=0.35, debounce_ms=80):
        assert on_thr < off_thr, "ON threshold must be less than OFF threshold"
        self.on_thr = on_thr
        self.off_thr = off_thr
        self.ema_a = ema_a
        self.debounce = debounce_ms / 1000.0
        self.v = None  # The smoothed "fist-ness" value
        self.state = False
        self.last_change = 0.0

    def update(self, lms, now):
        """
        Updates the fist state based on new hand landmarks.
        Returns (is_fist: bool, raw_value: float).
        """
        s = palm_scale(lms)
        wrist_lm = lms[0]
        
        # Calculate the average distance from the 4 fingertips to the wrist
        distances = [math.hypot(lms[tip].x - wrist_lm.x, lms[tip].y - wrist_lm.y) for tip in FINGER_TIPS]
        avg_dist = sum(distances) / len(distances)
        
        # Normalize the distance by the palm scale
        d = avg_dist / s

        # Smooth the value using EMA
        self.v = d if self.v is None else (self.ema_a * d + (1 - self.ema_a) * self.v)

        # Apply hysteresis and debouncing
        target_state = self.state
        if not self.state and self.v < self.on_thr:
            target_state = True
        elif self.state and self.v > self.off_thr:
            target_state = False

        if target_state != self.state and (now - self.last_change) >= self.debounce:
            self.state = target_state
            self.last_change = now

        return self.state, self.v

def init_system():
    """Initializes the camera, MediaPipe Hands, and the fist detector."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not found.")

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    fist_detector = StableFist(FIST_ON_THR, FIST_OFF_THR, FIST_EMA_A, FIST_DEBOUNCE)

    return cap, hands, fist_detector


# ---------- Main Loop ----------
def main(show_camera=False):
    cap, hands, fist_detector = init_system()

    # EMA for smoothing the cursor's target position
    cursor_x = EMA(CURSOR_EASE)
    cursor_y = EMA(CURSOR_EASE)

    print("Make a fist to move the mouse. Press 'q' in the camera window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hud_text = "READY"
        hud_color = (0, 180, 255) # Orange
        now = time.time()

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            if show_camera:
                mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # --- Fist Detection ---
            is_fist, fist_val = fist_detector.update(landmarks, now)

            # --- Cursor Control ---
            # Calculate a central point of the hand for cursor positioning
            all_xs = [lm.x for lm in landmarks]
            all_ys = [lm.y for lm in landmarks]
            centroid_x = sum(all_xs) / len(all_xs)
            centroid_y = sum(all_ys) / len(all_ys)

            if is_fist:
                # If a fist is detected, update and move the cursor
                smoothed_x = cursor_x.update(centroid_x)
                smoothed_y = cursor_y.update(centroid_y)
                move_cursor_norm(smoothed_x, smoothed_y)
                
                hud_text = f"FIST: {fist_val:.2f}"
                hud_color = (0, 255, 0) # Green
                
                if show_camera:
                    # Draw a circle at the cursor position for feedback
                    cv2.circle(frame, (int(smoothed_x * w), int(smoothed_y * h)), 10, (0, 255, 255), 2)
            else:
                hud_text = f"OPEN: {fist_val:.2f}"


        # --- Display camera feed and HUD ---
        if show_camera:
            cv2.putText(frame, hud_text, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
            cv2.imshow("Fist Control (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_camera:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Pass 'camera' as a command-line argument to see the video feed
    show_preview = "camera" in sys.argv
    main(show_camera=show_preview)
