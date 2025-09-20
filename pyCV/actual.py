# pinch_only_fixed.py
# Controls:
#   • Pinch (thumb + index together) to enter SCROLL mode.
#   • While pinched, move hand up/down to scroll (smooth Quartz CGEvent if available).
#   • Cursor is tethered to hand for spatial feedback.

import os, time, math
import numpy as np
import cv2

# ---- Quartz (scroll + cursor) ----
USE_QUARTZ = True
try:
    from Quartz.CoreGraphics import (
        CGEventCreateScrollWheelEvent, CGEventPost, kCGHIDEventTap,
        kCGScrollEventUnitLine, CGWarpMouseCursorPosition
    )
    from Quartz import CGMainDisplayID, CGDisplayBounds
except Exception:
    USE_QUARTZ = False

# ---- MediaPipe Hands ----
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------- macOS actions ----------
def smooth_scroll_lines(lines: int):
    """positive = up, negative = down (Quartz uses inverted sign)"""
    if lines == 0:
        return
    if USE_QUARTZ:
        ev = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitLine, 1, int(lines))
        CGEventPost(kCGHIDEventTap, ev)
    else:
        # Fallback: burst of Up/Down arrows via AppleScript
        code = 126 if lines > 0 else 125  # Up / Down
        for _ in range(abs(int(lines))):
            os.system(f'osascript -e \'tell application "System Events" to key code {code}\'')

def get_screen_size():
    if USE_QUARTZ:
        b = CGDisplayBounds(CGMainDisplayID())
        return int(b.size.width), int(b.size.height)
    return 1440, 900  # fallback guess

def move_cursor_norm(nx, ny):
    """nx, ny in [0,1] (camera-normalized). Map to screen pixels and warp cursor."""
    sw, sh = get_screen_size()
    nx = min(max(nx, 0.01), 0.99)
    ny = min(max(ny, 0.01), 0.99)
    x = int(nx * sw); y = int(ny * sh)
    if USE_QUARTZ:
        CGWarpMouseCursorPosition((x, y))

# ---------- Helpers & tuning ----------
TIP = [4, 8, 12, 16, 20]     # thumb, index, middle, ring, pinky

# Pinch thresholds now use "palm scale" (wrist↔middle MCP), not hand width.
PINCH_ON_THR    = 0.38
PINCH_OFF_THR   = 0.48
PINCH_EMA_A     = 0.35
PINCH_DEBOUNCE  = 80   # ms

SCROLL_GAIN     = 220.0    # lines per normalized Y (compressed by tanh)
SCROLL_DEADZONE = 0.008    # ignore tiny tremors
SCROLL_CLAMP    = 12       # max lines per tick
TICK_MS         = 33       # ~30 Hz scroll tick

CURSOR_TETHER   = True
CURSOR_EASE     = 0.35     # 0..1 (higher = snappier cursor)

class EMA:
    def __init__(self, a=0.5, init=None): self.a,self.v=a,init
    def update(self, x):
        self.v = x if self.v is None else (self.a*x + (1-self.a)*self.v)
        return self.v

def palm_scale(lms):
    """Rotation-robust scale: distance wrist (0) ↔ middle MCP (9)."""
    dx = lms[0].x - lms[9].x
    dy = lms[0].y - lms[9].y
    return max(1e-6, math.hypot(dx, dy))

class StablePinch:
    """Smoothed, hysteretic, debounced pinch detector."""
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

# ---------- Main loop ----------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found."); return

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # New pinch detector
    pinch = StablePinch(PINCH_ON_THR, PINCH_OFF_THR, PINCH_EMA_A, PINCH_DEBOUNCE)

    pinched = False
    scrolling = False
    baseline_y = None
    last_tick = time.time()

    # EMA for centroid (cursor) + index tip (scroll) + "velocity-ish"
    ema_x = EMA(0.45); ema_y = EMA(0.45)
    cursor_x = EMA(CURSOR_EASE); cursor_y = EMA(CURSOR_EASE)
    idx_y_ema = EMA(0.45)
    vel_ema   = EMA(0.4)   # used for clutching baseline

    print("Pinch to grab; move up/down to scroll. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        hud = []
        now = time.time()

        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0].landmark
            mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # ---- Pinch detection (smoothed + hysteresis + debounce) ----
            pinched, pinch_val = pinch.update(lms, now)

            # ---- Cursor tether: use centroid for spatial feedback ----
            # centroid from wrist + fingertips
            xs = [lms[i].x for i in TIP+[0]]
            ys = [lms[i].y for i in TIP+[0]]
            cx = ema_x.update(sum(xs)/len(xs))
            cy = ema_y.update(sum(ys)/len(ys))
            if CURSOR_TETHER:
                curx = cursor_x.update(cx)
                cury = cursor_y.update(cy)
                move_cursor_norm(curx, cury)
                cv2.circle(frame, (int(curx*w), int(cury*h)), 8, (0,255,255), 2)

            # ---- Scroll signal from index tip Y (not centroid) ----
            iy = idx_y_ema.update(lms[8].y)

            # clean entry/exit with fresh baseline
            if pinched and not scrolling:
                scrolling = True
                baseline_y = iy
            elif (not pinched) and scrolling:
                scrolling = False
                baseline_y = None

            # ---- Rate-controlled scroll (~30Hz) ----
            if scrolling and baseline_y is not None and (now - last_tick)*1000 >= TICK_MS:
                dy = baseline_y - iy  # up = positive

                # deadzone + gentle clutch to fight drift
                v = vel_ema.update(dy)
                if abs(dy) < SCROLL_DEADZONE:
                    baseline_y = 0.98*baseline_y + 0.02*iy  # micro re-center when still
                    lines = 0
                else:
                    # nonlinear compression avoids bursty spikes
                    dy_nl = math.tanh(3.0 * dy)
                    lines = int(max(-SCROLL_CLAMP, min(SCROLL_CLAMP, SCROLL_GAIN * dy_nl)))

                    # clutch: if near zero "velocity", re-center slowly
                    if abs(v) < SCROLL_DEADZONE * 0.5:
                        baseline_y = 0.95*baseline_y + 0.05*iy

                smooth_scroll_lines(lines)
                last_tick = now

            # ---- HUD ----
            if pinched: hud.append(f"PINCH:{pinch_val:.2f}")
            if scrolling: hud.append("SCROLL")

        # HUD & preview
        cv2.putText(frame, " | ".join(hud) if hud else "READY", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if pinched else (0,180,255), 2)
        cv2.imshow("Pinch Scroll (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
