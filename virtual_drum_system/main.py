# ================================================================
# ADVANCED VIRTUAL DRUM KIT – CORRECTED DRUM ORDER VERSION
# ================================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import mediapipe as mp
import pygame
import time
import math

# -------------------- Audio --------------------
pygame.mixer.init(buffer=512)

SOUNDS = {
    "CRASH": pygame.mixer.Sound("drum_sounds/crash.wav"),
    "CYMBAL": pygame.mixer.Sound("drum_sounds/cymbal.wav"),
    "SNARE": pygame.mixer.Sound("drum_sounds/snare.wav"),
    "TOM": pygame.mixer.Sound("drum_sounds/tom.wav"),
    "KICK": pygame.mixer.Sound("drum_sounds/kick.wav"),
}

# -------------------- MediaPipe --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -------------------- Camera --------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("Virtual Drum Kit", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Virtual Drum Kit",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# -------------------- DRUM ZONES (YOUR EXACT ORDER) --------------------
DRUMS = {
    # TOP ROW
    "CRASH":  (120, 140, 340, 300),
    "CYMBAL": (480, 140, 700, 300),
    "SNARE":  (840, 140, 1060, 300),

    # BOTTOM ROW
    "TOM":    (360, 420, 580, 620),
    "KICK":   (720, 420, 1000, 680),
}

COLORS = {
    "CRASH": (255, 140, 0),
    "CYMBAL": (255, 255, 0),
    "SNARE": (255, 0, 0),
    "TOM": (180, 0, 255),
    "KICK": (0, 255, 0),
}

# -------------------- Hit Control --------------------
COOLDOWN = 0.18
last_hit = {d: 0 for d in DRUMS}
impact_time = {d: 0 for d in DRUMS}
prev_tip = {}

# -------------------- Utils --------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def play_drum(drum, velocity):
    now = time.time()
    if now - last_hit[drum] > COOLDOWN:
        volume = min(max(velocity / 40, 0.25), 1.0)
        SOUNDS[drum].set_volume(volume)
        SOUNDS[drum].play()
        last_hit[drum] = now
        impact_time[drum] = now

# ==================== MAIN LOOP ====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # -------------------- Draw Drums --------------------
    for drum, (x1, y1, x2, y2) in DRUMS.items():
        shrink = 6 if time.time() - impact_time[drum] < 0.08 else 0

        cv2.rectangle(
            frame,
            (x1 + shrink, y1 + shrink),
            (x2 - shrink, y2 - shrink),
            COLORS[drum],
            4
        )

        cv2.putText(
            frame,
            drum,
            (x1 + 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            COLORS[drum],
            2
        )

    # -------------------- Hand Tracking --------------------
    if result.multi_hand_landmarks:
        for i, hand in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            wrist = hand.landmark[0]
            tip = hand.landmark[8]

            wx, wy = int(wrist.x * w), int(wrist.y * h)
            tx, ty = int(tip.x * w), int(tip.y * h)

            # Drum stick
            cv2.line(frame, (wx, wy), (tx, ty), (220, 220, 220), 6)
            cv2.circle(frame, (tx, ty), 10, (0, 255, 0), -1)

            prev = prev_tip.get(i)
            velocity = 0
            downward = False

            if prev:
                velocity = dist((tx, ty), prev)
                downward = ty > prev[1] + 4

            prev_tip[i] = (tx, ty)

            # -------------------- Collision --------------------
            if velocity > 12 and downward:
                for drum, (x1, y1, x2, y2) in DRUMS.items():
                    if x1 < tx < x2 and y1 < ty < y2:
                        play_drum(drum, velocity)

                        # Impact animation
                        cv2.circle(frame, (tx, ty), 26, (255, 255, 255), 2)
                        cv2.line(frame, (tx - 14, ty), (tx + 14, ty), (255, 255, 255), 2)
                        cv2.line(frame, (tx, ty - 14), (tx, ty + 14), (255, 255, 255), 2)

    cv2.imshow("Virtual Drum Kit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------- Cleanup --------------------
cap.release()
cv2.destroyAllWindows()
pygame.quit()
