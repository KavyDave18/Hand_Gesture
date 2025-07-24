import cv2
import mediapipe as mp
import numpy as np

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Shape config
filled_shapes = {
    'circle':  {'pos': [100, 100], 'radius': 30, 'color': (0, 0, 255)},
    'square':  {'pos': [100, 200], 'size': 60, 'color': (0, 255, 0)},
    'triangle': {'pos': [100, 300], 'size': 60, 'color': (255, 0, 0)},
    'star':    {'pos': [100, 400], 'size': 60, 'color': (255, 255, 0)},
}

empty_shapes = {
    'circle':  {'pos': [500, 100]},
    'square':  {'pos': [500, 200]},
    'triangle': {'pos': [500, 300]},
    'star':    {'pos': [500, 400]},
}

grabbed_shape = None
offset = [0, 0]
matched_shapes = []

def is_near(p1, p2, threshold=40):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < threshold

# Triangle drawing helper
def draw_triangle(img, center, size, color, filled=True):
    pts = np.array([
        [center[0], center[1] - size//2],
        [center[0] - size//2, center[1] + size//2],
        [center[0] + size//2, center[1] + size//2]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    if filled:
        cv2.fillPoly(img, [pts], color)
    else:
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)

# Star drawing helper (simple 4-point star)
def draw_star(img, center, size, color, filled=True):
    pts = np.array([
        [center[0], center[1] - size//2],
        [center[0] - size//4, center[1]],
        [center[0], center[1] + size//2],
        [center[0] + size//4, center[1]]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    if filled:
        cv2.fillPoly(img, [pts], color)
    else:
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)

# Start video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    index_finger = None
    thumb_tip = None
    pinch = False

    # Hand detection
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            index_finger = [int(lm[8].x * w), int(lm[8].y * h)]
            thumb_tip = [int(lm[4].x * w), int(lm[4].y * h)]
            if is_near(index_finger, thumb_tip, 30):
                pinch = True

    # Logic for grabbing and dragging
    if index_finger:
        ix, iy = index_finger

        if pinch:
            if grabbed_shape is None:
                for name, data in filled_shapes.items():
                    if name in matched_shapes:
                        continue
                    pos = data['pos']
                    if name == 'circle' and is_near(index_finger, pos, data['radius']):
                        grabbed_shape = name
                        offset = [pos[0] - ix, pos[1] - iy]
                        break
                    elif name == 'square':
                        s = data['size']//2
                        if pos[0]-s < ix < pos[0]+s and pos[1]-s < iy < pos[1]+s:
                            grabbed_shape = name
                            offset = [pos[0] - ix, pos[1] - iy]
                            break
                    elif name == 'triangle' or name == 'star':
                        if is_near(index_finger, pos, 40):
                            grabbed_shape = name
                            offset = [pos[0] - ix, pos[1] - iy]
                            break
            else:
                filled_shapes[grabbed_shape]['pos'][0] = ix + offset[0]
                filled_shapes[grabbed_shape]['pos'][1] = iy + offset[1]
        else:
            if grabbed_shape:
                # Check if close to empty shape
                pos_now = filled_shapes[grabbed_shape]['pos']
                empty_pos = empty_shapes[grabbed_shape]['pos']
                if is_near(pos_now, empty_pos, 40):
                    filled_shapes[grabbed_shape]['pos'] = empty_pos.copy()
                    matched_shapes.append(grabbed_shape)
            grabbed_shape = None

    # Draw empty shapes (outlines)
    for name, data in empty_shapes.items():
        pos = data['pos']
        color = (255, 255, 255)
        if name == 'circle':
            cv2.circle(frame, tuple(pos), 30, color, 3)
        elif name == 'square':
            s = 30
            cv2.rectangle(frame, (pos[0]-s, pos[1]-s), (pos[0]+s, pos[1]+s), color, 3)
        elif name == 'triangle':
            draw_triangle(frame, pos, 60, color, filled=False)
        elif name == 'star':
            draw_star(frame, pos, 60, color, filled=False)

    # Draw filled shapes
    for name, data in filled_shapes.items():
        pos = data['pos']
        color = data['color']
        if name == 'circle':
            cv2.circle(frame, tuple(pos), data['radius'], color, -1)
        elif name == 'square':
            s = data['size']//2
            cv2.rectangle(frame, (pos[0]-s, pos[1]-s), (pos[0]+s, pos[1]+s), color, -1)
        elif name == 'triangle':
            draw_triangle(frame, pos, data['size'], color, filled=True)
        elif name == 'star':
            draw_star(frame, pos, data['size'], color, filled=True)

    # Draw index dot
    if index_finger:
        cv2.circle(frame, tuple(index_finger), 8, (255, 255, 255), -1)
        if pinch:
            cv2.putText(frame, "Pinching", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Score
    cv2.putText(frame, f"Matched: {len(matched_shapes)}/4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if len(matched_shapes) == 4:
        cv2.putText(frame, "🎉 You matched all shapes!", (120, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Shape Matching Game", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
