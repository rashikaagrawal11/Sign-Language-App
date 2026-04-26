import cv2
import mediapipe as mp
import os

# Create folder to save images
save_path = "captured"
os.makedirs(save_path, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

count = 0

print("Press C to capture | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_img = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            x_list = []
            y_list = []

            for lm in handLms.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            # padding
            pad = 20
            xmin = max(0, xmin - pad)
            ymin = max(0, ymin - pad)
            xmax = min(w, xmax + pad)
            ymax = min(h, ymax + pad)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

            hand_img = frame[ymin:ymax, xmin:xmax]

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)

    # Press C to save
    if key == ord('c') and hand_img is not None:
        img = cv2.resize(hand_img, (64, 64))
        file_name = os.path.join(save_path, f"img_{count}.jpg")
        cv2.imwrite(file_name, img)
        print(f"Saved: {file_name}")
        count += 1

    # ESC to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()