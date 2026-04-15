import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import cv2
import mediapipe as mp
import time
import serial
import joblib
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.x_history = []
        self.y_history = []
        self.window_size = 5

        # ========== 自适应EMA滤波参数 ==========
        # EMA基础系数：运动慢时增大（增强平滑），运动快时减小（减少延迟）
        self.alpha_fast = 0.15   # 快速运动时的alpha
        self.alpha_slow = 0.05    # 慢速运动时的alpha
        self.velocity_threshold = 15  # 速度阈值，像素/帧

        # EMA滤波状态
        self.ema_x = None
        self.ema_y = None

        # 历史速度用于计算加速度
        self.velocity_x_history = []
        self.velocity_y_history = []
        self.max_velocity_history = 5

        # ========== 预测控制参数 ==========
        self.pred_x = None        # 预测位置
        self.pred_y = None
        self.last_ema_x = None     # 上一帧EMA值（用于计算速度）
        self.last_ema_y = None
        self.frame_count = 0       # 帧计数器
        self.use_prediction = True  # 是否使用预测
        # 预测速度阈值：速度低于此值时不预测（防止静止时噪声）
        self.min_velocity_for_prediction = 2
        # 预测误差容忍度：预测值与平滑值的偏差超过此值时不信任预测
        self.prediction_tolerance = 80
        # ======================================

        # ========== 手势状态机参数 ==========
        # 状态：WAITING(等待) -> CONFIRMING(确认中) -> TRIGGERED(已触发) -> WAITING
        self.gesture_state = "WAITING"
        self.last_gesture = None      # 上一次检测到的手势
        self.confirmed_gesture = None # 已确认的手势（触发后）
        self.hold_start_time = None   # 开始保持的时间戳
        self.hold_duration = 0.5      # 触发需要的保持时间（秒）
        self.cooldown_start = None    # 冷却开始时间
        self.cooldown_duration = 1.0 # 冷却时间（秒），防止重复触发
        # ======================================

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    if id == 4:
                        cv2.circle(img, (cx, cy), 15, (0, 0, 139), cv2.FILLED)
                    else:
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    ser = None
    try:
        ser = serial.Serial()
        ser.baudrate = 115200
        ser.port = 'COM8'
        ser.open()
        initial_data = '#'+str('320')+'$'+str('240')+'\r\n'
        ser.write(initial_data.encode())
    except serial.SerialException as e:
        ser = None

    try:
        model = joblib.load('D:/备份3/gesture_model.pkl')
    except:
        model = None

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from camera.")
            break
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=True)
        
        if len(lmList) != 0 and model:
            features = []
            for lm in lmList:
                features.extend([lm[1], lm[2]])
            prediction = model.predict([features])
            current_gesture = prediction[0]

            # ========== 手势状态机 ==========
            cTime = time.time()

            if detector.gesture_state == "WAITING":
                if current_gesture is not None:
                    detector.last_gesture = current_gesture
                    detector.hold_start_time = cTime
                    detector.gesture_state = "CONFIRMING"

            elif detector.gesture_state == "CONFIRMING":
                if current_gesture == detector.last_gesture:
                    if cTime - detector.hold_start_time >= detector.hold_duration:
                        detector.confirmed_gesture = current_gesture
                        detector.gesture_state = "TRIGGERED"
                        detector.cooldown_start = cTime
                        print(f"[STATE MACHINE] Gesture CONFIRMED and TRIGGERED: {current_gesture}")
                else:
                    detector.gesture_state = "WAITING"
                    detector.last_gesture = None
                    detector.hold_start_time = None

            elif detector.gesture_state == "TRIGGERED":
                if cTime - detector.cooldown_start >= detector.cooldown_duration:
                    detector.gesture_state = "WAITING"
                    detector.confirmed_gesture = None

            state_display = f"State: {detector.gesture_state}"
            if detector.gesture_state == "CONFIRMING":
                hold_time = cTime - detector.hold_start_time if detector.hold_start_time else 0
                state_display += f" ({hold_time:.2f}/{detector.hold_duration}s)"
            cv2.putText(img, state_display, (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
            cv2.putText(img, f"Gesture: {current_gesture}", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # ======================================
            
            thumb_tip_x = lmList[4][1]
            thumb_tip_y = lmList[4][2]

            # ========== 自适应EMA滤波 + 二阶离群点剔除 ==========
            alpha = 0.1
            if detector.ema_x is None:
                detector.ema_x = thumb_tip_x
                detector.ema_y = thumb_tip_y
                detector.last_ema_x = thumb_tip_x
                detector.last_ema_y = thumb_tip_y
                smooth_x = thumb_tip_x
                smooth_y = thumb_tip_y
                pred_x = thumb_tip_x
                pred_y = thumb_tip_y
            else:
                velocity_x = thumb_tip_x - detector.ema_x
                velocity_y = thumb_tip_y - detector.ema_y

                detector.velocity_x_history.append(abs(velocity_x))
                detector.velocity_y_history.append(abs(velocity_y))
                if len(detector.velocity_x_history) > detector.max_velocity_history:
                    detector.velocity_x_history.pop(0)
                    detector.velocity_y_history.pop(0)

                avg_velocity_x = sum(detector.velocity_x_history) / len(detector.velocity_x_history)
                avg_velocity_y = sum(detector.velocity_y_history) / len(detector.velocity_y_history)
                avg_velocity = (avg_velocity_x + avg_velocity_y) / 2

                if avg_velocity > detector.velocity_threshold:
                    alpha = detector.alpha_fast
                else:
                    alpha = detector.alpha_slow

                deviation_x = abs(thumb_tip_x - detector.ema_x)
                deviation_y = abs(thumb_tip_y - detector.ema_y)
                outlier_threshold = 50

                if deviation_x > outlier_threshold or deviation_y > outlier_threshold:
                    alpha = alpha * 0.3

                current_velocity = abs(velocity_x) + abs(velocity_y)
                if len(detector.velocity_x_history) >= 3:
                    history_avg = sum(detector.velocity_x_history) + sum(detector.velocity_y_history)
                    history_avg = history_avg / (len(detector.velocity_x_history) + len(detector.velocity_y_history)) * 2
                    if history_avg > 5 and current_velocity > history_avg * 3:
                        alpha = alpha * 0.2

                detector.last_ema_x = detector.ema_x
                detector.last_ema_y = detector.ema_y

                detector.ema_x = alpha * thumb_tip_x + (1 - alpha) * detector.ema_x
                detector.ema_y = alpha * thumb_tip_y + (1 - alpha) * detector.ema_y

                smooth_x = int(detector.ema_x)
                smooth_y = int(detector.ema_y)

                # ========== 预测控制 ==========
                pred_velocity_x = detector.ema_x - detector.last_ema_x
                pred_velocity_y = detector.ema_y - detector.last_ema_y

                pred_x = detector.ema_x + pred_velocity_x
                pred_y = detector.ema_y + pred_velocity_y

                pred_deviation_x = abs(pred_x - detector.ema_x)
                pred_deviation_y = abs(pred_y - detector.ema_y)
                pred_deviation = pred_deviation_x + pred_deviation_y

                if pred_deviation > detector.prediction_tolerance:
                    pred_x = detector.ema_x
                    pred_y = detector.ema_y
                    detector.use_prediction = False
                else:
                    detector.use_prediction = True

                if abs(pred_velocity_x) < detector.min_velocity_for_prediction and \
                   abs(pred_velocity_y) < detector.min_velocity_for_prediction:
                    pred_x = detector.ema_x
                    pred_y = detector.ema_y

                detector.frame_count += 1

            if ser and ser.is_open:
                send_x = int(pred_x)
                send_y = int(pred_y)
                data = '#'+str(send_x)+'$'+str(send_y)+'\r\n'
                pred_flag = "[PRED]" if detector.use_prediction else "[SMOOTH]"
                print(f"Raw: ({thumb_tip_x}, {thumb_tip_y}) -> Smooth: ({smooth_x}, {smooth_y}) -> {pred_flag} ({send_x}, {send_y})")
                try:
                    ser.write(data.encode())
                except serial.SerialException as e:
                    print(f"Error writing to serial port: {e}")
                    ser.close()
                    ser = None

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('s'):
            break

    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
