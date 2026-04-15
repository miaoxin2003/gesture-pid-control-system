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
        # print(results.multi_hand_landmarks)

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
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    if id == 4:  # Thumb tip
                        cv2.circle(img, (cx, cy), 15, (0, 0, 139), cv2.FILLED) # Dark Blue
                    else:
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED) # Green

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    # Serial port initialization
    ser = None # Initialize ser to None
    try:
        ser = serial.Serial()
        ser.baudrate = 115200
        ser.port = 'COM8'
        ser.open()
        # Send initial center coordinate to keep pan-tilt stable
        initial_data = '#'+str('320')+'$'+str('240')+'\r\n'
        ser.write(initial_data.encode())
        #print("Serial port opened successfully and initial data sent.")
    except serial.SerialException as e:
        #print(f"Error opening serial port: {e}. Please check if the port is correct and not in use.")
        ser = None # Set ser to None if opening fails

    # 加载模型
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
            cv2.putText(img, f"Gesture: {prediction[0]}", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 保持原有的串口逻辑
            thumb_tip_x = lmList[4][1]
            thumb_tip_y = lmList[4][2]

            # ========== 自适应EMA滤波 + 二阶离群点剔除 ==========
            alpha = 0.1  # 默认alpha值
            # 初始化EMA滤波状态
            if detector.ema_x is None:
                detector.ema_x = thumb_tip_x
                detector.ema_y = thumb_tip_y
                smooth_x = thumb_tip_x
                smooth_y = thumb_tip_y
            else:
                # 计算当前帧速度（与上一帧的差值）
                velocity_x = thumb_tip_x - detector.ema_x
                velocity_y = thumb_tip_y - detector.ema_y

                # 保存速度历史
                detector.velocity_x_history.append(abs(velocity_x))
                detector.velocity_y_history.append(abs(velocity_y))
                if len(detector.velocity_x_history) > detector.max_velocity_history:
                    detector.velocity_x_history.pop(0)
                    detector.velocity_y_history.pop(0)

                # 计算平均速度
                avg_velocity_x = sum(detector.velocity_x_history) / len(detector.velocity_x_history)
                avg_velocity_y = sum(detector.velocity_y_history) / len(detector.velocity_y_history)
                avg_velocity = (avg_velocity_x + avg_velocity_y) / 2

                # 根据速度自适应调整alpha
                # 速度越大 -> alpha越大 -> 对原始值响应越快（减少延迟）
                # 速度越小 -> alpha越小 -> 滤波越平滑
                if avg_velocity > detector.velocity_threshold:
                    alpha = detector.alpha_fast
                else:
                    alpha = detector.alpha_slow

                # 一阶离群点检测：单帧坐标突变
                deviation_x = abs(thumb_tip_x - detector.ema_x)
                deviation_y = abs(thumb_tip_y - detector.ema_y)
                outlier_threshold = 50

                if deviation_x > outlier_threshold or deviation_y > outlier_threshold:
                    alpha = alpha * 0.3

                # 二阶离群点检测：速度突变（当前速度远超历史平均）
                # 如果当前速度是历史平均的3倍以上，可能是MediaPipe跳变
                current_velocity = abs(velocity_x) + abs(velocity_y)
                if len(detector.velocity_x_history) >= 3:
                    history_avg = sum(detector.velocity_x_history) + sum(detector.velocity_y_history)
                    history_avg = history_avg / (len(detector.velocity_x_history) + len(detector.velocity_y_history)) * 2
                    if history_avg > 5 and current_velocity > history_avg * 3:
                        alpha = alpha * 0.2  # 极度保守

                # EMA公式：filtered = alpha * raw + (1 - alpha) * previous_filtered
                detector.ema_x = alpha * thumb_tip_x + (1 - alpha) * detector.ema_x
                detector.ema_y = alpha * thumb_tip_y + (1 - alpha) * detector.ema_y

                smooth_x = int(detector.ema_x)
                smooth_y = int(detector.ema_y)
            # ================================================

            if ser and ser.is_open:
                data = '#'+str(smooth_x)+'$'+str(smooth_y)+'\r\n'
                print(f"Raw: ({thumb_tip_x}, {thumb_tip_y}) -> Smooth: ({smooth_x}, {smooth_y}) | alpha: {alpha:.3f}")
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
