import cv2
import joblib
import numpy as np
import os
from pathlib import Path
from HandTrackingModule import handDetector

# 1. 加载模型
model = joblib.load('D:/备份3/gesture_model.pkl')

# 2. 初始化
cap = cv2.VideoCapture(0)
detector = handDetector()

print("开始实时手势识别，按 'q' 退出")

while True:
    success, img = cap.read()
    if not success:
        break
        
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        # 提取特征
        features = []
        for lm in lmList:
            features.extend([lm[1], lm[2]])
        
        # 预测
        prediction = model.predict([features])
        gesture = prediction[0]
        
        # 显示结果
        cv2.putText(img, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
