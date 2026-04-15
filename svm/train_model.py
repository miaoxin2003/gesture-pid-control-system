import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# 1. 加载数据
data = pd.read_csv('dataset.csv', header=None)
X = data.iloc[:, 1:]  # 坐标数据
y = data.iloc[:, 0]   # 标签

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建 SVM 模型 (使用 Pipeline 进行标准化)
model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 保存模型
joblib.dump(model, 'gesture_model.pkl')
print("模型训练完成并已保存为 gesture_model.pkl")
