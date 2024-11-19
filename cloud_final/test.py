
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 加載測試數據
test_data = np.load('validation.npz')
X_test = test_data['data']
y_test = test_data['label'].astype('int').flatten()

# 加載訓練好的模型
model = tf.keras.models.load_model('my_model.h5')

# 模型推論
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 評估模型性能
accuracy = accuracy_score(y_test, y_pred_classes)
conf_matrix = confusion_matrix(y_test, y_pred_classes)
class_report = classification_report(y_test, y_pred_classes, target_names=['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER'])

# 打印結果
print("測試準確度: {:.2f}%".format(accuracy * 100))
print("\n混淆矩陣:")
print(conf_matrix)
print("\n分類報告:")
print(class_report)
