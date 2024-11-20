import numpy as np
import tensorflow as tf

# 載入驗證資料
validation_data = np.load('validation.npz')
X_val = validation_data['data']
y_val = validation_data['label']

# 載入保存的模型
model = tf.keras.models.load_model('new_model.h5')

# 計算模型在驗證集上的準確率
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}")
