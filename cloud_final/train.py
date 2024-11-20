import numpy as np
import tensorflow as tf

# 載入訓練和驗證資料
train_data = np.load('train.npz')
validation_data = np.load('validation.npz')

X_train = train_data['data']
y_train = train_data['label']
X_val = validation_data['data']
y_val = validation_data['label']

# 使用 tf.distribute.MirroredStrategy 來實現多 GPU 支援
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# 設置模型和訓練過程
with strategy.scope():
    # 增加正則化和Dropout，並使用Batch Normalization
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(25,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 編譯模型並調整學習率
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# 評估模型
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}")

# 保存模型為 .h5 格式
model.save('my_model.h5')

print("模型已保存為 my_model.h5")
