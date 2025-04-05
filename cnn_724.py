import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.activations import elu
from tensorflow.keras.layers import Conv2D, LeakyReLU  # 导入LeakyReLU模块
from sklearn import svm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tf.keras.backend.clear_session()
#%%
# 定義文件路徑
npz_file = "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_data_7x24/processed_data_7x24_4704_processed_no.npz"

# 加載數據集
data = np.load(npz_file)

# 提取訓練數據和標籤
train_images = data['images']
train_labels = data['labels']

# 將數據轉換為 4 維格式
input_data = train_images.reshape(train_images.shape[0], 7, 24, 1)

# 劃分訓練集和驗證集
train_ratio = 0.5
X_train, X_val, y_train, y_val = train_test_split(input_data, train_labels, test_size=1-train_ratio, random_state=42)
#%%
# 建立 CNN 模型
model = Sequential()
model.add(Conv2D(8, (2, 3), padding='valid', activation=lambda x: elu(x, alpha=1.0), input_shape=(7, 24, 1), name='C1'))
model.add(Conv2D(16, (3, 3), padding='valid', activation=lambda x: elu(x, alpha=1.0), input_shape=(6, 22, 8), name='C2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='P1'))
model.add(Dropout(0.5, name='DR1'))
model.add(Flatten(name='F1'))
model.add(Dense(units=32, activation='relu', name='D1'))
model.add(Dense(units=10, activation='softmax', name='D2'))
print(model.summary())
#%%
# 編譯模型
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=5)
#%%
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel('train')
    plt.xlabel('epoch')
    # 設置圖例在左上角 
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
#%%
show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history,'loss','val_loss')
#%%
#%%
'''
# 提取訓練集和驗證集的特徵向量
train_features = model.predict(X_train)
val_features = model.predict(X_val)

# 不使用 one-hot 編碼的標籤轉換為整數形式
y_train_int = y_train
y_val_int = y_val

# 創建並訓練SVM模型
svm_model = svm.SVC()

# 直接使用fit方法進行訓練
svm_model.fit(train_features, y_train_int)

# 在驗證集上進行預測
val_predictions = svm_model.predict(val_features)

# 計算準確率
val_accuracy = np.mean(val_predictions == y_val_int)
print("Validation Accuracy:", val_accuracy)
'''