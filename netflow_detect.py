#notice: this code is original developed on google colaboratory
#it might not work well in the local machine

#colaboratory linK: https://colab.research.google.com/drive/1CzqhTYHV5vx5jCPxKDcORDO3_hosuKJN?usp=sharing

#mount the google drive for the dataset
from google.colab import drive  #Open files from Google Drive
drive.mount('/gdrive')
%cd /gdrive

#input dataset
import pandas as pd
df = pd.read_csv('/gdrive/MyDrive/CS7265/nf_wStatus300000.csv')

#import library
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

#show relationship heat map
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#data splitting
train, val = train_test_split(df, test_size=0.3, shuffle=True)
train, test = train_test_split(train, test_size=0.1, shuffle=True)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

#read data file function, match with the label(status)
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('status')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#features encoding
feature_columns = []
for header in ['TimeSeconds', 'dateTimeStr', 'ipLayerProtocol', 'firstSeenSrcPort', 'firstSeenDestPort', 'moreFragments', 'contFragments', 'durationSeconds', 'firstSeenSrcPayloadBytes', 'firstSeenDestPayloadBytes', 'firstSeenSrcTotalBytes', 'firstSeenDestTotalBytes', 'firstSeenSrcPacketCount', 'firstSeenDestPacketCount', 'recordForceOut']:
  feature_columns.append(feature_column.numeric_column(header))

ipLayerProtocolCode = feature_column.categorical_column_with_vocabulary_list('ipLayerProtocolCode',df.ipLayerProtocolCode.unique(), dtype=tf.string)
ipLayerProtocolCode_embedding = feature_column.embedding_column(ipLayerProtocolCode, dimension=2)
feature_columns.append(ipLayerProtocolCode_embedding)

firstSeenSrcIp = feature_column.categorical_column_with_vocabulary_list('firstSeenSrcIp',df.firstSeenSrcIp.unique(), dtype=tf.string)
firstSeenSrcIp_embedding = feature_column.embedding_column(firstSeenSrcIp, dimension=10)
feature_columns.append(firstSeenSrcIp_embedding)

firstSeenDestIp = feature_column.categorical_column_with_vocabulary_list('firstSeenDestIp',df.firstSeenDestIp.unique(), dtype=tf.string)
firstSeenDestIp_embedding = feature_column.embedding_column(firstSeenDestIp, dimension=10)
feature_columns.append(firstSeenDestIp_embedding)

#assign encoded feature column as layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

#model without regulation
from keras.layers import *
from keras.models import Sequential

#for regulation uncommented following line
#from keras.regularizers import L2
#feature_layer,
  #layers.Dense(64, kernel_regularizer=L2(0.001), activation='relu'),
  #layers.Dense(32, kernel_regularizer=L2(0.001), activation='relu'),
  #layers.Dropout(0.5),
  #layers.Dense(16, kernel_regularizer=L2(0.001), activation='relu'),
  #layers.Dense(5,activation='softmax')


model=Sequential()
model.add(feature_layer)
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=20, verbose=1)
print(model.summary())

# Predicting the Test set results
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
