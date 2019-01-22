#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 500
batch_size = 32
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(
num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


# In[4]:


#用 Embedding 层和 SimpleRNN 层来训练模型
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
epochs=10,
batch_size=128,
validation_split=0.2)


# In[7]:


# #使用 Keras 中的 LSTM 层http://localhost:8888/notebooks/%E6%96%87%E6%A1%A3/deeplearn/%E7%AC%AC%E5%85%AD%E7%AB%A0/%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/%E6%9C%AA%E5%91%BD%E5%90%8D.ipynb?kernel_name=python3#
# from keras.layers import LSTM
# model = Sequential()
# model.add(Embedding(max_features, 32))
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
# loss='binary_crossentropy',
# metrics=['acc'])
# history = model.fit(input_train, y_train,
# epochs=10,
# batch_size=128,
# validation_split=0.2)


# In[8]:


#绘制结果
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()


# In[9]:


plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:




