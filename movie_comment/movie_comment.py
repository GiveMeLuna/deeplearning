#!/usr/bin/env python
# coding: utf-8

# In[1]:

#这行在用Jupyter运行的时候才加上
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# word_index是一个将单词映射为整数索引的字典
word_index = imdb.get_word_index()
#键值颠倒，将整数索引映射为单词
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#将评论阶码，减去了3是因为0、1、2是保留的索引（0：padding填充；1：start of sequence序列开始；2：unknown未知词）。
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# In[3]:


print(decoded_review)


# In[4]:


#将整数序列编码为二进制矩阵
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))     #创建一个零矩阵
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.       #将result[i]的制定索引设置为1
    return results
x_train=vectorize_sequences(train_data)     #将训练数据向量化
#比如x_train[0]中的内容是[0,1,1...,0,0]，含义是train_data[0]中某个单词的索引数字为i，x_train[0][i]便等于1
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[ ]:


print(x_train[0])


# In[ ]:


from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])


# In[ ]:


x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[ ]:


model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])
history = model.fit(partial_x_train,
partial_y_train,
epochs=20,
batch_size=512,
validation_data=(x_val, y_val))


# In[ ]:


import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


#绘制训练精度和验证精度
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[ ]:


print(results)


# In[ ]:


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[ ]:


print(results)


# In[ ]:



