#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Embedding
embedding_layer = Embedding(1000, 64)


# In[2]:

#此处书本是from keras.layers import preprocessing，但是我下载的keras，preprocessing是在keras文件夹下的
from keras.datasets import imdb
from keras import preprocessing
#作为特征的单词个数
max_features = 10000
#在这么多单词后截断文本(这些单词都属于前 max_features 个最常见的单词)
maxlen = 20


# In[3]:


#将数据加载为整数列表
(x_train, y_train), (x_test, y_test) = imdb.load_data(
num_words=max_features)


# In[4]:


#将整数列表转换成形状为 (samples,maxlen) 的二维整数张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


# In[6]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
#指定 Embedding 层的最大输入长度,以便后面将嵌入输入展平。 Embedding 层激活的形状为 (samples, maxlen, 8)
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
#将三维的嵌入张量展平成形状为 (samples, maxlen * 8) 的二维张量
model.add(Flatten())
#在上面添加分类器
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train,
epochs=10,
batch_size=32,
validation_split=0.2)


# In[ ]:




