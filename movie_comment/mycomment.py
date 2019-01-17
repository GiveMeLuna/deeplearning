#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
num_words=10000)


# In[2]:


word_index = imdb.get_word_index()
reverse_word_index = dict(
[(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
[reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# In[3]:


import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results
x_train=vectorize_sequences(train_data)     #将训练数据向量化
#比如x_train[0]中的内容是[0,1,1...,0,0]，含义是train_data[0]中某个单词的索引数字为i，x_train[0][i]便等于1
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[9]:

#尝试的评论（我这里是在豆瓣直接搜索了一个差评然后翻译得到。）
test_comment="5 points, the original script is really excellent, so that every modification and localization of the domestic version, let a person feel superfluous and destroy the rhythm. Especially a few key turns and breaking points, the play is not completely out, a bit of a waste of such a good set. The original version is actually a set of chamber thriller, and every change makes people jumpy. This time, the dramatic treatment and all kinds of self-righteous chicken soup and warmth have destroyed the original plot depth. And is it really the cell phone that is to blame? It's also strange. Finally, as an actress, mengyao xi has a long, long, long way to go."


# In[10]:

#将句子通过索引变成整数序列
testcomment=test_comment.split()
mycomment=[]
mycomment.append(1)
for i in testcomment:
    if word_index.get(i):
        if(word_index[i]+3<10000):
            mycomment.append(word_index[i]+3)
    else:
        mycomment.append(2)
print(mycomment)


# In[11]:

#将序列向量化
my_comment = vectorize_sequences([mycomment])


# In[7]:

#训练模型
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[12]:

#预测评论的正面和负面
print(model.predict(my_comment))


# In[ ]:




