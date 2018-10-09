import os
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import string

base_path = 'data/20news-18828'

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

'''输入数据集'''
print('Inputting')
documents = []
lable = []
for folder in os.listdir(base_path):
    path = os.path.join(base_path, folder)
    for filename in os.listdir(path):
        lable.append(filename)
        filepath = os.path.join(path, filename)
        with open(filepath, encoding='latin-1') as file:
            document = file.read()
            documents.append(document)
# print(documents)

''''预处理数据集'''
print('Pre')
dictionary = []
regex_punctuation = re.compile('[%s]' % re.escape(string.punctuation))
# 生成波特词干算法实例
ps = PorterStemmer()
# Stopword
stop_words = set(stopwords.words("english"))
for i in range(len(documents)):
    document = documents[i]
    # Tokenization
    tokens = word_tokenize(document)
    # 去掉特殊字符
    tokens = filter(lambda token: token != "", [regex_punctuation.sub("", token) for token in tokens])
    # 将相应文本转换成tokens
    documents[i] = []
    for token in tokens:
        # normalization
        token = token.lower()
        # Stemming
        token = ps.stem(token)
        documents[i].append(token)
        if token not in stop_words and token not in dictionary:
            dictionary.append(token)
    print(i)
    print(dictionary)
# print(dictionary)

'''生成vector space'''
print('vector space')
vectors = []
for i in range(len(documents)):
    vector = []
    for j in range(len(dictionary)):
        if dictionary[j] in documents[i]:
            vector.append('1')
        else:
            vector.append('0')
    vectors.append(vector)
vectors = np.array(vectors)
# 将字典加在第一行
vector_space = np.vstack((dictionary, vectors))
# 将lable加在第一列
lable.insert(0, "Terms")
vector_space = np.hstack((np.array(lable).reshape(-1, 1), vector_space))

'''保存'''
print('save')
pd.DataFrame(vector_space).to_csv('data/out/vsm.csv', sep=" ", header=None, index=None)
