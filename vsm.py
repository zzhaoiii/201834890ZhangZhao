import os
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import string
import gc

base_path = 'data/20news-18828'


# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

# 输入数据集
def input_data(documents, labels):
    print('Inputting')
    i = 0
    for folder in os.listdir(base_path):
        path = os.path.join(base_path, folder)
        for filename in os.listdir(path):
            labels.append(filename)
            filepath = os.path.join(path, filename)
            with open(filepath, encoding='latin-1') as file:
                document = file.read()
                documents.append(document)
            #     break
            # break
            i += 1
        if i > 2:
            break


# 预处理
def preprocessing(document):
    back = []
    # 去特殊字符
    regex_punctuation = re.compile('[%s]' % re.escape(string.punctuation))
    # 生成波特词干算法对象
    ps = PorterStemmer()
    # Stopword
    stop_words = set(stopwords.words("english"))
    # Tokenization
    tokens = word_tokenize(document)
    for token in tokens:
        # normalization
        token = regex_punctuation.sub("", token)
        token = token.lower()
        # Stemming
        token = ps.stem(token)
        if token and token not in stop_words:
            back.append(token)
    # 过滤词频小于2的token
    back = list(filter(lambda token: str(back).count(token) >= 4, back))
    # print(len(back))
    return back


# 生成词典
def f_dictionary(documents, dictionary):
    print('Dictionary')
    for i in range(len(documents)):
        document = documents[i]
        document = preprocessing(document)
        documents[i] = document
        for token in document:
            if token not in dictionary:
                dictionary.append(token)
        print(i)
    print(len(dictionary))


# 生成vector space
def vsm(documents, labels, dictionary):
    print('vector space')
    vectors = []
    for document in documents:
        vector = []
        for item in dictionary:
            if item in document:
                vector.append('1')
            else:
                vector.append('0')
        vectors.append(vector)
    vectors = np.array(vectors)
    # 将字典加在第一行
    vector_space = np.vstack((dictionary, vectors))
    del dictionary, vectors
    gc.collect()
    # 将lable加在第一列
    labels.insert(0, "Terms")
    vector_space = np.hstack((np.array(labels).reshape(-1, 1), vector_space))
    del labels
    gc.collect()

    # 保存
    print('save')
    pd.DataFrame(vector_space).to_csv('data/out/vsm-0.csv', sep=" ", header=None, index=None)


if __name__ == '__main__':
    documents = []
    labels = []
    dictionary = []
    # 输入数据
    input_data(documents, labels)
    # 预处理
    f_dictionary(documents, dictionary)
    # 生成vector space
    vsm(documents, labels, dictionary)
