import os
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter


# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# 预处理
def preprocessing(document):
    back = []
    # 去特殊字符
    regex_punctuation = re.compile('[%s]' % re.escape(string.punctuation))
    # 生成波特词干算法对象
    # ps = PorterStemmer()
    # 加载Stemming类库
    lemmatizer = WordNetLemmatizer()
    # Stopword
    stop_words = set(stopwords.words("english"))
    # Tokenization
    tokens = word_tokenize(document)
    for token in tokens:
        # normalization
        token = regex_punctuation.sub("", token)
        token = token.lower()
        # Stemming
        # token = ps.stem(token)
        token = lemmatizer.lemmatize(token)
        if token and not token.isdigit() and token not in stop_words and wordnet.synsets(token):
            back.append(str(token))
    return back


# 输入数据集并预处理
def input_data(base_path='data/20news-18828', flush=False, out1='data/out/documents.csv', out2='data/out/labels.csv'):
    print('Inputting')
    documents = []
    labels = []
    # 若documents,labels已存在，只读取
    if not os.path.exists(out1) or flush:
        i = 0
        # j表示类型
        j = -1
        for folder in os.listdir(base_path):
            path = os.path.join(base_path, folder)
            j += 1
            for filename in os.listdir(path):
                labels.append(j)
                filepath = os.path.join(path, filename)
                with open(filepath, encoding='latin-1') as file:
                    document = file.read()
                    documents.append(preprocessing(document))
                i += 1
                print(i)
        docs = [str(doc) for doc in documents]
        pd.DataFrame(docs).to_csv(out1, sep=" ", header=None, index=None)
        pd.DataFrame(labels).to_csv(out2, sep=" ", header=None, index=None)
    else:
        labels = np.array(pd.read_csv(out2, sep=" ", header=None)).reshape(1, -1)[0]
        documents = np.array(pd.read_csv(out1, sep=" ", header=None))
        documents = [doc[0].replace(' ', '').replace('\'', '').replace('[', '').replace(']', '').split(',') for doc in
                     documents]
    return documents, labels


# 生成词典
def f_dictionary(documents, out='data/out/dictionary.csv'):
    dictionary = []
    print('Dictionary')
    # 若词典已存在，只读取
    if not os.path.exists(out):
        temp = []
        for document in documents:
            temp += document
        # 统计documents中出现的token频率
        temp = Counter(temp)
        for token in temp:
            # 过滤词频8以下token
            if temp[token] >= 4 and token not in dictionary:
                dictionary.append(str(token))
        # 保存词典
        pd.DataFrame(dictionary).to_csv(out, sep=" ", header=None, index=None)
    else:
        # 读取词典
        dictionary = np.array(pd.read_csv(out, sep=" ", header=None)).reshape(1, -1)[0]
    return dictionary


# 生成01型vector space
def vsm(documents, dictionary):
    print('vector space')
    vectors = []
    i = 0
    for document in documents:
        vector = []
        for item in dictionary:
            if item in document:
                vector.append(1)
            else:
                vector.append(0)
        vectors.append(vector)
        i += 1
        print(i)
    # 保存
    # print('save')
    # pd.DataFrame(vectors).to_csv('data/out/vsm-01.csv', sep=",", header=None, index=None)
    return vectors


if __name__ == '__main__':
    # 输入数据
    documents, labels = input_data()
    # 生成词典
    dictionary = f_dictionary(documents)
    # 生成vector space
    vsm(documents, dictionary)
