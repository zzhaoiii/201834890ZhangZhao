import vsm
import math
import pandas as pd
import nltk
import os
import numpy as np
import gc
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.metrics import accuracy_score
import shutil


# tf*idf 产生权重
def TF_IDF1(documents, dictionary):
    print('tf-idf')
    vectors = []
    i = 0
    for document in documents:
        vector = []
        for item in dictionary:
            tf = str(document).count(str(item))
            # Sub-linear TF scaling
            if tf > 0:
                tf = 1 + math.log(tf)
            else:
                tf = 0
            # 过滤掉不含item的document，再计算长度
            df = len(list(filter(lambda doc: item in doc, documents)))
            idf = math.log(len(documents) / df)
            weight = tf * idf
            print(weight)
            vector.append(weight)
        vectors.append(vector)
        print(i)
        i += 1
    return vectors


# 使用类库tf*idf 产生权重
def TF_IDF2(documents, dictionary):
    print('tf-idf')
    vectors = []
    i = 0
    # 重新构造文本集
    Texts = []
    for document in documents:
        Text = ''
        for token in document:
            if token in dictionary:
                Text += (' ' + token)
        Texts.append(Text)
        print(i)
        i += 1
    # 加载计算tf-idf类库
    tc = nltk.TextCollection(Texts)
    i = 0
    for document in Texts:
        vector = []
        for item in dictionary:
            # 计算tf-idf
            weight = tc.tf_idf(str(item), document)
            vector.append(weight)
        vectors.append(vector)
        print(i)
        i += 1
    # pd.DataFrame(vectors).to_csv(out, sep=",", header=None, index=None)

    return vectors


def knn(train_X, train_Y, test_X, test_Y):
    print('knn')
    # 计算余弦相似度
    similarity = cosine_similarity(test_X, train_X)
    del train_X, test_X
    gc.collect()
    for K in range(30, 50, 1):
        prediction = []
        for item in similarity:
            dic = dict(zip(item, train_Y))
            dic = sorted(dic.items(), key=lambda v: v[0], reverse=True)
            # 前K个临近的类别
            # 计数法：
            # classes = [str(i[1]) for i in dic[:K]]
            # prediction.append(int(Counter(classes).most_common(1)[0][0]))
            # 权重法：计算权重
            classes = np.zeros((1, len(train_Y)))[0]
            for i in dic[:K]:
                classes[i[1]] += (1 / (1 - i[0]) ** 2)
            dic = dict(zip(classes, range(len(train_Y))))
            dic = sorted(dic.items(), key=lambda v: v[0], reverse=True)
            prediction.append(dic[0][1])
        # 将预测结果输出
        # pd.DataFrame(prediction).to_csv('data/knn-out/prediction.csv', sep=" ", header=None, index=None)
        # 对结果评估
        print(K, "accuracy_score:\t", accuracy_score(test_Y, prediction))


# 将数据集分为train set,test set
def devide(base_path='data/20news-18828'):
    for folder in os.listdir(base_path):
        path = os.path.join(base_path, folder)
        os.makedirs('data/20news-train/' + folder)
        os.makedirs('data/20news-test/' + folder)
        i = 0
        for filename in os.listdir(path):
            if i < len(os.listdir(path)) * 0.8:
                shutil.copyfile(os.path.join(path, filename), os.path.join('data/20news-train/' + folder, filename))
            else:
                shutil.copyfile(os.path.join(path, filename), os.path.join('data/20news-test/' + folder, filename))
            i += 1


if __name__ == '__main__':
    # 将数据划分train set 与 test set
    # devide()
    # 输入train set
    train_X, train_Y = vsm.input_data(base_path='data/20news-train', out1='data/knn-out/train_X.csv',
                                      out2='data/knn-out/train_Y.csv')
    # 生成词典
    dictionary = vsm.f_dictionary(train_X, 'data/knn-out/dictionary.csv')
    # tf-idf 获取train_X 的 vector space
    vsm_train = vsm.vsm(train_X, dictionary)
    del train_X
    gc.collect()
    # 输入test set
    test_X, test_Y = vsm.input_data(base_path='data/20news-test', out1='data/knn-out/test_X.csv',
                                    out2='data/knn-out/test_Y.csv')
    # tf-idf 获取test_X 的 vector space
    vsm_test = vsm.vsm(test_X, dictionary)
    del test_X
    gc.collect()
    # knn分类
    knn(vsm_train, train_Y, vsm_test, test_Y)
