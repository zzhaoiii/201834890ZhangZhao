import vsm
import math
import pandas as pd
import nltk
import os
import numpy as np
import gc


# tf*idf 产生权重
def TF_IDF1(documents, dictionary):
    print('tf-idf')
    vectors = []
    i = 0
    for document in documents:
        vector = []
        for item in dictionary:
            tf = str(document).count(item)
            # Sub-linear TF scaling
            if tf > 0:
                tf = 1 + math.log(tf)
            # 过滤掉不含item的document，再计算长度
            df = len(list(filter(lambda doc: item in doc, documents)))
            idf = math.log(len(documents) / df)
            weight = tf * idf
            vector.append(weight)
        vectors.append(vector)
        print(i)
        i += 1
    return vectors


# 使用类库tf*idf 产生权重
def TF_IDF2(documents, dictionary):
    print('tf-idf')
    vectors = []
    # 判断是否之前算过，若存在直接读取
    if not os.path.exists('data/out/vsm-ft-idf.csv'):
        i = 0
        # 加载计算tf-idf类库
        tc = nltk.TextCollection([str(doc) for doc in documents])
        for document in documents:
            vector = []
            for item in dictionary:
                # 计算tf-idf
                weight = tc.tf_idf(str(item), str(document))
                vector.append(weight)
            vectors.append(vector)
            print(i)
            i += 1
        pd.DataFrame(vectors).to_csv('data/out/vsm-ft-idf.csv', sep=",", header=None, index=None)
    else:
        vectors = np.array(pd.read_csv('data/out/vsm-ft-idf.csv', sep="\n", header=None))
        vectors = [vector[0].split(',') for vector in vectors]
        gc.collect()
        print(vectors)
    return vectors


if __name__ == '__main__':
    # 输入数据
    documents, labels = vsm.input_data()
    # 生成词典
    dictionary = vsm.f_dictionary(documents)
    # tf-idf 获取vector space
    vectors = TF_IDF2(documents, dictionary)
