import vsm
import math
import pandas as pd
import nltk


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
        print(vector)
        print(i)
        i += 1
    return vectors


# 使用类库tf*idf 产生权重
def TF_IDF2(documents, dictionary):
    print('tf-idf')
    vectors = []
    i = 0
    tc = nltk.TextCollection([str(document) for document in documents])
    for document in documents:
        vector = []
        for item in dictionary:
            weight = tc.tf_idf(item, str(document))
            vector.append(weight)
        vectors.append(vector)
        print(i)
        i += 1
    return vectors


if __name__ == '__main__':
    documents = []
    labels = []
    dictionary = []
    # 输入数据
    vsm.input_data(documents, labels)
    # 预处理
    vsm.f_dictionary(documents, dictionary)
    # tf-idf 获取vector space
    vectors = TF_IDF2(documents, dictionary)
    # print(vectors)
    pd.DataFrame(vectors).to_csv('data/out/vsm-ft-idf.csv', sep=" ", header=None, index=None)
