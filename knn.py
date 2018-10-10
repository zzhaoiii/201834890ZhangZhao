import vsm
import math


# tf*idf 产生权重
def TF_IDF(documents, dictionary, tt):
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


if __name__ == '__main__':
    documents = []
    lables = []
    dictionary = []
    # 输入数据
    vsm.input_data(documents, lables)
    tt = documents[:]
    # 预处理
    vsm.preprocessing(documents, dictionary)
    # tf-idf 获取vector space
    vectors = TF_IDF(documents, dictionary, tt)
    print(vectors)
