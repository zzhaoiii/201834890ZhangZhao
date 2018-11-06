import vsm
from sklearn.metrics import accuracy_score
from collections import Counter
import math


def count(documents, labels, dictionary):
    print('preprocessing')
    # 按类记录token
    C_feature = []
    # 所有token
    C_all = []
    # 每类数量
    C_kind = []
    for i in range(len(documents)):
        # 只保留词典中出现的token
        document = list(filter(lambda token: token in dictionary, documents[i]))
        C_all += document
        if labels[i] < len(C_feature):
            C_feature[int(labels[i])] += document
            C_kind[int(labels[i])] += 1
        else:
            C_feature.append(document)
            C_kind.append(1)
        print(i)
    return C_feature, C_kind, C_all


def NBC(test_X, test_Y, C_feature, C_kind, C_all, dictionary):
    print('NBC')
    counter0 = []
    counter1 = Counter(C_all)
    for kind in range(20):
        counter0.append(Counter(C_feature[kind]))
    prediction = []
    for i in range(len(test_X)):
        pre = []
        for kind in range(20):
            # P('具有某特征'|'属于某类')
            Pik = 0
            # P('具有某特征')
            Pi = 0
            features = counter0[kind]
            for token in test_X[i]:
                if token in dictionary:
                    # P('发票'|'S')
                    P0 = math.log((features[token] + 1) / (len(C_feature[kind]) + len(dictionary)))
                    Pik += P0
                    # P('发票')
                    P0 = math.log((counter1[token] + 1) / (len(C_all[kind]) + len(dictionary)))
                    Pi += P0
            # P('属于某类')
            Pk = math.log(C_kind[kind] / 18828.0)
            # P('属于某类'|'具有某特征')
            Pki = Pik + Pk - Pi
            pre.append([kind, Pki])
        pre = sorted(pre, key=lambda item: -item[1])
        print(i, pre[0][0])
        prediction.append(pre[0][0])
    # 输出准确率
    print(" accuracy_score:\t", accuracy_score(test_Y, prediction))


if __name__ == '__main__':
    # 输入train set
    train_X, train_Y = vsm.input_data(base_path='data/20news-train', out1='data/knn-out/train_X.csv',
                                      out2='data/knn-out/train_Y.csv')
    # 生成词典
    print(8,4000)
    dictionary = vsm.f_dictionary(train_X, 'data/knn-out/dictionary.csv', 8)
    # 对train set预处理
    C_feature, C_kind, C_all = count(train_X, train_Y, dictionary)
    # 输入test set
    test_X, test_Y = vsm.input_data(base_path='data/20news-test', out1='data/knn-out/test_X.csv',
                                    out2='data/knn-out/test_Y.csv')
    # NBC分类
    NBC(test_X, test_Y, C_feature, C_kind, C_all, dictionary)
