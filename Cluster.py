from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.mixture import GaussianMixture
import json
import vsm
from sklearn import metrics
import knn


def read_data(path='data/cluster/Tweets.txt'):
    documents = []
    labels = []
    with open(path) as file:
        lines = file.read().split('\n')[:-1]
        for line in lines:
            line = json.loads(str(line))
            documents.append(line['text'].split())
            labels.append(line['cluster'])
    return documents, labels


# 评估函数
def score(labels, prediction):
    result_NMI = metrics.normalized_mutual_info_score(labels, prediction)
    print("result_NMI:", result_NMI)


def f_KMeans(X, Y, random_state):
    print('k-means')
    K = len(set(Y))
    prediction = KMeans(n_clusters=K, random_state=random_state).fit_predict(X)
    score(Y, prediction)


def f_AffinityPropagation(X, Y):
    print('AffinityPropagation')
    prediction = AffinityPropagation().fit_predict(X)
    score(Y, prediction)


def f_MeanShift(X, Y):
    print('MeanShift')
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    prediction = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(X)
    score(Y, prediction)


def f_SpectralClustering(X, Y, gamma):
    print('spectral_clustering')
    K = len(set(Y))
    prediction = SpectralClustering(n_clusters=K, gamma=gamma).fit_predict(X)
    score(Y, prediction)


def f_AgglomerativeClustering(X, Y, linkage):
    print('AgglomerativeClustering: ' + linkage)
    K = len(set(Y))
    prediction = AgglomerativeClustering(n_clusters=K, linkage=linkage).fit_predict(X)
    score(Y, prediction)


def f_DBSCAN(X, Y, eps, min_samples):
    print('DBSCAN')
    prediction = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    score(Y, prediction)


def f_GaussianMixture(X, Y, cov_type):
    print('GaussianMixture: ' + cov_type)
    K = len(set(Y))
    gmm = GaussianMixture(n_components=K, covariance_type=cov_type).fit(X)
    prediction = gmm.predict(X)
    score(Y, prediction)


if __name__ == '__main__':
    documents, Y = read_data()
    dictionary = vsm.f_dictionary(documents, 'data/cluster-out/dictionary.csv', 0)
    # X = vsm.vsm(documents, dictionary)
    X = knn.TF_IDF2(documents, dictionary)
    # score:0.771
    f_KMeans(X, Y, random_state=13)
    # score: 0.733
    f_AffinityPropagation(X, Y)
    # score: 0.110
    f_MeanShift(X, Y)
    # score:0.759
    f_SpectralClustering(X, Y, gamma=0.06)
    #  score:
    #       'ward': 0.792
    #       'average': 0.167
    #       'complete': 0.463
    linkages = ['ward', 'average', 'complete']
    for linkage in linkages:
        f_AgglomerativeClustering(X, Y, linkage)
    # score: 0.733
    f_DBSCAN(X, Y, eps=0.3, min_samples=1)
    #  score:
    #       'spherical': 0.663
    #       'diag': 0.716
    #       'tied': 0.727
    #       'full': MemoryError
    cov_types = ['spherical', 'diag', 'tied', 'full']
    for cov_type in cov_types:
        f_GaussianMixture(X, Y, cov_type)
