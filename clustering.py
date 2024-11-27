from src.fetch_dataset import fetch_dataset as _fetch
import numpy as np

DATA_PATH = "Data\sonar\sonar.all-data"
K = 2
M = 2
MAX_ITERATIONS = 100

def _intialize_centroids(data: list, k: int):
    """
    初始化聚类中心。
    Parameters:
        data: 数据集
        k: 聚类数
    Returns:
        centroids: 随机选取的聚类中心
    """
    import random
    centroids = []
    for i in range(k):
        centroids.append(random.choice(data))
    return centroids

def _assign_clusters(data: list, centroids: list):
    """
    将数据分配到最近的聚类中心。
    Parameters:
        data: 数据集
        centroids: 聚类中心
    Returns:
        clusters: 数据分配的聚类
    """
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    return clusters

def _seek_centroids(clusters: list):
    """
    计算新的聚类中心。
    Parameters:
        clusters: 数据分配的聚类
    Returns:
        centroids: 新的聚类中心
    """
    centroids = []
    for cluster in clusters:
        centroids.append(np.mean(cluster, axis=0))
    return centroids

def kMeans(data: list, k: int, centroids: list):
    """
    k均值算法实现的无监督聚类。
    Parameters:
        data: 数据集
        k: 聚类数
        centroids: 初始聚类中心
    Returns:
        clusters: 数据分配的聚类
    """
    while True:
        clusters = _assign_clusters(data, centroids)
        new_centroids = _seek_centroids(clusters)
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters

def _FCM_update_centroids(data: list, memberships: list, m: int):
    """
    更新聚类中心。
    Parameters:
        centroids: 聚类中心
        clusters: 数据分配的聚类
        m: 模糊系数
    Returns:
        centroids: 新的聚类中心
    """
    centroids = [0 for _ in range(len(memberships))]
    for i in range(len(centroids)):
        for j in range(len(data)):
            centroids[i] += np.array(data[j]) * (memberships[i][j] ** m)
        centroids[i] /= sum(memberships[i][j] ** m for j in range(len(data)))
    return centroids

def _FCM_update_membership(data: list, centroids: list, m: int):
    """
    更新数据点的隶属度矩阵。
    Parameters:
        data: 数据集
        centroids: 聚类中心
        m: 模糊系数
    Returns:
        membership: 数据点的隶属度矩阵。
    """
    memberships = [[0 for _ in range(len(data))] for _ in range(len(centroids))]
    for i in range(len(centroids)):
        for j in range(len(data)):
            memberships[i][j] = (1 / np.sum(np.linalg.norm(np.array(data[j] - np.array(centroids[i]))))) ** (1 / (m - 1)) 
            memberships[i][j] /= np.sum((1 / np.linalg.norm(np.array(data[j] - np.array(centroid)))) ** (1 / (m - 1)) for centroid in centroids)

    return memberships

def FCM(data: list, m: int, centroids: list):
    """
    Fuzzy C-Means算法实现的无监督聚类。
    Parameters:
        data: 数据集
        m: 模糊系数
        centroids: 初始聚类中心
    Returns:
        clusters: 数据分配的聚类
    """
    local_centroids = centroids
    for _ in range(MAX_ITERATIONS):
        memberships = _FCM_update_membership(data, local_centroids, m)
        local_centroids = _FCM_update_centroids(data, memberships, m)
    clusters = _assign_clusters(data, centroids)
    return clusters

def _evaluate_a(cluster: list, cluster_data: list):
    """
    与同簇内样本的平均距离。
    Parameters:
        cluster: 数据分配的聚类
    Returns:
        a: 平均距离
    """
    a = 0
    for j in range(len(cluster)):
            a += np.linalg.norm(np.array(cluster_data) - np.array(cluster[j]))
    a /= len(cluster)
    return a

def _evaluate_b(clusters: list):
    """
    簇间最小距离。
    Parameters:
        cluster: 数据分配的聚类
    Returns:
        b: 平均距离矩阵
    """
    b = []
    tmp = []

    for i in range(len(clusters)):
        k = list(range(len(clusters)))
        k.remove(i)
        for j in range(len(clusters[i])):
            for _ in k:
                tmp.append(_evaluate_a(clusters[_], clusters[i][j]))
            if len(tmp) != 0:
                b.append(min(tmp))
                tmp = []
    return b

def _evaluate_s(clusters: list):
    """
    采取轮廓系数作为评价指标。
    Parameters:
        clusters: 数据分配的聚类
    Returns:
        s: 轮廓系数矩阵
    """
    b = _evaluate_b(clusters)
    a = []

    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            a.append(_evaluate_a(clusters[i], clusters[i][j]))
    
    s = []
    for i in range(len(a)):
        s.append((a[i] - b[i]) / max(a[i], b[i]))
    return s

if __name__ == "__main__":
    data_X, data_y = _fetch(DATA_PATH)
    centroids = _intialize_centroids(data_X, K)

    clusters = kMeans(data_X, K, centroids)
    s = _evaluate_s(clusters)

    import matplotlib.pyplot as plt
    i = list(range(len(s)))
    plt.plot(i, s)
    plt.xlabel("i")
    plt.ylabel("s")
    plt.show()