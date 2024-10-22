from src.fetch_dataset import fetch_dataset
from src.split_dataset import split_dataset
from collections import Counter
import numpy as np

DATA_PATH = "Data\processed\iris_precessed.data"
SPLIT_RATIO = 0.2
K = range(1, 31)

def judge_target(X_test_data: np.array, X_train: np.array, y_train: np.array, k: int):
    """
    Parameters:
        `X_test_data`: 测试集中的目标数据
        `X_train`: 训练集
        `y_train`: 训练集标签
        `k`: k值
    Returns:
        `target`: 目标数据所属的类别
    """

    distances = [np.sqrt(np.sum((X_test_data - X_train_data)**2)) for X_train_data in X_train]

    sorted_indices = np.argsort(distances)
    top_k_indices = sorted_indices[:k]
    top_k_labels = y_train[top_k_indices]

    target = Counter(top_k_labels).most_common(1)[0][0]
    return target

def evaluate_accuracy(k: int, i: int, X_splited: list, y_splited: list):
    """
    Parameters:
        `k`: k值
        `i`: 当前测试集的索引
        `X_splited`: 数据集被切分后的列表
        `y_splited`: 标签集被切分后的列表
    Returns:
        `accuracy`: 准确率
    """
    X_test = X_splited[i]
    y_test = y_splited[i]
    X_train = []
    y_train = []

    for j in range(len(X_splited)):
        if j != i:
            X_train += X_splited[j]
            y_train += y_splited[j]

    y_pred = [judge_target(X_test_data = np.array(x), X_train = np.array(X_train), y_train = np.array(y_train), k = k) for x in X_test]
    
    accuracy = np.sum(np.array(y_pred) == np.array(y_test))/ len(y_test)
    return accuracy

def kNN_classifier(X_splited: list, y_splited: list, K: range):
    """
    基于kNN算法的分类器
    Parameters:
        `X_splited`: 数据集被切分后的列表
        `y_splited`: 标签集被切分后的列表
        `K`: k值范围
    """
    for k in K:
        accuracy = np.mean([evaluate_accuracy(k, i, X_splited, y_splited) for i in range(len(X_splited))])
        print(f"K = {k}, accuracy = {accuracy}")

if __name__ == "__main__":
    X, y = fetch_dataset(file_path = DATA_PATH)
    X_splited = split_dataset(X, split_ratio = SPLIT_RATIO)
    y_splited = split_dataset(y, split_ratio = SPLIT_RATIO)
    kNN_classifier(X_splited, y_splited, K)