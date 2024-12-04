from src.fetch_dataset import fetch_dataset
import numpy as np
import random

DATA_PATH = "Data\sonar\sonar.all-data" # 数据集路径
C = 1 # 惩罚因子
TOL = 0.001 # 容忍误差
MAX_ITER = 100 # 最大迭代次数

LABEL = ["M", "R"]

def _clean_label(label: list):
    """
    处理标签。
    """
    label_cleaned = []
    for i in label:
        if i == LABEL[0]:
            label_cleaned.append(-1)
        else:
            label_cleaned.append(1)
    return label_cleaned

def _initialize_alpha(data: list, c: int):
    """
    初始化alpha。
    """
    n = len(data)
    alpha = [random.random() * c for _ in range(n)]
    return alpha

def _evaluate_deviation(xi_list: list, xj_list_sets: list, yi: float, yj_sets: list, alpha_j_sets: list, b: float):
    """
    计算误差。
    """
    E = 0
    xi = np.array(xi_list).T
    for j in range(len(xj_list_sets)):
        xj = np.array(xj_list_sets[j]).T
        E += yj_sets[j] * alpha_j_sets[j] * xi.T * xj
    E = E + b - yi
    return E

def _calculate_L_H_matirx(alpha_i: list, alpha_j: list, yi: list, yj: list, c: int):
    """
    计算L和H。
    """
    L = [0 for _ in range(len(alpha_i))]
    H = [0 for _ in range(len(alpha_i))]
    for i in range(len(alpha_i)):
        if yi[i] == yj[i]:
            L[i] = max(0, alpha_i[i] + alpha_j[i] - c)
            H[i] = min(c, alpha_i[i] + alpha_j[i])
        else:
            L[i] = max(0, alpha_j[i] - alpha_i[i])
            H[i] = min(c, c + alpha_j[i] - alpha_i[i])
    return L, H

def _calculate_eta(xi_list: list, xj_list: list):
    """
    计算eta。
    """
    xi = np.array(xi_list).T
    xj = np.array(xj_list).T
    eta = xi.T * xi + xj.T * xj - 2 * xi.T * xj
    return eta

def _update_alpha_j(alpha_j_old: float, E_i: float, E_j: float, yj: float, eta: float):
    """
    更新alpha_j。
    """
    alpha_j_new = alpha_j_old + (E_i - E_j) * yj / eta
    return alpha_j_new

def _clip_alpha_j(alpha_j: float, L: float, H: float):
    """
    修剪alpha_j。
    """
    alpha_j = alpha_j[0]
    alpha_j_new = max(L, min(H, alpha_j))
    return alpha_j_new

def _updata_alpha_i(alpha_i_old: float, alpha_j_old: float, alpha_j_new: float, yi: float, yj: float):
    """
    更新alpha_i。
    """
    alpha_i_new = alpha_i_old + yi * yj * (alpha_j_old - alpha_j_new)
    return alpha_i_new

def _update_b(b_old: float, alpha_i_new: float, alpha_i_old: float, alpha_j_new: float, alpha_j_old: float, yi: float, yj: float, xi_list: list, xj_list: list, Ei: float, Ej: float):
    """
    更新b。
    """
    xi = np.array(xi_list).T
    xj = np.array(xj_list).T
    b1 = b_old - Ei - yi * (alpha_i_new - alpha_i_old) * xi.T * xi - yj * (alpha_j_new - alpha_j_old) * xi.T * xj
    b2 = b_old - Ej - yi * (alpha_i_new - alpha_i_old) * xi.T * xj - yj * (alpha_j_new - alpha_j_old) * xj.T * xj
    if alpha_i_new > 0 and alpha_i_new < C:
        b_new = b1
    elif alpha_j_new > 0 and alpha_j_new < C:
        b_new = b2
    else:
        b_new = (b1 + b2) / 2
    return b_new

def svm_train(X: list, y: list, c: int, tol: float, max_iter: int):
    """
    训练SVM。

    Parameters:
    X: 训练数据集。
    y: 训练数据集的标签。
    C: 惩罚参数。
    tol: 收敛条件。
    max_iter: 最大迭代次数。

    Returns:
    alpha: 拉格朗日乘子。
    b: 偏置项。
    """
    b = [0 for _ in range(len(X))]
    m, n = np.shape(X) # m: 样本数量，n: 特征数量
    alphas_i = _initialize_alpha(X, c)
    alphas_j = _initialize_alpha(X, c)
    iter_num = 0 # 迭代次数
    L = []
    H = []
    eta = [_calculate_eta(x, x) for x in X]
    while iter_num < max_iter:
        Ei = [_evaluate_deviation(x, X, y[X.index(x)], y, alphas_j, b[X.index(x)]) for x in X]
        Ej = [_evaluate_deviation(x, X, y[X.index(x)], y, alphas_i, b[X.index(x)]) for x in X]
        L, H = _calculate_L_H_matirx(alphas_i, alphas_j, y, y, c)
        
        alphs_i_old = alphas_i
        alphs_j_old = alphas_j
        for j in range(m):
            alphas_j[j] = _update_alpha_j(alphs_j_old[j], Ei[j], Ej[j], y[j], eta[j])
            alphas_j[j] = _clip_alpha_j(alphas_j[j], L[j], H[j])

        for i in range(m):
            alphas_i[i] = _updata_alpha_i(alphs_i_old[i], alphs_j_old[i], alphas_j[i], y[i], y[i])

        for i in range(m):
            b[i] = _update_b(b[i], alphas_i[i], alphs_i_old[i], alphas_j[i], alphs_j_old[i], y[i], y[i], X[i], X[i], Ei[i], Ej[i])

        iter_num += 1
    return alphas_i, b

def _calculate_w(X: list, y: list, alphas: list):
    """
    计算权重w。
    """
    w = np.zeros(len(X[0]))
    for i in range(len(X)):
        w += alphas[i] * y[i] * np.array(X[i])
    return w

def _predict(tests: list, w: np.ndarray, b: float):
    """
    预测。
    """
    results = []
    for i in range(len(tests)):
        result = w * np.array(tests[i]).T + b
        result = result[0]
        if result > 0:
            results.append(1)
        else:
            results.append(-1)
    return results

def _evaluate_accuracy(y: list, y_pred: list):
    """
    计算准确率。
    """
    return np.sum(np.array(y_pred) == np.array(y))/ len(y)

if __name__ == '__main__':
    data_X, data_Y = fetch_dataset(DATA_PATH)
    label_cleaned = _clean_label(data_Y)
    X = data_X
    y = label_cleaned
    alphas, b = svm_train(X, y, c=C, tol=TOL, max_iter=MAX_ITER)

    w = _calculate_w(X, y, alphas)
    b = sum(b) / len(b)
    results = _predict(X, w, b)
    accuracy = _evaluate_accuracy(y, results)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))