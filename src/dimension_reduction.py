import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FactorAnalysis as FA

class DimensionReduction: # 定义一个类来封装PCA、LDA和FA
    def __init__(self, X, y, feature_names):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
    def pca(self, n_components):
        # 执行PCA
        pca = PCA(n_components=n_components)  # 选择n_components个主成分
        X_pca = pca.fit_transform(self.X_scaled)

        return X_pca

    def lda(self, n_components):
        # 执行LDA
        lda = LDA(n_components=n_components)
        X_lda = lda.fit_transform(self.X_scaled, y)

        return X_lda

    def fa(self, n_components):
        # 执行FA
        fa = FA(n_components=n_components)
        X_fa = fa.fit_transform(self.X_scaled)

        return X_fa
    
    def _rd(self, method, n_components):
        if method == 'pca':
            return self.pca(n_components)
        elif method == 'lda':
            return self.lda(n_components)
        elif method == 'fa':
            return self.fa(n_components)
        else:
            raise ValueError('Invalid method. Choose from pca, lda, or fa.')

if __name__ == '__main__':
    from sklearn.datasets import load_wine
    data = load_wine()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    dr = DimensionReduction(X, y, feature_names)
    X_dr = dr._rd('fa', 2)
    
    # 创建DataFrame以便于绘图
    df = pd.DataFrame(data=X_dr, columns=['Principal Component 1', 'Principal Component 2'])
    df['Class'] = y

    # 绘制PCA结果图
    plt.figure(figsize=(8, 6))
    for target in df['Class'].unique():
        plt.scatter(df[df['Class'] == target]['Principal Component 1'], 
                    df[df['Class'] == target]['Principal Component 2'], 
                    label=f'Class {target}')

    plt.title('PCA of Sonar Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()