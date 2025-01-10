from sklearn.model_selection import StratifiedKFold
import json
import pandas as pd
import numpy as np
from typing import Union
from sklearn.decomposition import PCA

def _model_pred(X_input, model):
    return model.predict(X_input)

def run_pca(X_data: Union[pd.DataFrame, np.array], ratio=0.88):
    """
    对样本特征进行PCA降维，保留给定比例的主成分信息ratio
    :param X_data: 样本特征空间
    :param ratio: 保留多大比例的信息
    :return: 降维后结果
    """
    n = min(X_data.shape[0], X_data.shape[1])
    pca = PCA(n_components=n)
    pca.fit(X_data)
    X_new = pca.transform(X_data)
    # 筛选解释力占比ratio的主成分
    cols_need = np.where(np.cumsum(pca.explained_variance_ratio_) <= ratio)
    cols_need = list(cols_need[0])
    X_new = X_new[:, cols_need + [cols_need[-1] + 1]]
    return pd.DataFrame(X_new, columns=[f'pca_{x}' for x in range(X_new.shape[1])]), pca


def save_json(d:dict, json_path):
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=2)

def kfold_split(df:pd.DataFrame, labelname:str, k_cv:int, fit_type:str):
    # kfolder split
    assert fit_type in ['regression', 'classification']
    if fit_type == 'regression':
        boxes = 50
        df.loc[:, 'box'] = pd.qcut(df[labelname], q=boxes, duplicates='drop', labels=False)
    else:
        df.loc[:, 'box'] = df[labelname]
    skfold = StratifiedKFold(n_splits=k_cv, shuffle=True, random_state=2023)
    skfold_split = skfold.split(df.index, df.box)
    return skfold_split

def standardize_features(X:np.ndarray, mode='train', mean=0, std_dev=1):
    # feature standerdize
    if mode == 'train':
        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        return (X - mean) / std_dev, mean, std_dev
    else:
        return (X - mean) / std_dev