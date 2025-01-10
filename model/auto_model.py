import numpy as np
import pandas as pd
import logging
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.ensemble import (
                        RandomForestClassifier, 
                        AdaBoostClassifier, 
                        AdaBoostRegressor, 
                        RandomForestRegressor)
from functools import partial
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')
from logger import Logger
from sklearn.decomposition import PCA

# TODO 增加新的集成模型
# TODO 增加PCA降维技术 done
# TODO 增加PCA模型保存、增加归一化模型保存，用于预测阶段的数据预处理

class SetupBase(object):
    """
    初始化基模型、参数空间、评价指标、超参数搜索器
    """
    def __init__(self):
        self.fit_type = 'regression'
        self.parms_search = 'grid'
        self.fit_metric = None

    def _set_models(self):
        """
        初始化模型
        :return:分类问题与回归问题的模型字典
        """
        reg_keys = ['lasso', 'ridge', 'svm', 'cart', 'xgb', 'lgb', 'cab', 'adb', 'rdf']
        clf_keys = ['logit', 'svm', 'cart', 'xgb', 'lgb', 'cab', 'adb', 'rdf']
        lasso = Lasso
        ridge = Ridge
        logit = LogisticRegression
        svm_reg = SVR
        svm_clf = SVC
        cart_reg = DecisionTreeRegressor
        cart_clf = DecisionTreeClassifier
        xgb_clf = XGBClassifier
        xgb_reg = XGBRegressor
        lgb_reg = LGBMRegressor
        lgb_clf = LGBMClassifier
        cab_reg = CatBoostRegressor
        cab_clf = CatBoostClassifier
        adb_reg = AdaBoostRegressor
        adb_clf = AdaBoostClassifier
        rdf_reg = RandomForestRegressor
        rdf_clf = RandomForestClassifier
        reg_model = dict(zip(reg_keys,
                             [lasso, ridge, svm_reg, cart_reg, xgb_reg,
                              lgb_reg, cab_reg, adb_reg, rdf_reg]))
        clf_model = dict(zip(clf_keys,
                             [logit, svm_clf, cart_clf, xgb_clf, lgb_clf,
                              cab_clf, adb_clf, rdf_clf]))
        self.search_models = reg_model if self.fit_type == 'regression' else clf_model

    def _set_params(self):
        """
        根据不同的问题，初始化各个模型超参空间字典
        TODO 持续更新分类或回归参数空间
        """
        random_seed = 2023
        if self.fit_type == 'regression':
            cart_params = {'max_depth': [x for x in range(2, 15, 2)],
                           'criterion': ['mse', 'friedman_mse', 'mae'],
                           'min_samples_split': [6, 11, 21, 31]}
            xgb_params = {'max_depth': [x for x in range(2, 9, 2)],
                          'n_estimators': [25, 50, 75, 100],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'seed': [random_seed],
                          'importance_type': ['gain']}
            lgb_params = {'objective': ['regression', 'regression_l1'],
                          'boost': ['gbdt', 'dart'], # dart
                          'max_depth': [x for x in range(2, 9, 2)],
                          'num_leaves': [21, 31],
                          'min_data_in_leaf': [25, 50],
                          'bagging_fraction': [0.8, 1],
                          'n_estimators': [50, 100, 150],
                          # 'early_stopping_round': [70],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'random_state': [random_seed],
                          'verbosity': [-1],  # 隐藏警告信息
                          'importance_type': ['gain']}
            cab_params = {'max_depth': [x for x in range(2, 9, 2)],
                          'iterations': [25, 50, 75, 100],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'random_state': [random_seed],
                          'verbose': [False]}
            lasso_params = {'alpha': list([0.1 * x for x in range(1, 101)])}
            ridge_params = {'alpha': list([0.1 * x for x in range(1, 101)])}
            svm_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                          'C': [1, 10, 100, 1000, 5000, 10000]}
            adb_params = {'n_estimators': [25, 50, 75, 100],
                          'learning_rate': [1, 5e-1, 1e-1, 5e-2],
                          'random_state': [random_seed]}
            rdf_params = {'n_estimators': [20, 40, 60, 80],
                          'max_depth': [x for x in range(2, 9, 2)],
                          'random_state': [random_seed]}
            self.search_params = {'lasso': lasso_params,
                                  'ridge': ridge_params,
                                  'svm': svm_params,
                                  'cart': cart_params,
                                  'xgb': xgb_params,
                                  'lgb': lgb_params,
                                  'cab': cab_params,
                                  'adb': adb_params,
                                  'rdf': rdf_params
                                  }
        else:
            cart_params = {'max_depth': [x for x in range(2, 15, 2)],
                           'criterion': ['gini', 'entropy'],
                           'min_samples_split': [11, 21, 31]}
            xgb_params = {'max_depth': [x for x in range(2, 9, 2)],
                          'min_child_weight': [21, 31, 50],
                          'n_estimators': [25, 50, 75, 100],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'seed': [random_seed],
                          'importance_type': ['gain']}

            lgb_params = {'objective': ['binary'],
                          'boost': ['dart'],  # 'gbdt'
                          'max_depth': [x for x in range(2, 9, 2)],
                          'num_leaves': [21, 31],
                          'min_child_weight': [21, 31, 50],
                          'bagging_fraction': [0.8, 1],
                          'n_estimators': [50, 100, 150],
                          'early_stopping_round': [70],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'random_state': [random_seed],
                          'verbosity': [-1],  # 隐藏警告信息
                          'importance_type': ['gain']}
            cab_params = {'max_depth': [x for x in range(2, 9, 2)],
                          'iterations': [25, 50, 75, 100],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'random_state': [random_seed],
                          'verbose': [False]}
            logit_params = {'penalty': ['l1'],
                            'solver': ['saga'],
                            'C': list([0.1 * x for x in range(1, 101, 5)])}
            svm_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                          'C': [1, 10, 100, 1000, 5000, 10000]}
            adb_params = {'n_estimators': [25, 50, 75, 100],
                          'learning_rate': [1, 5e-1, 1e-1, 5e-2],
                          'random_state': [random_seed]}
            rdf_params = {'n_estimators': [20, 40, 60, 80],
                          'max_depth': [x for x in range(2, 9, 2)],
                          'random_state': [random_seed]}
            self.search_params = {'logit': logit_params,
                                  'svm': svm_params,
                                  'cart': cart_params,
                                  'xgb': xgb_params,
                                  'lgb': lgb_params,
                                  'cab': cab_params,
                                  'adb': adb_params,
                                  'rdf': rdf_params
                                  }

    @staticmethod
    def metric_rec_pre(y_true, y_pred):
        rec = metrics.recall_score(y_true, y_pred)
        pre = metrics.precision_score(y_true, y_pred)
        return (rec + pre) / 2

    def _set_metrics(self):
        """
        定义评估指标字典
        """
        reg = {'r2': metrics.r2_score,
               'mape': metrics.mean_absolute_percentage_error,
               'mse': metrics.mean_squared_error}
        clf = {'auc': metrics.accuracy_score,
               'recall': metrics.recall_score,
               'precision': metrics.precision_score,
               'rec_pre': self.metric_rec_pre,
               'f1': metrics.f1_score,
               'roc_auc': metrics.roc_auc_score}
        self.search_metrics = reg if self.fit_type == 'regression' else clf

    def _metric_fun(self, y_true, y_pred):
        """
        指标评价函数
        :param y_true: 真实值
        :param y_pred: 预测值
        :return:
        """
        self._set_metrics()
        if self.fit_type == 'regression':
            if self.fit_metric is None:
                self.fit_metric = 'r2'  # 回归问题默认评价指标为r2
        else:
            if self.fit_metric is None:
                self.fit_metric = 'auc'  # 分类问题默认是准确率
        return self.search_metrics[self.fit_metric](y_true, y_pred)

    def _set_seacher(self, model, param_dist: dict, scoring_fun, cv=None):
        """
        自定义超参寻优方法
        :param model: 模型
        :param param_dist: 参数空间
        :param scoring_fun: 评价函数， str or function
        :param cv: k折，int or 自定义划分方法
        :return:
        """
        if cv is None:
            cv = 4
        if self.parms_search == 'grid':
            searcher = GridSearchCV(model,
                                    param_grid=param_dist,
                                    cv=cv,
                                    scoring=scoring_fun)
        elif self.parms_search == 'random':
            searcher = RandomizedSearchCV(model,
                                          param_dist,
                                          cv=cv,
                                          scoring=scoring_fun,
                                          n_iter=50, random_state=42)
        else:
            searcher = BayesSearchCV(estimator=model,
                                     search_spaces=param_dist, n_jobs=-1, cv=cv,
                                     scoring=scoring_fun)
        return searcher


class AutoModel(SetupBase):
    """
    自动建模算法
    方案：训练多个模型，对多模型结果进行集成，用于分类、回归问题
    """
    def __init__(self, fit_type='regression', fit_metric=None, k_cv=4, metric_filter=0.8,
                 params_searcher='grid', pca_ratio=0.88, log_path='auto_model.log'):
        super(AutoModel, self).__init__()
        self.fit_type = fit_type                # 问题是回归还是分类问题
        self.fit_metric = fit_metric            # 模型多折验证评价指标
        self.k_cv = k_cv                        # 进行几折交叉验证
        self.metric_filter = metric_filter      # 过滤评价指标低于该值的基模型
        self.params_searcher = params_searcher  # 超参搜索器 grid or random or bayes
        self.pca_ratio = pca_ratio              # pca降维保留主成分信息比例，None表示不进行pca降维
        self.stack_model = {}
        if self.fit_type not in ['regression', 'classification']:
            raise ValueError(f'错误的值{self.fit_type}, fit_type取值为regression或classification')
        if self.params_searcher not in ['grid', 'random', 'bayes']:
            raise ValueError(f"错误的值{self.params_searcher}, params_searcher取值为['grid', 'random', 'bayes']")

        # 日志模块
        self.log = Logger(path=log_path)
        self.log('--' * 10 + f'自动化建模日志 建模类型{self.fit_type}' + '--' * 10)

    def _pca_trans(self, X_data: pd.DataFrame or np.array, ratio=0.88):
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
        # 筛选ratio解释力的主成分
        cols_need = np.where(np.cumsum(pca.explained_variance_ratio_) <= ratio)
        cols_need = list(cols_need[0])  # tuple()[0]
        X_new = X_new[:, cols_need + [cols_need[-1] + 1]]
        return pd.DataFrame(X_new, columns=[f'pca_{x}' for x in range(X_new.shape[1])])

    def _k_split(self, X_train, y_train):
        """
        将数据集按照标签分布拆分为k折
        :param df:数据集
        :param labelname:标签名称
        :return: 拆分后的标签
        """
        if self.fit_type == 'regression':
            boxes = 50
            box_ = pd.qcut(y_train, q=boxes,
                                       duplicates='drop', labels=False)
        else:
            box_ = y_train
        skfold = StratifiedKFold(n_splits=self.k_cv,
                                 shuffle=True,
                                 random_state=2023)
        skfold_split = skfold.split(X_train,
                                    box_)
        return skfold_split

    def _split_train_test(self, df, feture_ls, label_name):
        """
        划分训练集合测试集
        """
        if self.pca_ratio is not None:
            X_data = self._pca_trans(df[feture_ls], self.pca_ratio)
            self.log(f'执行PCA降维处理，原始特征数量{len(feture_ls)}，保留特征数量{X_data.shape[1]}，保留主成分信息比例{self.pca_ratio}')
        else:
            X_data = df[feture_ls]  # TODO不再复制一份
        # 划分测试集验证集
        # X_train, X_test, y_train, y_test = train_test_split(df[feture_ls], df[label_name],
        #                                                     test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_data, df[label_name],
                                                            test_size=0.2, random_state=2023)

        # 标准化
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train, self.y_test = y_train, y_test

    def best_fit(self, df: pd.DataFrame, feture_ls: list, label_name: str, models=None):
        """
        训练基模型，并寻找最优集成路径
        :param df: 数据集
        :param feture_ls: 使用的特征列名
        :param label_name: 标签名称
        :param models: 制定的基模型，默认None，采用所有预设基模型
        :return: 最优模型
        """
        self.log('--' * 5 + f'进行多模型训练并过滤，过滤阈值{self.metric_filter}' + '--' * 5)
        self._set_models()
        self._set_params()
        self._split_train_test(df, feture_ls, label_name)
        score_fun = metrics.make_scorer(self._metric_fun)   # make_score封装
        # 自定义基模型
        if models is not None:
            self.search_models = dict([item for item in self.search_models.items()
                                       if item[0] in models])
        # 多折寻找最优超参模型
        self.models_fit, self.train_matric, self.test_matric = {}, {}, {}
        mat_train, mat_test = {}, {}  # 保存训练集和验证集预测结果
        for i, (k, model) in enumerate(self.search_models.items()):
            if k == 'adb':  # 选择已搜索的cart树作为adb的基学习器
                self.search_params[k]['estimator'] = [self.models_fit['cart']]
            seacher = self._set_seacher(model(), self.search_params[k],
                                        scoring_fun=score_fun,
                                        cv=self._k_split(self.X_train, self.y_train))
            seacher.fit(self.X_train, self.y_train)
            best_params = seacher.best_params_
            model = model(**best_params)
            model.fit(self.X_train, self.y_train)

            # 记录训练集和验证集结果
            y_test_pred = model.predict(self.X_test)
            metric_test = self._metric_fun(self.y_test, y_test_pred)
            if metric_test >= self.metric_filter:
                # 只保留大于过滤阈值的结果
                y_train_pred = model.predict(self.X_train)
                metric_train = self._metric_fun(self.y_train, y_train_pred)
                self.train_matric[k] = metric_train
                self.test_matric[k] = metric_test
                self.models_fit[k] = model
                mat_train[k] = y_train_pred
                mat_test[k] = y_test_pred
                self.log(f'载入{k}模型，测试集评价指标={metric_test}, 最优超参{best_params}')
            else:
                self.log(f'过滤{k}模型，测试集评价指标={metric_test}, 最优超参{best_params}')

        if len(self.models_fit) == 0:
            self.log(f"模型过滤后为空, 请调整过滤阈值{self.metric_filter}或重新进行特征工程")
            raise ValueError(f"模型过滤后为空, 请调整过滤阈值{self.metric_filter}或重新进行特征工程")

        self.log('--' * 5 + f'基模型训练并过滤完成，寻找最优集成路径' + '--' * 5)
        self._set_ensemble_method()
        self.best_model = self.best_ensemble(mat_train, mat_test)

    def best_ensemble(self, mat_train: np.array or dict, mat_test: np.array or dict):
        """
        进行最优最优集成选型
        """
        self.log(f'训练集基模型效果{self.train_matric}')
        self._set_ensemble_method()
        if isinstance(mat_train, dict):
            mat_train = np.array(list(mat_train.values())).T  # shape = (n_samples, n_models)
            mat_test = np.array(list(mat_test.values())).T
        ensemble_metric = {}
        for k, ensemble in self.ensembler.items():
            if k not in ['stack', 'stack_cart']:
                m = self._metric_fun(self.y_test, ensemble(mat_test))
            else:
                m = self._metric_fun(self.y_test, ensemble(mat_train, self.y_train, mat_test))
            ensemble_metric[k] = m
            self.log(f'集成方法{k}测试集效果={m}')
        # 对比训练集筛选最好的集成方法
        k_e, m_test_e = max(ensemble_metric.items(), key=lambda x: x[1])
        # k_train, m_train = max(self.train_matric.items(), key=lambda x: x[1])
        k_test, m_test = max(self.test_matric.items(), key=lambda x: x[1])
        # 更新集成函数字典
        for sk in ['stack', 'stack_cart']:
            self.ensembler[sk] = partial(self._model_pred,
                                                        model=self.stack_model[sk])
        self.log(f'最优集成方法验证集效果={m_test_e}, 单模型验证集最好效果={m_test}')
        if m_test_e > m_test:
            # 集成路径有效
            self.log(f'集成路径{k_e}最优, 验证集评估指标={m_test_e}')
            return 1, k_e, self.ensembler[k_e]
        else:
            self.log(f'所有集成路径无效, 最优模型为{k_test}, 模型评估指标={m_test}')
            return 0, k_test, partial(self._model_pred, model=self.models_fit[k_test])

    def predict(self, X_feature: pd.DataFrame or np.array):
        """
        结果预测
        :param X_feature:特征数据
        :return:
        """
        opt, k_, func = self.best_model
        if opt:  # 多模型集成
            mat_pred = np.zeros((X_feature.shape[0], len(self.models_fit)))
            for i, (k, model) in enumerate(self.models_fit.items()):
                mat_pred[:, i] = model.predict(X_feature)
            return func(mat_pred)
        else:
            return func(X_feature)

    """
    集成方案设计：
    分类和回归：结果平均、结果加权、stack; stack只选择最简单的回归和cart树模型
    分类: 硬投票
    自动寻找最好的集成路径
    """
    def _model_pred(self, X_input, model):
        """
        模型预测方法封装
        """
        return model.predict(X_input)

    def _set_ensemble_method(self):
        """
        初始化集成方法
        """
        if self.fit_type == 'regression':
            self.ensembler = {'softw': partial(self._voting, weight=self.test_matric),
                              'soft': self._voting,
                              'stack': self._stacking,
                              'stack_cart': partial(self._stacking, base_mode='cart')}

        else:
            self.ensembler = {'softw': partial(self._voting, weight=self.test_matric),
                              'soft': self._voting,
                              'hard': partial(self._voting, soft=False),
                              'stack': self._stacking,
                              'stack_cart': partial(self._stacking, base_mode='cart')}

    def _voting(self, mat_rslt: np.array, soft=True, weight=None):
        """
        适用分类的概率结果或回归的数值结果加权
        """
        if soft:
            if weight is None:
                result = np.mean(mat_rslt, axis=1)
            else:
                if isinstance(weight, dict):
                    w = np.array(list(weight.values()))
                else:
                    w = weight
                w = w * w
                w = w / sum(w)  # 归一化
                result = np.sum(mat_rslt * w, axis=1)
            if self.fit_type == 'classification':
                # 0-1分割概率阈值0.5
                result[np.where(result < 0.5)] = 0
                result[np.where(result >= 0.5)] = 1
        else:
            # 硬投票适用于分类问题
            thre = np.floor(mat_rslt.shape[1] / 2)
            result = np.sum(mat_rslt, axis=1) - thre
            result[np.where(result <= 0)] = 0
            result[np.where(result > 0)] = 1
        return result

    def _stacking(self, mat_train: np.array, y_train, mat_test: np.array, base_mode=None):
        """
        satacking基模型不能太复杂
        回归选择cart lasso
        分类选择logist cart
        """
        if base_mode is None:
            if self.fit_type == 'regression':
                base_mode = 'lasso'
            else:
                base_mode = 'logit'
        self._set_models()
        self._set_params()
        if base_mode == 'cart':
            # 树模型要求要简单
            self.search_params[base_mode] = {'max_depth': [1, 2, 3]}
        score_fun = metrics.make_scorer(self._metric_fun)  # make_score封装
        model = self.search_models[base_mode]
        # 寻找最优参数
        seacher = self._set_seacher(model(), self.search_params[base_mode],
                                    scoring_fun=score_fun,
                                    cv=self._k_split(mat_train, y_train))
        seacher.fit(mat_train, y_train)
        model = model(**seacher.best_params_)
        model.fit(mat_train, y_train)
        if base_mode == 'cart':
            self.stack_model['stack_cart'] = model  # 用于预测阶段
        else:
            self.stack_model['stack'] = model
        return model.predict(mat_test)

    def save_model(self, path=None):
        """
        path: path_like str, e.g path/file/
        """
        if path is None:
            path = r'./auto_model'
            if not os.path.exists(path):
                os.mkdir(path)
        opt, k_name, func = self.best_model
        if opt:
            # 保存多个模型以及集成方法
            for k, model in self.models_fit.items():
                joblib.dump(model, os.path.join(path, f'{k_name}_{k}.pkl'))  # 保存多个集成方式_模型.pkl
            # stack还有再多保存一个模型
            if k_name in ['stack', 'stack_cart']:
                joblib.dump(self.stack_model[k_name],
                            os.path.join(path, f'{k_name}_{k_name}.pkl'))
            if k_name == 'softw':
                json.dump(self.test_matric, fp=os.path.join(path, 'weight.json'))
        else:
            model = func.keywords['model']
            joblib.dump(model, os.path.join(path, f'{k_name}.pkl'))  # 保存一个模型.pkl

    def load_model(self, path=None):
        if path is None:
            path = r'./auto_model'
        # 解析通道类型
        files = [x for x in os.listdir(path) if x.split('.')[1] == 'pkl']
        if len(files) == 1:
            # 只载入一个模型
            k = files[0].split('.')[0]
            model = joblib.load(os.path.join(path, files[0]))
            self.best_model = 0, k, partial(self._model_pred, model=model)
        else:
            k_ensemble = files[0].split('_')[0]
            self.models_fit = {}
            for f in files:
                if f.endswith('json'):
                    self.test_matric = json.load(os.path.join(path, f))
                else:
                    self.models_fit[f.split('_')[1].split('.')[0]] = joblib.load(os.path.join(path, f))
            # 进行初始化
            self._set_ensemble_method()
            if k_ensemble in ['stack', 'stack_cart']:
                model_stack = self.models_fit.pop(k_ensemble)
                self.ensembler[k_ensemble] = partial(self._model_pred,
                                                                    model=model_stack)
            self.best_model = 1, k_ensemble, self.ensembler[k_ensemble]

# TODO 增加评价阈值过滤
if __name__ == '__main__':
    # 功能测试
    # automodel = AutoModel(fit_type='regression', fit_metric='r2')
    automodel = AutoModel(fit_type='classification', fit_metric='rec_pre')
    df = pd.read_csv(r'E:\02code\01_EasyPlot\sample.csv')
    # feature = ['feature_3', 'feature_15', 'feature_8', 'feature_270', 'feature_25', 'feature_188',
    #            'feature_281', 'feature_250', 'feature_294', 'feature_299'] # reg
    feature = ['feature_18', 'feature_3', 'feature_25', 'feature_294',
                   'feature_12', 'feature_23', 'feature_7', 'feature_8']  # clf
    label_name = 'price'
    df['price'] = pd.qcut(df['price'], q=2, labels=[x for x in range(2)])
    automodel.best_fit(df, feature, label_name)
    y_pred = automodel.predict(df[feature])
    automodel.save_model(path=None)

    # 载入预测: 已测试单个模型无集成策略 stack集成 TODO 待测试加权集成
    automodel.load_model(path=None)
    yp = automodel.predict(df[feature])
    print(yp)

