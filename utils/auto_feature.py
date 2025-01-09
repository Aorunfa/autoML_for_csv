import numpy as np
import pandas as pd
import logging
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, chi2
from skopt import BayesSearchCV
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')
from logger import Logger

class SetupBase(object):
    """
    初始化基模型、超参空间、评价指标、超参数搜索器
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
        reg_keys = ['lasso', 'cart', 'xgb', 'lgb', 'cab']
        clf_keys = ['logit', 'cart', 'xgb', 'lgb', 'cab']
        lasso = Lasso
        logit = LogisticRegression
        cart_reg = DecisionTreeRegressor
        cart_clf = DecisionTreeClassifier
        xgb_clf = XGBClassifier
        xgb_reg = XGBRegressor
        lgb_reg = LGBMRegressor
        lgb_clf = LGBMClassifier
        cab_reg = CatBoostRegressor
        cab_clf = CatBoostClassifier

        reg_model = dict(zip(reg_keys,
                             [lasso, cart_reg, xgb_reg, lgb_reg, cab_reg]))
        clf_model = dict(zip(clf_keys,
                             [logit, cart_clf, xgb_clf, lgb_clf, cab_clf]))
        self.search_models = reg_model if self.fit_type == 'regression' else clf_model

    def _set_params(self):
        """
        初始化各个模型超参空间字典
        """
        random_seed = 2023
        if self.fit_type == 'regression':
            # 自定义回归问题参数空间
            cart_params = {'max_depth': [x for x in range(2, 15, 2)]}
            xgb_params = {'max_depth': [x for x in range(2, 11, 2)],
                          'n_estimators': [25, 50, 75, 100],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'seed': [random_seed],
                          'importance_type': ['gain']}
            lgb_params = {'objective': ['regression'],
                          'max_depth': [x for x in range(2, 11, 2)],
                          'n_estimators': [30, 60, 100],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'random_state': [random_seed],
                          'verbosity': [-1],  # 隐藏警告信息
                          'importance_type': ['gain']}
            cab_params = {'max_depth': [x for x in range(2, 11, 2)],
                          'iterations': [25, 50, 75, 100],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'random_state': [random_seed],
                          'verbose': [False]}
            lasso_params = {'alpha': list([0.1 * x for x in range(1, 101, 5)])}
            logit_params = {'penalty': ['l1'],
                            'solver': ['saga'],
                            'C': list([0.1 * x for x in range(1, 101, 5)])}
            self.search_params = {'lasso': lasso_params,
                                  'logit': logit_params,
                                  'cart': cart_params,
                                  'xgb': xgb_params,
                                  'lgb': lgb_params,
                                  'cab': cab_params
                                  }
        else:
            cart_params = {'max_depth': [x for x in range(2, 15, 2)]}
            xgb_params = {'max_depth': [x for x in range(2, 11, 2)],
                          'n_estimators': [25, 50, 75, 100],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'seed': [random_seed],
                          'importance_type': ['gain']}
            lgb_params = {'objective': ['classification'],
                          'max_depth': [x for x in range(2, 11, 2)],
                          'n_estimators': [30, 60, 100],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'random_state': [random_seed],
                          'verbosity': [-1],  # 隐藏警告信息
                          'importance_type': ['gain']}
            cab_params = {'max_depth': [x for x in range(2, 11, 2)],
                          'iterations': [25, 50, 75, 100],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'random_state': [random_seed],
                          'verbose': [False]}
            lasso_params = {'alpha': list([0.1 * x for x in range(1, 101, 5)])}
            logit_params = {'penalty': ['l1'],
                            'solver': ['saga'],
                            'C': list([0.1 * x for x in range(1, 101, 5)])}
            self.search_params = {'lasso': lasso_params,
                                  'logit': logit_params,
                                  'cart': cart_params,
                                  'xgb': xgb_params,
                                  'lgb': lgb_params,
                                  'cab': cab_params
                                  }

    @staticmethod
    def metric_rec_pre(y_true, y_pred):
        rec = metrics.recall_score(y_true, y_pred)
        pre = metrics.precision_score(y_true, y_pred)
        return (rec + pre) / 2

    def _set_metrics(self):
        # 定义评估指标字典
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
        进行指标评价
        :param y_pred:
        :param y_true:
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


class AutoFeature(SetupBase):
    """
    自动筛选特征算法，输出最终k个最重要的特征 -- 非全局最优优组合
    """
    def __init__(self, corrval_withlabel=0.35, corrval_withothers=0.85, p_val=0.05,
                 fit_type='regression', fit_metric=None, k_cv=4, top_n=50,
                 group_n=-1, params_searcher='grid', log_path='auto_feature.log'):
        super(AutoFeature, self).__init__()
        # 日志模块
        self.log = Logger(path=log_path)
        self.log('--'*10 + '特征自动筛选日志' + '--'*10)
        self.fit_type = fit_type                        # 问题是回归还是分类问题
        # 过滤式方法参数
        self.corrval_withlabel = corrval_withlabel      # 与标签最低的相关性系数
        self.corrval_withothers = corrval_withothers    # 特征间最大的相关系数
        self.p_val = p_val                              # 独立性检验p值
        # 嵌套式方法参数
        self.fit_metric = fit_metric                    # 模型多折验证评价指标
        self.k_cv = k_cv                                # 进行几折交叉验证
        self.top_n = top_n                              # 保留重要性topN的特征组
        # 包裹式方法参数
        self.group_n = group_n                          # 包裹式筛选的最大特征数量
        # 超参搜索器
        self.params_searcher = params_searcher          # random or bayes

        if self.fit_type not in ['regression', 'classification']:
            raise ValueError(f'错误的值{self.fit_type}, fit_type取值为regression或classification')
        if self.params_searcher not in ['grid', 'random', 'bayes']:
            raise ValueError(f"错误的值{self.params_searcher}, params_searcher取值为['grid', 'random', 'bayes']")

    def _k_split(self, df, labelname):
        """
        将数据集按照标签分布拆分为k折
        :param df:数据集
        :param labelname:标签名称
        :return: 拆分后的标签
        """
        if self.fit_type == 'regression':
            boxes = 50
            df.loc[:, 'box'] = pd.qcut(df[labelname], q=boxes,
                                       duplicates='drop', labels=False)
        else:
            df.loc[:, 'box'] = df[labelname]
        skfold = StratifiedKFold(n_splits=self.k_cv,
                                 shuffle=True,
                                 random_state=2023)
        skfold_split = skfold.split(df.index,
                                    df.box)
        return skfold_split

    def _standardize_features(self, X, mode='train', mean=0, std_dev=1):
        """
        特征标准化
        :param X:
        :param mode:
        :param mean:
        :param std:
        :return:
        """
        if mode == 'train':
            mean = np.mean(X, axis=0)
            std_dev = np.std(X, axis=0)
            return (X - mean) / std_dev, mean, std_dev
        else:
            return (X - mean) / std_dev

    def _indpendent_test(self, dist_feature, dist_target: pd.Series,
                         box_trans=False, method='ch2'):
        """
        变量独立性检验：卡方检验、互信息计算数值、KL散度
        使用于一组变量为类别变量
        :param dist_feature: 特征数组
        :param dist_target: 标签数组
        :param box_trans: 是否对特征数组进行分箱，用于数值特征离散化
        :param method: 选择独立性检验方法
        :return:
        """
        boxes = 20
        if box_trans:  # 特征离散化
            dist_feature = pd.qcut(dist_feature, q=boxes,
                                   duplicates='drop', labels=False)  # 自适应最小unique
        if self.fit_type == 'regression':  # 标签离散化
            dist_target = pd.qcut(dist_target, q=boxes,
                                  duplicates='drop', labels=False)
        # 转换为二维矩阵
        dist_feature = np.asarray(dist_feature).reshape(-1, 1)
        dist_target = np.asarray(dist_target).reshape(-1, 1)
        if method == 'ch2':
            # _, p = chi2(dist_feature, dist_target)
            p = chi2(dist_feature, dist_target)[1][0]  # p<0.05 拒绝原假设 变量不独立
            if p <= self.p_val:
                return p, False
            return p, True
        else:
            # kl散度 互信息方法
            if box_trans:
                mi = mutual_info_regression(dist_feature, dist_target)[0]  # 越趋近0越独立
            else:
                mi = mutual_info_classif(dist_feature, dist_target)[0]
            if mi >= 0.2:
                return mi, False
            return mi, True

    def filtering_reg(self, df, feature_num, feature_clf, label_name):
        """
        根据相关性阈值过滤特征
        :param df: 数据集
        :param feature_num: 特征列名 list
        :param label_name: 标签名 col
        :return:
        """
        # 过滤与标签不相关的数值特征
        corr_withlabel = df[feature_num].corrwith(df[label_name],
                                                  method='pearson').abs().fillna(0)
        corr_withlabel = corr_withlabel[corr_withlabel > self.corrval_withlabel]
        if corr_withlabel.shape[0] == 0:
            self.log('无与标签相关的数值特征，重新进行特征工程')
            col_filter = []
        else:
            self.log('过滤与标签无关的数值特征: {}'.format([x for x in feature_num if x not in corr_withlabel.index]))
            if corr_withlabel.shape[0] >= 2:
                corr_withlabel = corr_withlabel.sort_values(ascending=False)
                # 过滤特征之间的相关的组合
                corr_withothers = df[corr_withlabel.index].corr(method='pearson').abs()
                n = 0
                col_drop = []
                while n <= len(corr_withothers.columns) - 1:
                    col = corr_withothers.columns[n]
                    corr_del = corr_withothers[col][corr_withothers[col] >= self.corrval_withothers]
                    corr_del = corr_del.drop(index=col)
                    if len(corr_del.index) > 0:
                        for col_ in corr_del.index:
                            corr_withothers = corr_withothers.drop(index=col_, columns=col_)
                            col_drop.append(col_)
                    n += 1
                col_filter = corr_withothers.columns.to_list()
                self.log(f'过滤自相关特征: {col_drop}')
            else:
                col_filter = list(corr_withlabel.index)

        # 过滤与标签独立的分类变量
        col_drop = []
        for f in feature_clf:
            p, jude = self._indpendent_test(df[f], df[label_name])
            if jude:
                col_drop.append(f)
        col_filter2 = [x for x in feature_clf if x not in col_drop]
        self.log(f'过滤与标签无关的分类特征: {col_drop}')
        self.log(f'剩余数值特征: {col_filter}')
        self.log(f'剩余分类特征: {col_filter2}')
        return df[col_filter + col_filter2 + [label_name]], col_filter, col_filter2

    def filtering_clf(self, df, feature_num, feature_clf, label_name):
        """
        根据相关性阈值过滤分类变量特征
        :param df: 数据集
        :param feture_ls: 特征列名 list
        :param label_name: 标签名 col
        :return:
        """
        # 过滤与标签独立的数值变量
        col_drop = []
        p_ls = []
        for f in feature_num:
            p, jude = self._indpendent_test(df[f], df[label_name], box_trans=True)
            if jude:
                col_drop.append(f)
            else:
                p_ls.append(p)
        col_filter = [x for x in feature_num if x not in col_drop]
        self.log(f'过滤与标签无关的数值特征: {col_drop}')

        # 过滤自相关的数值变量
        if len(col_filter) >= 2:
            # 按照p值降序排序
            col_dict = dict(sorted(dict(zip(col_filter, p_ls)).items(), key=lambda x: x[1])[::-1])
            corr_withothers = df[col_dict.keys()].corr(method='pearson').abs()
            n = 0
            col_drop = []
            while n <= len(corr_withothers.columns) - 1:
                col = corr_withothers.columns[n]
                corr_del = corr_withothers[col][corr_withothers[col] >= self.corrval_withothers]
                corr_del = corr_del.drop(index=col)
                if len(corr_del.index) > 0:
                    for col_ in corr_del.index:
                        corr_withothers = corr_withothers.drop(index=col_, columns=col_)
                        col_drop.append(col_)
                n += 1
            col_filter = corr_withothers.columns.to_list()
            self.log(f'过滤自相关的数值特征: {col_drop}')

        # 过滤与标签独立的分类变量
        col_drop = []
        for f in feature_clf:
            p, jude = self._indpendent_test(df[f], df[label_name])
            if jude:
                col_drop.append(f)
        col_filter2 = [x for x in feature_clf if x not in col_drop]
        self.log(f'过滤与标签无关的分类特征: {col_drop}')
        self.log(f'剩余数值特征: {col_filter}')
        self.log(f'剩余分类特征: {col_filter2}')
        return df[col_filter + col_filter2 + [label_name]], col_filter, col_filter2

    def filtering(self, df:pd.DataFrame,
                  feature_num:list, feature_clf:list, label_name:str):
        """
        以指标过滤式进行特征筛选
        :param df:数据集
        :param feature_num:数值特征列
        :param feature_clf:分类特征列
        :param label_name:标签列名
        :return:过滤后的数据集，数值特征列，分类特征列
        """
        self.log('--'*5 + f'进行过滤式操作, 操作类型{self.fit_type}' + '--'*5)
        if self.fit_type == 'regression':
            return self.filtering_reg(df, feature_num, feature_clf, label_name)
        else:
            return self.filtering_clf(df, feature_num, feature_clf, label_name)

    def nesting(self, df, feture_ls, label_name):
        """
        根据基准模型特征排序重要性进行嵌套式过滤
        :param df: 数据集
        :param feture_ls: 特征列名 list
        :param label_name: 标签名 col
        :return:
        """
        self.log('--' * 5 + f'进行嵌套式过滤' + '--' * 5)
        self._set_models()
        self._set_params()
        score_fun = metrics.make_scorer(self._metric_fun)  # make_score封装
        # 初始化
        feture_importance = np.zeros((len(self.search_models), len(feture_ls)))
        weights = np.zeros(len(self.search_models))
        X_search, _, _ = self._standardize_features(df[feture_ls])
        for i, (k, model) in enumerate(self.search_models.items()):
            mat_importance = np.zeros((self.k_cv, len(feture_ls)))
            ls_metrics = []
            # 确定最优参数
            seacher = self._set_seacher(model(), self.search_params[k],
                                        scoring_fun=score_fun,
                                        cv=self._k_split(df, label_name))
            seacher.fit(X_search, df[label_name])
            best_params = seacher.best_params_
            model = model(**best_params)
            # 获取最优评分
            for j, (train_idx, valid_idx) in enumerate(self._k_split(df, label_name)):
                X_train, y_train = df.loc[train_idx, feture_ls], \
                                   df.loc[train_idx, label_name]
                X_valid, y_valid = df.loc[valid_idx, feture_ls], \
                                   df.loc[valid_idx, label_name]
                # 标准化
                X_train, u, std = self._standardize_features(X_train)
                X_valid = self._standardize_features(X_valid, 'valid', u, std)
                model.fit(X_train, y_train)
                try:
                    feature_importance = np.abs(model.feature_importances_)
                except:
                    feature_importance = np.abs(model.coef_)
                feature_importance = feature_importance / np.sum(feature_importance)  # 归一化
                val_metrics = self._metric_fun(y_valid, model.predict(X_valid))
                # 记录中间数据
                mat_importance[j] = feature_importance
                ls_metrics.append(val_metrics)
            # 指标处理
            feture_importance[i] = np.mean(mat_importance, axis=0)
            weights[i] = np.mean(ls_metrics)
            self.log(f'模型{k}权值: {weights[i]}, 特征评分向量：{feture_importance[i]}')

        # 多模型评分加权
        weights_ = np.reshape(weights / sum(weights), (-1, 1))
        feture_importance = tuple(zip(feture_ls, np.sum(feture_importance * weights_, axis=0)))
        feture_importance = sorted(feture_importance, key=lambda x: x[1])[::-1]
        self.log(f'最终特征评分结果{dict(feture_importance)}')
        self.log(f'评估模型指标排序：{sorted(tuple(zip(self.search_models.keys(), weights)), key=lambda x:x[1])[::-1]}')
        return dict(feture_importance[:self.top_n])  # 取top_n的重要特征组

    def wrapping(self, df, feature_ls: list, label_name: str, base_model='cart', group_n=-1):
        """
        包裹式特征筛选
        :param df:
        :param feature_ls:
        :param label_name:
        :param base_model:
        :return:
        """
        self.log('--' * 5 + f'进行包裹式过滤, 基模型{base_model}' + '--' * 5)
        self._set_models()
        self._set_params()
        score_fun = metrics.make_scorer(self._metric_fun)  # make_score封装
        if group_n == -1:
            self.group_n = len(feature_ls)
        else:
            self.group_n = group_n
        model = self.search_models[base_model]
        """
        逐步加入特征，确定每一次加入的最优特征
        """
        n = 1
        feature_opt_ls = []
        metric_opt_ls = []
        X_train, u, std = self._standardize_features(df[feature_ls])
        while n <= self.group_n or len(feature_ls) > 0:
            select_dict = {}
            for f in feature_ls:
                feature_slect = feature_opt_ls + [f]
                # 获取最优参数对应评估指标
                seacher = self._set_seacher(model(), self.search_params[base_model],
                                            scoring_fun=score_fun,
                                            cv=self._k_split(df, label_name))
                seacher.fit(X_train[feature_slect], df[label_name])
                select_dict[f] = seacher.best_score_  # 记录增加该特征的评价指标

            f_opt = max(select_dict.items(), key=lambda x: x[1])
            feature_opt_ls.append(f_opt[0])
            metric_opt_ls.append(f_opt[1])
            feature_ls.remove(f_opt[0])
            n += 1
            self.log(f'增加第{n-1}个特征: {f_opt[0]}, 评价指标数值{f_opt[1]}')

        # 在group_n内筛选最优特征组合
        feature_opt_ls = feature_opt_ls[:np.argmax(metric_opt_ls) + 1]
        self.log(f'最终评价结果: {max(metric_opt_ls)}, 特征组合{feature_opt_ls}')
        return feature_opt_ls

    def __call__(self):
        """
        完整进行特征筛选通
        :return:
        """
        pass

if __name__ == '__main__':
    # 功能测试
    # af = AutoFeature(fit_type='classification', fit_metric='rec_pre')
    af = AutoFeature(fit_type='regression', fit_metric='r2')
    # af = AutoFeature()
    df = pd.read_csv(r'E:\02code\01_EasyPlot\sample.csv')
    feature_num = ['feature_3', 'feature_15', 'feature_26', 'feature_11',
       'feature_12', 'feature_194', 'feature_18', 'feature_210', 'feature_22',
       'feature_5', 'feature_270', 'feature_6', 'feature_267', 'feature_204',
       'feature_7', 'feature_281', 'feature_24', 'feature_23', 'feature_193',
       'feature_213', 'feature_191', 'feature_230', 'feature_250',
       'feature_297', 'feature_299', 'feature_90', 'feature_8', 'feature_188',
       'feature_343', 'feature_352', 'feature_25']
    feature_clf = ['feature_347', 'feature_298', 'feature_294']
    label_name = 'price'
    # df['price'] = pd.qcut(df['price'], q=2, labels=[x for x in range(2)])
    df_filter, col_num, col_clf = af.filtering(df, feature_num, feature_clf, label_name)
    # 嵌套过滤
    feature_top = af.nesting(df_filter, col_num + col_clf, label_name)
    # 包裹过滤
    feature_opt_ls = af.wrapping(df, list(feature_top.keys()), 'price', base_model='cart')