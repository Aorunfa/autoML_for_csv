import numpy as np
import pandas as pd
import logging
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, chi2
from functools import partial
try:
    from utils import (kfold_split, standardize_features, save_json)
    from config import FeatureConfig
except:
    from .utils import (kfold_split, standardize_features, save_json)
    from .config import FeatureConfig
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
log = logging.info

class AutoFeature(FeatureConfig):
    def __init__(self, corrval_withlabel=0.35, corrval_withothers=0.85, p_val=0.05, fit_type='regression', fit_metric=None, k_cv=4, group_n=-1, params_searcher='grid'):
        super(AutoFeature, self).__init__()
        self.fit_type = fit_type                     
        # corr fliter threshold
        self.corrval_withlabel = corrval_withlabel   
        self.corrval_withothers = corrval_withothers 
        self.p_val = p_val
        # nesting filter
        self.fit_metric = fit_metric                    # model metric func
        self.k_cv = k_cv                                # k fold
        # wrapping filter
        self.group_n = group_n                          # max feature counts
        # hyp searcher
        self.params_searcher = params_searcher          # random or bayes

        self._k_split = partial(kfold_split, k_cv=self.k_cv, fit_type=self.fit_type)  # passing params

        if self.fit_type not in ['regression', 'classification']:
            raise ValueError(f'fit_type erro, need to be regression or classification, get {self.fit_type}')
        if self.params_searcher not in ['grid', 'random', 'bayes']:
            raise ValueError(f"params_searcher erro, only suport 'grid, random, bayes', get {self.params_searcher}, ")

    def _indpendent_test(self, dist_feature, dist_target: pd.Series, box_trans=False, method='ch2'):
        boxes = 20
        if box_trans:  # discrete
            dist_feature = pd.qcut(dist_feature, q=boxes, duplicates='drop', labels=False)
        
        if self.fit_type == 'regression':
            dist_target = pd.qcut(dist_target, q=boxes, duplicates='drop', labels=False)
        
        if not isinstance(dist_feature, np.ndarray):
            dist_feature = np.asarray(dist_feature).reshape(-1, 1)
            dist_target = np.asarray(dist_target).reshape(-1, 1)
        
        if method == 'ch2':
            # _, p = chi2(dist_feature, dist_target)
            p = chi2(dist_feature, dist_target)[1][0]                      # p<0.05 refuse H0, no independet
            if p <= self.p_val:
                return p, False
            return p, True
        
        else:
            # Kullback-Leibler divergence
            if box_trans:
                mi = mutual_info_regression(dist_feature, dist_target)[0]  # mi --> 0, more dependet
            else:
                mi = mutual_info_classif(dist_feature, dist_target)[0]
            if mi >= 0.2:
                return mi, False
            return mi, True

    def filtering_reg(self, df:pd.DataFrame, feature_num:list, feature_clf:list, label_name:str):
        # filter num feature the no corr to label or corr to each other
        corr_withlabel = df[feature_num].corrwith(df[label_name], method='pearson').abs().fillna(0)
        corr_withlabel = corr_withlabel[corr_withlabel > self.corrval_withlabel]
        if corr_withlabel.shape[0] == 0:
            log('no num feature corr to label, reback to make new feature')
            col_filter = []
        else:
            log(f'filter num_feature no corr to label: {[x for x in feature_num if x not in corr_withlabel.index]}, threshold {self.corrval_withlabel}')
            if corr_withlabel.shape[0] >= 2:
                corr_withlabel = corr_withlabel.sort_values(ascending=False)
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
                log(f'filter num_feature corr to each other: {col_drop}, threshold{self.corrval_withothers}')
            else:
                col_filter = list(corr_withlabel.index)

        # filter cat feature the no corr to label
        col_drop = []
        for f in feature_clf:
            p, jude = self._indpendent_test(df[f], df[label_name])
            if jude:
                col_drop.append(f)
        col_filter2 = [x for x in feature_clf if x not in col_drop]
        log(f'filter cat_feature no corr to label: {col_drop}')
        log(f'num left: {col_filter}')
        log(f'cat left: {col_filter2}')

        result = {'num_feature': {'left': col_filter, 'filter': [i for i in feature_num if i not in col_filter]},
                  'cat_feature': {'left': col_filter2, 'filter': [i for i in feature_clf if i not in col_filter2]},
                  'threshold': {
                            'corrval_withlabel': self.corrval_withlabel,
                            'corrval_withothers': self.corrval_withothers,
                            'p_val': 0.05}
                            }
        return df[col_filter + col_filter2 + [label_name]], result


    def filtering_clf(self, df:pd.DataFrame, feature_num:list, feature_clf:list, label_name:str):
        # filter num feature no corr to label
        col_drop = []
        p_ls = []
        for f in feature_num:
            p, jude = self._indpendent_test(df[f], df[label_name], box_trans=True)
            if jude:
                col_drop.append(f)
            else:
                p_ls.append(p)
        col_filter = [x for x in feature_num if x not in col_drop]
        log(f'filter num feature no corr to label: {col_drop}')

        # filter num feature corr to each other
        if len(col_filter) >= 2:
            # order by p
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
            log(f'filter num feature corr to each other: {col_drop}')

        # filter cat feature no corr to label
        col_drop = []
        for f in feature_clf:
            p, jude = self._indpendent_test(df[f], df[label_name])
            if jude:
                col_drop.append(f)
        col_filter2 = [x for x in feature_clf if x not in col_drop]
        log(f'filter cat feature no corr to label: {col_drop}')
        log(f'num left: {col_filter}')
        log(f'cat left: {col_filter2}')

        result = {'num_feature': {'left': col_filter, 'filter': [i for i in feature_num if i not in col_filter]},
                  'cat_feature': {'left': col_filter2, 'filter': [i for i in feature_clf if i not in col_filter2]},
                  'threshold': {
                            'corrval_withothers': self.corrval_withothers,
                            'p_val': 0.05}
                            }
        return  df[col_filter + col_filter2 + [label_name]], result
        
    def filter_corr(self, df:pd.DataFrame, feature_num:list, feature_clf:list, label_name:str):
        log('--'*5 + f'filter corr' + '--'*5)
        if self.fit_type == 'regression':
            return self.filtering_reg(df, feature_num, feature_clf, label_name)
        else:
            return self.filtering_clf(df, feature_num, feature_clf, label_name)

    def filter_nesting(self, df, feature_ls, label_name, top_k=10):
        log('--' * 5 + f'filter nesting' + '--' * 5)
        self._set_models()
        self._set_params()
        score_fun = metrics.make_scorer(self._metric_fun)
        
        # init
        result_dict = {'filter_type': 'nesting', 'label': label_name}
        feature_importance = np.zeros((len(self.search_models), len(feature_ls)))
        weights = np.zeros(len(self.search_models))
        X_search, _, _ = standardize_features(df[feature_ls])
        
        for i, (k, model) in enumerate(self.search_models.items()):
            mat_importance = np.zeros((self.k_cv, len(feature_ls)))
            ls_metrics = []
            # print('kkkkk', k)
            if k == 'xgb':
                continue

            # find best hyps
            seacher = self._set_seacher(model(), self.search_params[k],
                                        scoring_fun=score_fun,
                                        cv=self._k_split(df, label_name))
            seacher.fit(X_search, df[label_name])
            best_params = seacher.best_params_
            model = model(**best_params)
            
            # get best scores 
            for j, (train_idx, valid_idx) in enumerate(self._k_split(df, label_name)):
                X_train, y_train = df.loc[train_idx, feature_ls], df.loc[train_idx, label_name]
                X_valid, y_valid = df.loc[valid_idx, feature_ls], df.loc[valid_idx, label_name]
                # standerdize 
                X_train, u, std = standardize_features(X_train)
                X_valid = standardize_features(X_valid, 'valid', u, std)
                model.fit(X_train, y_train)
                
                try:
                    fi = np.abs(model.feature_importances_)
                except:
                    fi = np.abs(model.coef_)
                
                fi = fi / np.sum(fi)  # normalize
                val_metrics = self._metric_fun(y_valid, model.predict(X_valid))
                
                mat_importance[j] = fi
                ls_metrics.append(val_metrics)
            
            feature_importance[i] = np.mean(mat_importance, axis=0)
            weights[i] = np.mean(ls_metrics)
            log(f'model {k} metric: {weights[i]}, feature importance：{feature_importance[i]}')
            result_dict[k] = {'metric': weights[i], 'feature_importance': dict(zip(feature_ls, feature_importance[i]))}

        # bagging
        weights_ = np.reshape(weights / sum(weights), (-1, 1))
        feature_importance = tuple(zip(feature_ls, np.sum(feature_importance * weights_, axis=0)))
        feature_importance = dict(sorted(feature_importance, key=lambda x: x[1])[::-1])
        model_weights = sorted(tuple(zip(self.search_models.keys(), weights)), key=lambda x:x[1])[::-1]
        log(f'final feature_importance: {feature_importance}')
        log(f'model metric sorted: {model_weights}')
        
        result_dict['integrated'] = {'model_weight': dict(model_weights), 'feature_importance': dict(feature_importance)}
        feature_importance = dict(list(feature_importance.items())[:top_k])
        return feature_importance, result_dict

    def filter_wrapping(self, df:pd.DataFrame, feature_ls: list, label_name: str, base_model='cart', group_n=-1):
        # 包裹式特征筛选
        log('--' * 5 + f'filter wrapping, base model: {base_model}' + '--' * 5)
        self._set_models()
        self._set_params()
        score_fun = metrics.make_scorer(self._metric_fun)
        if group_n == -1:
            self.group_n = len(feature_ls)
        else:
            self.group_n = group_n
        model = self.search_models[base_model]

        # greedy add best feature in group
        result_dict = {'filter_type': 'wrapping', 
                       'label': label_name, 
                       'base_model': base_model, 
                       'group_n': group_n}
        result_add = {}
        n = 1
        feature_opt_ls = []
        metric_opt_ls = []
        X_train, _, _ = standardize_features(df[feature_ls])
        while n <= self.group_n or len(feature_ls) > 0:
            select_dict = {}
            for f in feature_ls:
                feature_slect = feature_opt_ls + [f]
                # get best hyps and its best score
                seacher = self._set_seacher(model(), self.search_params[base_model],
                                            scoring_fun=score_fun,
                                            cv=self._k_split(df, label_name))
                seacher.fit(X_train[feature_slect], df[label_name])
                select_dict[f] = seacher.best_score_ 

            f_opt = max(select_dict.items(), key=lambda x: x[1])
            feature_opt_ls.append(f_opt[0])
            metric_opt_ls.append(f_opt[1])
            feature_ls.remove(f_opt[0])
            n += 1
            log(f'add {n-1} fea: {f_opt[0]}, metric: {f_opt[1]}')
            result_add['add %d' % (n-1)] = {'feature_add': f_opt[0], 'metric_update': f_opt[1]}

        # find best combination in group_n
        feature_opt_ls = feature_opt_ls[:np.argmax(metric_opt_ls) + 1]
        log(f'final metric: {max(metric_opt_ls)}, feature group: {feature_opt_ls}')
        result_dict['final'] = {'metric': max(metric_opt_ls), 'feature_group': feature_opt_ls}
        result_dict.update(result_add)
        return feature_opt_ls, result_dict

if __name__ == '__main__':
    # test
    # af = AutoFeature(fit_type='classification', fit_metric='rec_pre')
    af = AutoFeature(fit_type='regression', fit_metric='r2')
    df = pd.read_csv('/home/chaofeng/autoML_for_csv/data/sample.csv')
    # df['price'] = pd.qcut(df['price'], q=2, labels=[x for x in range(2)])

    feature_num = ['feature_3', 'feature_15', 'feature_26', 'feature_11',
                    'feature_12', 'feature_194', 'feature_18', 'feature_210', 'feature_22',
                    'feature_5', 'feature_270', 'feature_6', 'feature_267', 'feature_204',
                    'feature_7', 'feature_281', 'feature_24', 'feature_23', 'feature_193',
                    'feature_213', 'feature_191', 'feature_230', 'feature_250',
                    'feature_297', 'feature_299', 'feature_90', 'feature_8', 'feature_188',
                    'feature_343', 'feature_352', 'feature_25']
    feature_clf = ['feature_347', 'feature_298', 'feature_294']
    label_name = 'price'

    df_filter, result_corr = af.filter_corr(df, feature_num, feature_clf, label_name)
    col_num = result_corr['num_feature']['left']
    col_clf = result_corr['cat_feature']['left']

    feature_top, result_nest = af.filter_nesting(df_filter, col_num + col_clf, label_name)
    feature_opt_ls, result_wrap = af.filter_wrapping(df, list(feature_top.keys()), 'price', base_model='cart')


    save_json(result_corr, '/home/chaofeng/autoML_for_csv/doc/filter_corr.json')
    save_json(result_nest, '/home/chaofeng/autoML_for_csv/doc/filter_nest.json')
    save_json(result_wrap, '/home/chaofeng/autoML_for_csv/doc/filter_wrap.json')
