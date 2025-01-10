from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from typing import Callable
from sklearn import metrics
import json
import pandas as pd
import numpy as np

class BaseFeature(object):
    def __init__(self):
        self.fit_type = 'regression'
        self.parms_search = 'grid'
        self.fit_metric = None

    def _set_models(self):
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
        random_seed = 2023
        if self.fit_type == 'regression':
            # hyper params
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
                          'verbosity': [-1],                # no print
                          'importance_type': ['gain']}
            cab_params = {'max_depth': [x for x in range(2, 11, 2)],
                          'iterations': [25, 50, 75, 100],
                          'learning_rate': [5e-1, 1e-1, 5e-2],
                          'random_state': [random_seed],
                          'verbose': [False],
                          # 'silent': [True]
                          # 'logging_level':['Silent']
                          }
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
                          'verbose': [False],
                          # 'silent': [True]
                          'logging_level':['Silent']
                          }
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
        self._set_metrics()
        if self.fit_type == 'regression':
            if self.fit_metric is None:
                self.fit_metric = 'r2'  # regess default
        else:
            if self.fit_metric is None:
                self.fit_metric = 'auc'  # classifier default
        return self.search_metrics[self.fit_metric](y_true, y_pred)

    def _set_seacher(self, model, param_dist: dict, scoring_fun: Callable, cv=None):
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
                                          n_iter=50, 
                                          random_state=42)
        else:
            searcher = BayesSearchCV(estimator=model,
                                     search_spaces=param_dist, 
                                     n_jobs=-1, 
                                     cv=cv,
                                     scoring=scoring_fun)
        return searcher


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