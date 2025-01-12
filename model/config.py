from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.svm import SVC, SVR
from skopt import BayesSearchCV
from typing import Callable
from sklearn import metrics
from sklearn.ensemble import (
                        RandomForestClassifier, 
                        AdaBoostClassifier, 
                        AdaBoostRegressor, 
                        RandomForestRegressor)

class FeatureConfig(object):
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



class ModelConfig(object):
    def __init__(self):
        self.fit_type = 'regression'
        self.parms_search = 'grid'
        self.fit_metric = None

    def _set_models(self):
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
                self.fit_metric = 'r2' 
        else:
            if self.fit_metric is None:
                self.fit_metric = 'auc'
        return self.search_metrics[self.fit_metric](y_true, y_pred)

    def _set_seacher(self, model, param_dist: dict, scoring_fun, cv=None):
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
