import numpy as np
import pandas as pd
try:
    from utils import run_pca, _model_pred, save_json
    from config import ModelConfig
except:
    from .utils import run_pca, _model_pred, save_json
    from .config import ModelConfig


import logging
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from functools import partial
import joblib
import os
import json
import warnings
from typing import Union
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
log = logging.info

"""
# TODO 增加PCA降维技术 done
# TODO 保存json结果 done
# TODO 增加PCA模型保存、增加归一化模型保存，用于预测阶段的数据预处理 done
"""

class AutoModel(ModelConfig):
    """
    auto modeling
    method: trainning multi models, and then ensamble
    """
    def __init__(self, fit_type='regression', fit_metric=None, k_cv=4, metric_filter=0.8,
                 params_searcher='grid', pca_ratio=0.88, log_path='auto_model.log'):
        super(AutoModel, self).__init__()
        log('--' * 10 + f'auto modeling, fit type{self.fit_type}' + '--' * 10)
        self.fit_type = fit_type                
        self.fit_metric = fit_metric           
        self.k_cv = k_cv                        # kfold
        self.metric_filter = metric_filter      # filer threshold for weak model
        self.params_searcher = params_searcher 
        self.pca_ratio = pca_ratio              # ratio of tainning data info to keep for pca, if None, then no pca for data
        self.pca = None
        self.scaler = None
        self.stack_model = {}

        if self.fit_type not in ['regression', 'classification']:
            raise ValueError(f'fit_type erro, need to be regression or classification, get {self.fit_type}')
        if self.params_searcher not in ['grid', 'random', 'bayes']:
            raise ValueError(f"params_searcher erro, only suport 'grid, random, bayes', get {self.params_searcher}, ")

    def _k_split(self, X_train, y_train):
        # kfold split
        if self.fit_type == 'regression':
            boxes = 50
            box_ = pd.qcut(y_train, q=boxes, duplicates='drop', labels=False)
        else:
            box_ = y_train
        skfold = StratifiedKFold(n_splits=self.k_cv, shuffle=True, random_state=2023)
        skfold_split = skfold.split(X_train, box_)
        return skfold_split

    def _split_train_test(self, df, feture_ls, label_name):
        if self.pca_ratio is not None:
            X_data, self.pca = run_pca(df[feture_ls], self.pca_ratio)
            log(f'run PCA, ori feature count: {len(feture_ls)}, left feature count: {X_data.shape[1]}, info ratio left: {self.pca_ratio}')
        else:
            X_data = df[feture_ls]
        X_train, X_test, y_train, y_test = train_test_split(X_data, df[label_name], test_size=0.2, random_state=2023)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train, self.y_test = y_train, y_test

    def fit(self, df: pd.DataFrame, feture_ls:list, label_name:str, models:list=None):
        # models: point what model to use, default None to use all base model
        log('--' * 5 + f'multi model trainning and filter, threshold {self.metric_filter}' + '--' * 5)
        self._set_models()
        self._set_params()
        self._split_train_test(df, feture_ls, label_name)
        score_fun = metrics.make_scorer(self._metric_fun)
        if models is not None:
            self.search_models = dict([item for item in self.search_models.items() if item[0] in models])
        
        result_dict = {'label': label_name, 
                       'metric_filter_threshold': self.metric_filter, 
                       'features': feture_ls
                       }
        # kfold to find best hyps
        self.models_fit, self.train_matric, self.test_matric = {}, {}, {}
        mat_train, mat_test = {}, {}
        for i, (k, model) in enumerate(self.search_models.items()):
            print('kkkk', k)
            if k == 'xgb':
                continue

            if k == 'adb':  # use cart for adb
                self.search_params[k]['estimator'] = [adb_base_model]
            seacher = self._set_seacher(model(), self.search_params[k],
                                        scoring_fun=score_fun,
                                        cv=self._k_split(self.X_train, self.y_train))
            seacher.fit(self.X_train, self.y_train)
            best_params = seacher.best_params_
            model = model(**best_params)
            model.fit(self.X_train, self.y_train)
            y_test_pred = model.predict(self.X_test)
            metric_test = self._metric_fun(self.y_test, y_test_pred)
            
            # filter weak model
            is_flitered = False
            if metric_test >= self.metric_filter:
                y_train_pred = model.predict(self.X_train)
                metric_train = self._metric_fun(self.y_train, y_train_pred)
                self.train_matric[k] = metric_train
                self.test_matric[k] = metric_test
                self.models_fit[k] = model
                mat_train[k] = y_train_pred
                mat_test[k] = y_test_pred
                log(f'get model {k}, test metric={metric_test}, best hyps: {best_params}')
            else:
                log(f'filter model {k}, test metric={metric_test}, best hyps: {best_params}')
                is_flitered = True
            
            if k == 'cart':
                adb_base_model = model
            if k == 'adb':
                best_params['estimator'] = 'cart'
            result_dict[f'model_{k}'] = {'metric_test': metric_test, 
                                         'is_flitered': is_flitered,
                                         'best_hyps': best_params}

        if len(self.models_fit) == 0:
            raise ValueError(f"filter all model, need to lower metric_filter threshold {self.metric_filter} or reback to make nwe feature group")

        log('--' * 5 + f'all models trainning done, start to find the best ensamble path' + '--' * 5)
        self.best_model, result_ensemble = self.best_ensemble(mat_train, mat_test)

        print('aaaaaaaaa')
        print(result_ensemble)

        result_dict['best_ensemble'] = result_ensemble
        return result_dict

    def best_ensemble(self, mat_train: Union[np.ndarray, dict], mat_test: Union[np.ndarray, dict]):
        # find best ensemble path
        log(f'model metric: {self.train_matric}')
        self._set_ensemble_method()
        if isinstance(mat_train, dict):
            mat_train = np.array(list(mat_train.values())).T  # shape = (n_samples, n_models)
            mat_test = np.array(list(mat_test.values())).T
        result_dict = {}
        ensemble_metric = {}

        for k, ensemble in self.ensembler.items():
            if k not in ['stack', 'stack_cart']:
                m = self._metric_fun(self.y_test, ensemble(mat_test))
            else:
                m = self._metric_fun(self.y_test, ensemble(mat_train, self.y_train, mat_test))
            ensemble_metric[k] = m
            log(f'ensamble method {k}, test metric={m}')
            result_dict[k] = {'metric_test': m}

        # compare ensemble and single model, find the best
        k_e, m_test_e = max(ensemble_metric.items(), key=lambda x: x[1])
        k_test, m_test = max(self.test_matric.items(), key=lambda x: x[1])
        for sk in ['stack', 'stack_cart']:
            self.ensembler[sk] = partial(_model_pred, model=self.stack_model[sk])

        log(f'best ensamble model metric={m_test_e}, best single model metric={m_test}')
        result_dict['best_ensemble'] = {'method': k_e, 'metric_test': m_test_e}
        
        if m_test_e > m_test:
            log(f'best ensamble method: {k_e}, metric={m_test_e}')
            return (1, k_e, self.ensembler[k_e]), result_dict
        else:
            log(f'all ensamble methods no work, bset model: {k_test}, metric={m_test}')
            return (0, k_test, partial(_model_pred, model=self.models_fit[k_test])), result_dict

    def predict(self, X_feature: Union[pd.DataFrame, np.array]):
        # peprocess: standerdize and pca
        if self.pca is not None:
            X_feature = self.pca.transform(X_feature)
            cols_need = np.where(np.cumsum(self.pca.explained_variance_ratio_) <= self.pca_ratio)
            cols_need = list(cols_need[0])
            X_feature = X_feature[:, cols_need + [cols_need[-1] + 1]]
        
        if self.scaler is not None:
            X_feature = self.scaler.transform(X_feature)

        # inference
        ensemble, k_, func = self.best_model
        if ensemble:
            mat_pred = np.zeros((X_feature.shape[0], len(self.models_fit)))
            for i, (k, model) in enumerate(self.models_fit.items()):
                mat_pred[:, i] = model.predict(X_feature)
            return func(mat_pred)
        else:
            return func(X_feature)

    def _set_ensemble_method(self):
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
        # voting in metric weight
        if soft:
            if weight is None:
                result = np.mean(mat_rslt, axis=1)
            else:
                if isinstance(weight, dict):
                    w = np.array(list(weight.values()))
                else:
                    w = weight
                w = w * w
                w = w / sum(w) 
                result = np.sum(mat_rslt * w, axis=1)
            if self.fit_type == 'classification':
                result[np.where(result < 0.5)] = 0
                result[np.where(result >= 0.5)] = 1
        else:
            thre = np.floor(mat_rslt.shape[1] / 2)
            result = np.sum(mat_rslt, axis=1) - thre
            result[np.where(result <= 0)] = 0
            result[np.where(result > 0)] = 1
        return result

    def _stacking(self, mat_train: np.array, y_train, mat_test: np.array, base_mode=None):
        # train new model with all model preds, base model should simple
        # regress use cart lasso, classify use logist cart
        if base_mode is None:
            if self.fit_type == 'regression':
                base_mode = 'lasso'
            else:
                base_mode = 'logit'
        self._set_models()
        self._set_params()
        if base_mode == 'cart':
            self.search_params[base_mode] = {'max_depth': [1, 2, 3]}
        score_fun = metrics.make_scorer(self._metric_fun)
        model = self.search_models[base_mode]
        # best hyps
        seacher = self._set_seacher(model(), self.search_params[base_mode],
                                    scoring_fun=score_fun,
                                    cv=self._k_split(mat_train, y_train))
        seacher.fit(mat_train, y_train)
        model = model(**seacher.best_params_)
        model.fit(mat_train, y_train)
        if base_mode == 'cart':
            self.stack_model['stack_cart'] = model  # for pred
        else:
            self.stack_model['stack'] = model
        return model.predict(mat_test)

    def save_model(self, path=None):
        """
        path: path_like str, e.g path/file/
        """
        if path is None:
            path = './checkpoint'
        os.makedirs(path, exist_ok=True)
    
        opt, k_name, func = self.best_model
        if opt:
            for k, model in self.models_fit.items():
                joblib.dump(model, os.path.join(path, f'{k_name}_{k}.pkl'))  # save all model pkl
            # stack save one more
            if k_name in ['stack', 'stack_cart']:
                joblib.dump(self.stack_model[k_name],
                            os.path.join(path, f'{k_name}_{k_name}.pkl'))
            if k_name == 'softw':
                json.dump(self.test_matric, fp=os.path.join(path, 'weight.json'))
        else:
            model = func.keywords['model']
            joblib.dump(model, os.path.join(path, f'{k_name}.pkl'))
        
        # pca
        if self.pca is not None:
            joblib.dump(self.pca, os.path.join(path, f'pca.pkl'))
        
        # scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(path, f'scaler.pkl'))


    def load_model(self, path=None):
        if path is None:
            path = './checkpoint'
        files = [x for x in os.listdir(path) if x.split('.')[-1] == 'pkl' and not x.startswith(('pca', 'scaler'))]
        if len(files) == 1:
            k = files[0].split('.')[0]
            model = joblib.load(os.path.join(path, files[0]))
            self.best_model = 0, k, partial(_model_pred, model=model)
        else:
            k_ensemble = files[0].split('_')[0]
            self.models_fit = {}
            for f in files:
                if f.endswith('json'):
                    self.test_matric = json.load(os.path.join(path, f))
                else:
                    self.models_fit[f.split('_')[1].split('.')[0]] = joblib.load(os.path.join(path, f))
            self._set_ensemble_method()

            if k_ensemble in ['stack', 'stack_cart']:
                model_stack = self.models_fit.pop(k_ensemble)
                self.ensembler[k_ensemble] = partial(_model_pred, model=model_stack)
            self.best_model = 1, k_ensemble, self.ensembler[k_ensemble]
        
        pca_path = os.path.join(path, 'pca.pkl')
        scaler_path = os.path.join(path, 'scaler.pkl')
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)


if __name__ == '__main__':
    # test
    automodel = AutoModel(fit_type='regression', fit_metric='r2')
    # automodel = AutoModel(fit_type='classification', fit_metric='rec_pre')
    # feature = ['feature_18', 'feature_3', 'feature_25', 'feature_294',
    #                'feature_12', 'feature_23', 'feature_7', 'feature_8']  # clf
    
    # df['price'] = pd.qcut(df['price'], q=2, labels=[x for x in range(2)])
    
    df = pd.read_csv('/home/chaofeng/autoML_for_csv/data/sample.csv')
    feature = ['feature_3', 'feature_15', 'feature_8', 'feature_11', 'feature_25', 'feature_12',
               'feature_5', 'feature_210', 'feature_6', 'feature_22'] # reg
    label_name = 'price'
    result_dict = automodel.fit(df, feature, label_name)

    y_pred = automodel.predict(df[feature])
    automodel.save_model()

    save_json(result_dict, '/home/chaofeng/autoML_for_csv/doc/automodel_fit.json')


    # 载入预测: 已测试单个模型无集成策略 stack集成 TODO 待测试加权集成
    automodel.load_model(path=None)
    yp = automodel.predict(df[feature])
    print(yp)
    r2 = metrics.r2_score(y_true=df['price'].to_list(), y_pred=yp)
    print(r2)

