from model.auto_model import AutoModel
from model.auto_feature import AutoFeature
from model.utils import save_json
import pandas as pd
from sklearn import metrics

############################################ demo: run auto feature ############################################
# 定义问题时回归还是分类，定义模型评价指标
af = AutoFeature(fit_type='regression', fit_metric='r2')

# 列定义: 标签，连续特征，分类特征
df = pd.read_csv('./data/sample.csv')
feature_num = ['feature_3', 'feature_15', 'feature_26', 'feature_11',
                'feature_12', 'feature_194', 'feature_18', 'feature_210', 'feature_22',
                'feature_5', 'feature_270', 'feature_6', 'feature_267', 'feature_204',
                'feature_7', 'feature_281', 'feature_24', 'feature_23', 'feature_193',
                'feature_213', 'feature_191', 'feature_230', 'feature_250',
                'feature_297', 'feature_299', 'feature_90', 'feature_8', 'feature_188',
                'feature_343', 'feature_352', 'feature_25']
feature_clf = ['feature_347', 'feature_298', 'feature_294']
label_name = 'price'

# 相关性过滤
df_filter, result_corr = af.filter_corr(df, feature_num, feature_clf, label_name)
col_num = result_corr['num_feature']['left']
col_clf = result_corr['cat_feature']['left']

# nesting 过滤: 多模型对特征进行评分，以加权评分最为特征重要性
feature_top, result_nest = af.filter_nesting(df_filter, col_num + col_clf, label_name)

# wrapping 过滤: 依次贪婪地往特征池中加入一个新的特征
# 贪婪是指当前加入的特征是所有特征中最优的
# 最后输出一个特征组合
feature_opt_ls, result_wrap = af.filter_wrapping(df, list(feature_top.keys()), 'price', base_model='cart')

# 结果存储
save_json(result_corr, './doc/filter_corr.json')
save_json(result_nest, './doc/filter_nest.json')
save_json(result_wrap, './doc/filter_wrap.json')


############################################ demo: run auto model ############################################
# # 定义问题时回归还是分类，定义模型评价指标
# automodel = AutoModel(fit_type='regression', fit_metric='r2')

# # # 指定标签名称，指定需要使用的特征列:
# df = pd.read_csv('./data/sample.csv')
# feature = ['feature_3', 'feature_15', 'feature_8', 'feature_11', 'feature_25', 
#             'feature_12', 'feature_5', 'feature_210', 'feature_6', 'feature_22'] 
# label_name = 'price'

# # 训练
# result_dict = automodel.fit(df, feature, label_name)

# # 预测
# y_pred = automodel.predict(df[feature])

# # 模型保存
# automodel.save_model(path='./checkpoint')

# # 结果存储
# save_json(result_dict, './doc/automodel_fit.json')

# # 模型加载预测
# automodel = AutoModel(fit_type='regression', fit_metric='r2')
# automodel.load_model(path='./checkpoint')
# preds = automodel.predict(df[feature])
# r2 = metrics.r2_score(y_true=df['price'].to_list(), y_pred=preds)
# print(r2)
