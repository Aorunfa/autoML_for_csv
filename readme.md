## 持续更新中
# 介绍

&emsp;&emsp;该项目主要面向结构化数据进行有监督学习的场景，主要目的是快速进行特征挖掘，特征筛选，特征建模，并以类似torch的形式保存或载入相关建模结果。为此，项目主要开发了三个模块，帮助数据分析人员进行快速验证:
1. auto_feature主要针对特征组冗余过滤，快速完成关键特征组的确定
2. auto_model主要针对建模及集成寻优，快速完成模型选型与集成方式选择 
3. auto_plot主要针对特征探索过程的可视化，快速完成相关特征的可视化分析，特征探索

一些细节介绍见`autoML_for_csv/doc/introduce.md`

# 环境准备
```bash
git clone https://github.com/Aorunfa/autoML_for_csv.git
conda create -n automl python=3.10
cd ./autoML_for_csv
conda activate automl
pip install -r requirements.txt
```

# 快速使用
## 自动化特征筛选
```python
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

# 保存中间结果
save_json(result_corr, './doc/filter_corr.json')
save_json(result_nest, './doc/filter_nest.json')
save_json(result_wrap, './doc/filter_wrap.json')
```

## 自动化建模
模型训练
```python
# 定义问题时回归还是分类，定义模型评价指标
automodel = AutoModel(fit_type='regression', fit_metric='r2')

# 指定标签名称，指定需要使用的特征列:
df = pd.read_csv('/home/chaofeng/autoML_for_csv/data/sample.csv')
feature = ['feature_3', 'feature_15', 'feature_8', 'feature_11', 'feature_25', 
            'feature_12', 'feature_5', 'feature_210', 'feature_6', 'feature_22'] 
label_name = 'price'

# 训练
result_dict = automodel.fit(df, feature, label_name)

# 预测
y_pred = automodel.predict(df[feature])

# 模型保存
automodel.save_model(path='/checkpoint')

# 中间结果存储
save_json(result_dict, '/doc/automodel_fit.json')
```

模型加载预测
```python
automodel = AutoModel(fit_type='regression', fit_metric='r2')
automodel.load_model(path=None)
preds = automodel.predict(df[feature])
```

## 快速绘图
使用说明见`doc/plot_indroduce.md`，代码操作见`autoML_for_csv/doc/auto_plot_demo.ipynb`

