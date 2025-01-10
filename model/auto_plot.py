import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus']=False
from scipy import stats
import matplotlib.pyplot as plt
colors = plt.cm.viridis(np.linspace(0, 1, 10))
# sns.set_theme()


class AutoPlot(object):
    """
    功能:集成的绘图类，包含多场景的绘图方法；
    单变量与单变量组合绘图:
        ·单数值变量绘图:核密度图，箱线图，小提琴图
        ·单数值变量与序列变量绘图: 核密度图，箱线图，小提琴图，序列图（可选散点、小提琴、箱线）
        ·单数值变量-序列变量-分类变量绘图: 序列图（可选散点、小提琴、箱线）
        ·单个分类变量:unique数量条形图
        ·单分类变量-分类变量:单分类变量按一个分类变量切片的unique数量条形图
    多变量绘图:
        ·多数值变量-多数值变量绘图:热力图，散点图
    其他绘图:
        ·qq图比较两个给定分布的差异
    """
    def __init__(self, save_path, dpi=1200, pic_format='png', font='Simhei',
                 font_scale=2, style='darkgrid', color_map=None):
        """

        :param save_path: 图片存储路径
        :param dpi: 存储像素
        :param pic_format:存储格式
        :param font: 字体
        :param font_scale: 字体大小
        :param style: 背景风格
        :param color_map: 颜色字典
        """
        if color_map is None:
            self.color_map = ['red', 'blue', 'cyan', 'olive', 'green', 'gray',
                         'purple', 'brown', 'black', 'pink', 'orange']
            # self.color_map = plt.cm.viridis(np.linspace(0, 1, 10))
        else:
            self.color_map = color_map

        self.path = save_path
        self.format = pic_format
        self.dpi = dpi
        self.font = font
        self.font_scale = font_scale
        self.style = style
        sns.set_theme(font_scale=self.font_scale, font=self.font, style=self.style)

    def _save_fig(self, fig, title, type):
        """
        存储图片
        :param fig: None or 画布
        :return:
        """
        if fig is not None:
            fig.savefig(self.path.format(title, type, self.format),
                        format=self.format, dpi=self.dpi)
        else:
            pass

    def num_dist_plot(self, df, num_name, ax_kde=None, ax_box=None,
                         ax_nbox=None, ax_vln=None, clr='b', bin_val=50,
                         label_name=None, fig=None):
        """
        绘制单个连续型变量分布，可选核密度、小提琴、箱线图
        :param df:特征数据 dataframe
        :param num_name: 连续型特征名称 str
        :param ax_kde、ax_box、ax_nbox、ax_vln: 指定绘制子图axe, eg: fig,axe=plt.subplots(**arg)
        :param clr: 绘图颜色 str
        :param bin_val: 核密度分箱数 int
        :param label_name: 核密度图例 str
        param fig: 画布，默认None，表示不保存画布图片
        """
        series = df[num_name]
        if not label_name:
            label_name = num_name

        if ax_kde:
            sns.distplot(series, bins=bin_val, ax=ax_kde, color=clr, label=label_name)
            ax_kde.set_title(f'{num_name} distribution')
            ax_kde.set_xlabel(f'{num_name} value')
            ax_kde.legend()

        if ax_box:
            sns.boxplot(series, linewidth=1, ax=ax_box, color=clr)
            ax_box.set_title(f'{num_name} boxplot')

        if ax_nbox:
            sns.boxenplot(series, linewidth=1, ax=ax_nbox, color=clr)
            ax_nbox.set_title(f'{num_name} nboxplot')

        if ax_vln:
            sns.violinplot(series, scale='width', linewidth=1, ax=ax_vln, color=clr)
            ax_vln.set_title(f'{num_name} violinplot')

        self._save_fig(fig, num_name, 'single_dist')

    def num_seq_plot(self, df, num_name, seq_name, order_sort=None,
                             ax_kde=None, ax_sca=None, ax_line=None, ax_box=None,
                             ax_vln=None, fig=None, scatter_size=1):
        """
        :param df:特征数据 dataframe
        :param num_name:连续型特征名称 str
        :param cat_name:分类特征名称 str
        :param seq_name:序列特征名称 str
        :param ax_sca，ax_line，ax_box，ax_vln: 指定绘制子图
        :param order_sort:指定序列变量的排序顺序 list
        :param lengend_font:调整图例参数 dict
        :param fig:画布,控制是否保存图片,默认为None
        """
        if ax_kde:
            """
            绘制按照分类切片连续变量密度分布，分类变量seq_name
            """
            n = df[seq_name].nunique()
            if n > len(self.color_map):
                self.color_map = plt.cm.viridis(np.linspace(0, 1, n))

            for i, cat in enumerate(df[seq_name].unique()):
                self.num_dist_plot(df[df[seq_name] == cat],
                                      num_name=num_name,
                                      ax_kde=ax_kde,
                                      clr=self.color_map[i],
                                      label_name=str(cat))

        if ax_box:
            """
            绘制连续变量 - 连续变量/分类变量箱线
            """
            sns.boxplot(data=df,
                        x=seq_name,
                        y=num_name,
                        order=order_sort,
                        linewidth=1,
                        ax=ax_box)
            ax_box.set_title(f'{num_name}-{seq_name} boxplot')

        if ax_vln:
            sns.violinplot(data=df,
                           x=seq_name,
                           y=num_name,
                           order=order_sort,
                           scale='width',
                           linewidth=1,
                           ax=ax_vln)
            ax_vln.set_title(f'{num_name}-{seq_name} violinplot')

        if ax_sca:
            """
            绘制两个连续变量的散点图
            """
            sns.scatterplot(x=df[seq_name],
                            y=df[num_name],
                            ax=ax_sca,
                            alpha=1,
                            size=scatter_size,
                            label=num_name)
            ax_sca.set_title(f'{num_name}-{seq_name} scatter')

        if ax_line:
            """
            绘制两个连续变量折线图，linewidth可调整线宽
            """
            sns.lineplot(data=df,
                         x=seq_name,
                         y=num_name,
                         ax=ax_line,
                         linewidth=2)
            ax_line.set_title(f'{num_name}-{seq_name} lineplot')

        self._save_fig(fig, num_name+'_'+seq_name, 'sequence_dist')

    def scatter_plot(self, df, y_name, x_name, axe, date_trans=False,
                     point_size=50, font_size=20, fig=None):
        """
        绘制散点图
        :param df:
        :param y_name:
        :param x_name:
        :param axe:
        :param date_trans: 是否转换日期格式
        :param point_size: 点大小 int
        :param font_size: 文字大小 int
        :param fig:
        :return:
        """
        data = df.copy()
        if date_trans:
            # 日期格式强转
            data[x_name] = pd.to_datetime(data[x_name])
            data.sort_values(by=x_name, ascending=True, inplace=True)

        axe.scatter(data[x_name],
                    data[y_name],
                    marker='o',
                    s=point_size)
        # 添加元素
        title = f'{y_name}_{x_name}'
        axe.set_title(title,
                      fontdict={'size': font_size},
                      loc='left')
        axe.set_xlabel(x_name,
                       fontdict={'size': font_size})
        axe.set_ylabel(y_name,
                       fontdict={'size': font_size})

        self._save_fig(fig, y_name + '_' + x_name, 'scatter')

    def num_num_plot(self, df, corr_type='pearson', ax_hotmap=None,
                         ax_pairplot=None, hue=None, kind='scatter', diag_kind='kde', fig=None):
        """
        绘制多个连续变量之间的相关性热力图、配对散点图
        :param df: 特征数据 dataframe
        :param corr_type: 需要计算的相关系数类型 str
        :param ax_hotmap: 热力图子图 axe
        :param ax_pairplot: 配对散点图子图 axe
        :param hue: 配对散点图分类变量 str
        :param kind、diag_kind: 配对散点图子图非对角线与对角线出图类型
        :param fig:
        :return:
        """
        if ax_hotmap:
            corr = round(df.corr(method=corr_type), 2)
            mine = np.min(corr.min())
            maxe = np.max(corr.max())
            sns.heatmap(corr,
                        annot=True,
                        annot_kws={'alpha': 0.8, 'size': 15},
                        vmin=mine,
                        vmax=maxe,
                        center=0,
                        linewidths=0.5,
                        cmap='Blues',
                        ax=ax_hotmap)  # annot_kws调整显示数值格式
            ax_hotmap.set_title(corr_type)

        if ax_pairplot:
            """
            配对散点图传入特征数量不能太多，否则出图很慢
            参数diag_kind用于指定对角线出图类型
            参数kind用于指定非对角线出图类型
            """
            if not hue:
                sns.pairplot(data=df,
                             kind=kind,
                             diag_kind=diag_kind,
                             plot_kws={'color': 'b', 'alpha': 0.5})
            else:
                # 分类统计
                sns.pairplot(data=df,
                             hue=hue,
                             kind=kind,
                             diag_kind=diag_kind,
                             plot_kws={'color': 'b', 'alpha': 0.5})

        self._save_fig(fig, 'num_num', '_plot')

    def num_num_2d_dist(self, df, num_name0, num_name1, axe, fig=None, n_levels=10):
        """
        绘制两个数值变量的二维核密度图
        :param df: 特征数据 dataframe
        :param num_name0: 数值变量 str
        :param num_name1: 数值变量 str
        :param axe:
        :param fig:
        :param n_levels:
        :return:
        """
        sns.kdeplot(x=df[num_name0],
                    y=df[num_name1],
                    ax=axe,
                    shade=True,
                    shade_lowest=False,
                    cbar=True,
                    n_levels=n_levels,
                    cmap='Oranges')

        self._save_fig(fig, num_name0 + '_' + num_name1, '2d_dist')

    def qq_plot(self, data0, data1, label0, label1, quantile_num=50,
                ax_cuml=None, ax_qq=None, ax_qq_ref=None,
                fig=None):
        """
        绘制两组数据的累计分布图，qq图
        注意: 默认qq图ax_qq_standar的基准数据服从标准正太分布，
        当不是时，可以修改stats.xxx.ppf获取其他累积分布对应的分位值
        :param data0,data1: 分布数据  np.array, series
        :param label0, label1: 图例标签 str
        :param quantile_num: 绘制qq图提取多少的分位数 int
        :param ax_cuml: 绘制累计分布 axe
        :param ax_qq: 绘制分位数对比图 axe
        :param ax_qq_ref: 绘制分位数对比图 axe
        :param ax_qq_standar: 绘制分位数对比图与已知分布进行对比
        :param fig:
        :return:
        """
        def cuml_prob_plot(data, axe=None, clr=None, label=None):
            """
            绘制一组数据的累积密度，并返回累积密度数组
            """
            y_vals = np.arange(len(data)) / float(len(data))  # 计算累积密度
            if axe:
                sns.lineplot(x=np.sort(data),
                             y=y_vals,
                             ax=axe,
                             color=clr,
                             label=label)
            return y_vals
        if ax_cuml:
            cuml_prob_plot(data0,
                           axe=ax_cuml,
                           clr=self.color_map[0],
                           label=label0)
            cuml_prob_plot(data1,
                           axe=ax_cuml,
                           clr=self.color_map[1],
                           label=label1)
            ax_cuml.set_title(f'{label0} and {label1} cumulative prob')
            ax_cuml.set_xlabel('value')
        if ax_qq:
            """
            绘制qq图, x为等距分位数、y轴为分位数数值
            """
            # 获取分位数
            q_aray = np.linspace(0, 1, quantile_num)
            quantile0 = np.quantile(data0, q_aray)
            quantile1 = np.quantile(data1, q_aray)
            sns.scatterplot(x=q_aray,
                            y=quantile0,
                            color=self.color_map[0],
                            alpha=0.5,
                            label=label0,
                            ax=ax_qq)
            sns.scatterplot(x=q_aray,
                            y=quantile1,
                            color=self.color_map[1],
                            alpha=0.5,
                            label=label1,
                            ax=ax_qq)
            ax_qq.set_title(f'{label0} and {label1} q-q plot')
            ax_qq.set_xlabel('quantile')
        if ax_qq_ref:
            """
            绘制qq图，x轴为分布0、y轴为分布1的相同分位数值
            """
            # 获取分位数
            q_aray = np.linspace(0, 1, quantile_num)
            quantile0 = np.quantile(data0, q_aray)
            quantile1 = np.quantile(data1, q_aray)
            # 绘制散点、折线图，折线作为为基础参照分布
            sns.lineplot(x=quantile0,
                         y=quantile0,
                         color=self.color_map[0],
                         label=f'{label0}_benchmark',
                         ax=ax_qq_ref)
            sns.scatterplot(x=quantile0,
                            y=quantile1,
                            color=self.color_map[1],
                            alpha=0.5,
                            label=label1,
                            ax=ax_qq_ref)
            title=f'{label1} and {label0} q-q plot'
            ax_qq_ref.set_title(title, fontdict={'size':20}, loc='left')
            ax_qq_ref.set_xlabel(f'{label0} quantile', fontdict={'size': 20})
            ax_qq_ref.set_ylabel(f'{label1} quantile', fontdict={'size': 20})
        self._save_fig(fig, label0 + '_' + label1, 'qq_plot')


    def qq_plot_standar(self, data, label, ax_qq_standar, standar_dist='chi2', **parms):
        """
        绘制给定数据与标准分布的qq图
        :param data: 给定数据
        :param label: 数据显示标签
        :param ax_qq_standar: 子图
        :param standar_dist: 标准分布类型
        :param parms: 标准分布相关参数
        :return:
        """
        def cuml_prob_plot(data, axe=None, clr=None, label=None):
            """
            绘制一组数据的累积密度，并返回累积密度数组
            """
            y_vals = np.arange(len(data)) / float(len(data))  # 计算累积密度
            if axe:
                sns.lineplot(x=np.sort(data),
                             y=y_vals,
                             ax=axe,
                             color=clr,
                             label=label)
            return y_vals

        if ax_qq_standar:
            """
            绘制qq图，x轴为标准分布的相同分位数值
            """
            datal_y = cuml_prob_plot(data)
            # 对目标累计分布函数值求已知分布的累计分布函数的逆；
            # 可求f，chi2，t等，默认已知分布为标准正态分布
            if standar_dist == 'norm':
                x_label = stats.norm.ppf(datal_y, **parms)
                # x_label = stats.norm.ppf(datal_y, loc=0, scale=1)
            elif standar_dist == 'chi2':
                x_label = stats.chi2.ppf(datal_y, **parms)  # 绘制卡方分布
                # x_label = stats.chi2.ppf(datal_y, df=4) # 绘制卡方分布
            elif standar_dist == 't':
                x_label = stats.t.ppf(datal_y, **parms)  # t分布
                # x_label = stats.t.ppf(datal_y, df=4) # t分布
            elif standar_dist == 'f':
                x_label = stats.f.ppf(datal_y, **parms)  # f分布
                # x_label = stats.f，ppf(datal_y, dfn=4, dfd=5) # f分布
            else:
                raise ValueError(f'当前内置分布[norm, chi2, t, f]，请自定义相关分布{standar_dist}')
            sns.lineplot(x=x_label,
                         y=x_label,
                         color=self.color_map[0],
                         label=standar_dist+'line',
                         ax=ax_qq_standar)
            sns.scatterplot(x=x_label,
                            y=np.sort(data),
                            color=self.color_map[1],
                            alpha=0.5,
                            label=label,
                            ax=ax_qq_standar)
            ax_qq_standar.set_title(f'{standar_dist} and {label} q-q plot')
            ax_qq_standar.set_xlabel(f'{standar_dist} quantile value')

    def cat_dist_plot(self, df, cat_name, axe, pct=0.8,
                        x_tick_rotation=45, x_tick_fontsize=15, sort_val=True, fig=None):
        """
        对单个分类变量进行数量分布统计
        :param df: 特征数据 dataframe
        :param cat_name: 分类特征名称 str
        :param axe: 子图 axe
        :param pct: 比例分割阈值默认为0.8,即标注累计比例刚好超过0.8的类别
        :param sort_val: 按统计量降序排列，若为False，则按分类变量的unique升序排列，适用于时间序列
        :param fig:
        :return:
        """
        y = df[cat_name]
        if sort_val:
            x = y.value_counts().sort_values(ascending=False).index
            y = y.value_counts().sort_values(ascending=False).values
        else:
            x = y.value_counts().sort_index().index
            y = y.value_counts().sort_index().values

        # 绘制数量柱状图, 并在8-2分割点出画出一条直线增加注释
        sns.barplot(x=np.arange(len(y)),
                    y=y,
                    ax=axe)
        # 组合图将累计数量折线加进去
        axe_1 = axe.twinx()
        sns.lineplot(x=np.arange(len(y)),
                     y=y.cumsum() / y.sum(),
                     ax=axe_1)
        if pct:
            # 获取比例分割点标注分割线
            num_82 = np.where(y.cumsum() / y.sum() - pct >= 0)[0][0]
            axe.axvline(x=num_82,
                        color='r',
                        linestyle='--',
                        alpha=1,
                        linewidth=0.5)
            y_arry = y.cumsum() / y.sum()
            axe_1.annotate(str(round(y_arry[num_82],2) * 100) + '%',
                           xy=(num_82 + 0.2, y_arry[num_82]))

        axe.set_xticklabels(labels=x,
                            rotation=x_tick_rotation,
                            fontsize=x_tick_fontsize)
        axe.set_xlabel(cat_name)
        axe.set_ylabel('count / count_pct')
        axe.set_title(cat_name + 'count_dist')

        self._save_fig(fig, cat_name, '_bar_plot')

    def cat_cat_plot(self, df, cat_name0, cat_name1, axe_bar=None,
                     auto_vehival=True, normalize=True, axe_hot=None, lengend_font=None, fig=None):
        """
        对两个分类变量的分布进行数量交叉统计，输出条形图或频率分布热力图
        :param df: 特征数据 dataframe
        :param cat_name0, cat_name1: 分类特征名称 str
        :param axe_bar, axe_hot: 绘制分布柱状图或热力图 axe
        :param auto_vehival: 自动控制柱状图是否水平绘制
        :param normalize: 控制同类别分布是否归一化
        :param lengend_font: 图例参数 dict
        :param fig:
        :return:
        """
        if axe_bar:
            df_tmpt = pd.DataFrame(df.groupby(
                [cat_name0, cat_name1]).apply(len))
            df_tmpt = df_tmpt.reset_index()
            df_tmpt.rename(columns={0: 'count'}, inplace=True)
            if normalize:
                """
                数量归一化处理
                """
                num_tol = df_tmpt.groupby(cat_name1).agg({'count': sum}).to_dict()['count']
                df_tmpt.loc[:, 'count'] = df_tmpt['count'] / df_tmpt[cat_name1].map(num_tol)
            if auto_vehival and df_tmpt[cat_name1].nunique() >= 10:
                """
                判断类别数量，超过10个采用水平柱状图
                """
                sns.barplot(data=df_tmpt,
                            x='count',
                            y=cat_name1,
                            hue=cat_name0,
                            orient='h',
                            ax=axe_bar)
            else:
                sns.barplot(data=df_tmpt,
                            x=cat_name1,
                            y='count',
                            hue=cat_name0,
                            orient='v',
                            ax=axe_bar)
                axe_bar.set_xticklabels(axe_bar.get_xticklabels(),
                                        rotation=45)
            axe_bar.legend(loc='best',
                           prop=lengend_font)
            axe_bar.set_title('{}_{}_unique_barplot'.format(cat_name1, cat_name0))

        if axe_hot:
            """
            数量分布热力图
            """
            df_tmpt = pd.crosstab(index=df[cat_name0],
                                  columns=df[cat_name1],
                                  dropna=True)
            df_tmpt = round(df_tmpt / df_tmpt.sum().sum(), 3)
            sns.heatmap(df_tmpt,
                        annot=True,
                        annot_kws={'alpha': 0.8, 'size': 15},
                        center=0,
                        linewidths=0.5,
                        ax=axe_hot)  # annot_kws调整显示数值

        self._save_fig(fig, cat_name0 + '_' + cat_name1, 'cross_plot')

    """
    以下根据天池大数据整理
    """
    def violine_cat_dist(self, df, cat_name, num_name_ls, axe, fig=None):
        """
        绘制小提琴图 小提琴的每个半边为不同的类别的分布
        :param df: 特征数据 dataframe
        :param cat_name: 分类特征 , 必须是二分类 str
        :param num_name_ls: 数值特征列表 list
        :param axe:
        :return:
        """
        # 去中心化
        data = (df[num_name_ls] - df[num_name_ls].mean()) / df[num_name_ls].std()
        data = pd.concat([df[cat_name], data], axis=1)
        data = pd.melt(data,
                       id_vars=cat_name,
                       var_name='Features',
                       value_name='Values')
        # 绘制小提琴图
        sns.violinplot(data=data,
                       x='Features',
                       y='Values',
                       hue=cat_name,
                       split=True,
                       inner='quart',
                       ax=axe,
                       palette='Blues')
        plt.xticks(rotation=45)
        self._save_fig(fig, cat_name + '_nums', 'dist_plot')

    def swarm_cat_plot(self, df, cat_name, num_name_ls, axe, sample=1000, fig=None):
        """
        将数值变量按照分类变量分散开来，避免重叠
        :param df: 特征数据 dataframe
        :param cat_name: 分类特征 str
        :param num_name_ls: 数值特征列表 list
        :param axe:
        :return:
        """
        if df.shape[0] <= sample:
            sample = df.shape[0]
        else:
            pass
        data = df.sample(sample)
        data_std = (data[num_name_ls] - data[num_name_ls].mean()) / data[num_name_ls].std()
        data = pd.concat([data[cat_name], data_std], axis=1)
        data = pd.melt(data,
                       id_vars=cat_name,
                       var_name='Features',
                       value_name='Values')
        sns.swarmplot(data=data,
                      x='Features',
                      y='Values',
                      hue=cat_name,
                      ax=axe)
        plt.xticks(rotation=45)
        self._save_fig(fig, cat_name + '_nums', 'swarm_plot')

    def dist_plot_nums(self, df, num_name_ls, figsize=(12, 10), bins=20):
        """
        一次性快速绘制多个单个变量的频率分布图
        :param df: 特征数据 dataframe
        :param num_name_ls: 数值特征列表 list
        :return:
        """
        df[num_name_ls].hist(figsize=figsize,
                             bins=bins)
        plt.show()


