import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px # 用于交互式绘图
import plotly.graph_objects as go # 更底层的Plotly绘图
from plotly.subplots import make_subplots # 创建子图
import statsmodels.api as sm # 用于统计分析，如相关性

# --- 设置 Matplotlib 中文和坐标轴负号显示 (确保这个设置有效) ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("已设置字体为 SimHei for Matplotlib")
except Exception as e:
    print(f"设置 Matplotlib 字体失败: {e}")

# --- 文件路径 ---
file_paths = {
    'other_trading': 'other_guba_trading_time.csv',
    'other_daily': 'other_guba_daily.csv',
    'company_trading': 'company_guba_trading_time.csv',
    'company_daily': 'company_guba_daily.csv'
}

# --- 定义列名清洗函数 ---
def clean_col_names(df):
    """去除列名中的中括号及其内容"""
    df.columns = df.columns.str.replace(r'\[.*?\]', '', regex=True)
    # 修正字段说明中不一致的 Tradetm_dum -> Tradetime_Dum
    if 'Tradetm_dum' in df.columns:
         df.rename(columns={'Tradetm_dum': 'Tradetime_Dum'}, inplace=True)
    return df

# --- 加载和初步处理数据 --- ==
dfs = {}
expected_date_formats = ['%Y/%m/%d', '%Y-%m-%d'] # 可能的日期格式列表

for name, path in file_paths.items():
    try:
        df = pd.read_csv(path, encoding='utf-8') # 使用 utf-8-sig 读取带BOM的文件
        df = clean_col_names(df)

        # 尝试多种格式转换日期列 'Date'
        converted = False
        for fmt in expected_date_formats:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format=fmt)
                converted = True
                print(f"文件 '{path}' 的 'Date' 列成功按格式 '{fmt}' 转换。")
                break # 成功转换后跳出循环
            except (ValueError, TypeError):
                continue # 格式不匹配，尝试下一种
        if not converted:
            print(f"警告：文件 '{path}' 的 'Date' 列未能按预期格式转换，将尝试自动推断。")
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                print(f"文件 '{path}' 的 'Date' 列成功自动推断格式。")
            except Exception as e:
                print(f"错误：文件 '{path}' 的 'Date' 列转换失败: {e}")
                # 可以选择跳过这个文件或停止
                continue

        # 统一转换为 YYYY-MM-DD 格式显示 (可选)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df['Date'] = pd.to_datetime(df['Date']) # 转回datetime对象

        dfs[name] = df
        print(f"成功加载并初步处理文件: {path}")
        print(f"数据预览 (前3行):\n{df.head(3)}\n")
        print(f"数据列名: {df.columns.tolist()}\n")

    except FileNotFoundError:
        print(f"错误：文件 '{path}' 未找到。请检查路径。")
    except Exception as e:
        print(f"加载或处理文件 '{path}' 时出错: {e}")

# --- 筛选感兴趣的时间段 ---
# 定义核心研究时段
periods_of_interest = {
    '疫情期': ('2020-07-01', '2020-10-31'),
    '俄乌冲突期': ('2022-02-01', '2022-05-31')
}

# 对每个 DataFrame 进行筛选
dfs_filtered = {}
for name, df in dfs.items():
    df_period_list = []
    for period_name, (start_date, end_date) in periods_of_interest.items():
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df_period = df[mask].copy() # 使用 .copy() 避免 SettingWithCopyWarning
        df_period['Period'] = period_name # 添加时期标签
        df_period_list.append(df_period)
    if df_period_list:
        dfs_filtered[name] = pd.concat(df_period_list)
        print(f"已筛选 '{name}' 数据集中感兴趣的时期。筛选后行数: {len(dfs_filtered[name])}")
    else:
        print(f"'{name}' 数据集中未找到感兴趣时期的数据。")

# --- 查看筛选后的数据结构 (示例) ---
if 'company_daily' in dfs_filtered:
    print("\n筛选后 'company_daily' 数据预览:")
    print(dfs_filtered['company_daily'].head())
    print("\n筛选后 'company_daily' 时期分布:")
    print(dfs_filtered['company_daily']['Period'].value_counts())

print("\n--- 开始构建情感指数 ---")

def calculate_sentiment_indices(df):
    """计算各种情感指数"""
    # 处理分母为0的情况，避免除零错误
    epsilon = 1e-6 # 一个很小的数，防止除以零

    # 情感净值
    df['Sent_NetValue'] = df['Pospostnum'] - df['Negpostnum']

    # 情感比率 (-1 到 1)
    denominator_ratio = df['Pospostnum'] + df['Negpostnum']
    df['Sent_Ratio'] = (df['Pospostnum'] - df['Negpostnum']) / (denominator_ratio + epsilon)
    # 可以使用.clip限制在 [-1, 1] 区间内，虽然理论上不会超出
    df['Sent_Ratio'] = df['Sent_Ratio'].clip(-1, 1)

    # 看涨看跌比率 (可能非常大或为0)
    denominator_bullbear = df['Negpostnum']
    df['Sent_BullBear'] = df['Pospostnum'] / (denominator_bullbear + epsilon)
    
    # 对于分母为0，分子也为0的情况，设为1 (中性)；分母为0，分子>0，设为一个较大的数或NaN
    df.loc[(df['Pospostnum'] == 0) & (df['Negpostnum'] == 0), 'Sent_BullBear'] = 1.0
    df.loc[(df['Pospostnum'] > 0) & (df['Negpostnum'] == 0), 'Sent_BullBear'] = np.inf # 或设为某个大数，e.g., 999

    print("已计算 Sent_NetValue, Sent_Ratio, Sent_BullBear 指数。")
    return df

# 对筛选后的所有 DataFrame 应用计算
for name in dfs_filtered.keys():
    print(f"\n计算 '{name}' 数据集的情感指数...")
    dfs_filtered[name] = calculate_sentiment_indices(dfs_filtered[name])

# 查看带有情感指数的数据(示例)
if 'company_daily' in dfs_filtered:
    print('\n带有情感指数的‘company_daily’数据预览：')
    print(dfs_filtered['company_daily'][['Date','Coname','Period','Pospostnum', 'Negpostnum', 'Sent_NetValue', 'Sent_Ratio', 'Sent_BullBear']].head())

print("\n--- 开始构建加权情感指数 ---")

def calculate_weighted_sentiment(df, weight_col='Readnum'):
    """使用指定权重列计算加权情感指数"""
    if weight_col not in df.columns:
        print(f"警告：权重列 '{weight_col}' 不存在于DataFrame中，无法计算加权指数。")
        return df

    epsilon = 1e-6
    weight = df[weight_col] + epsilon # 避免权重为0

    # 加权情感净值 (按权重调整后的净值)
    # 这个指标意义可能不直观，更常用的是加权平均情感比率
    # df['Sent_NetValue_Weighted'] = df['Sent_NetValue'] * weight

    # 加权情感比率 (更常用：用权重乘以比率)
    # 注意：如果 Sent_Ratio 本身已经是 -1 到 1，乘以一个大的权重会改变范围
    # 另一种方法是计算加权平均比率，但这需要先计算每个时间段的总权重
    # 为了简单，我们先计算一个加权得分 (未标准化)
    df[f'Sent_Ratio_Weighted_{weight_col}'] = df['Sent_Ratio'] * weight

    print(f"已计算基于 '{weight_col}' 的加权情感比率得分。")
    return df

# 对筛选后的所有 DataFrame 应用计算 (分别用阅读数和评论数加权)
for name in dfs_filtered.keys():
    print(f"\n计算 '{name}' 数据集的加权情感指数...")
    dfs_filtered[name] = calculate_weighted_sentiment(dfs_filtered[name], weight_col='Readnum')
    dfs_filtered[name] = calculate_weighted_sentiment(dfs_filtered[name], weight_col='Commentnum')

# --- 查看加权指数 (示例) ---
if 'company_daily' in dfs_filtered:
    print("\n带有加权情感指数的 'company_daily' 数据预览:")
    cols_to_show = ['Date', 'Coname', 'Sent_Ratio', 'Readnum', 'Sent_Ratio_Weighted_Readnum', 'Commentnum', 'Sent_Ratio_Weighted_Commentnum']
    # 检查列是否存在
    cols_to_show = [col for col in cols_to_show if col in dfs_filtered['company_daily'].columns]
    print(dfs_filtered['company_daily'][cols_to_show].head())

print("\n--- 开始时间序列聚合与可视化 ---")

# --- 准备绘图数据 ---
# 我们重点关注每日数据 ('other_daily', 'company_daily')
# 对于上市公司数据，我们需要按日期聚合所有公司，或分别绘制
# 对于其他股吧数据，也需要按日期聚合

# 示例：聚合每日的公司总体情绪和关注度指标
if 'company_daily' in dfs_filtered:
    df_comp_daily = dfs_filtered['company_daily']
    # --- 第一步：进行简单的列聚合 ---
    print("进行初步聚合 (sum, mean)...")
    daily_agg_comp_simple = df_comp_daily.groupby(['Date', 'Period']).agg(
        Tpostnum_sum=('Tpostnum', 'sum'),
        Pospostnum_sum=('Pospostnum', 'sum'),
        Negpostnum_sum=('Negpostnum', 'sum'),
        Readnum_sum=('Readnum', 'sum'),
        Commentnum_sum=('Commentnum', 'sum'),
        Sent_Ratio_mean=('Sent_Ratio', 'mean'), # 计算简单平均情感比率
        # 我们还需要原始的 Sent_Ratio 和权重列，以便后续计算加权平均
        # 可以传入列表来获取，但这样不太方便，不如分开算
    ).reset_index() # 将分组后重复/无意义的索引列重新转换为普通默认整数索引

    # --- 第二步：定义加权平均函数 (作用于分组后的 DataFrame) ---
    def weighted_average(group, avg_col, weight_col):
        """计算分组DataFrame的加权平均值"""
        d = group[avg_col]
        w = group[weight_col] + 1e-6 # 避免权重为0
        try:
            # 只有当权重和 > 0 时才计算
            if w.sum() > 1e-6:
                 return np.average(d, weights=w)
            else:
                 return 0 # 或者返回 np.nan
        except ZeroDivisionError:
            return 0 # 或者返回 np.nan

    # --- 第三步：使用 groupby().apply() 计算加权平均 ---
    # apply() 允许将函数应用于每个分组的 DataFrame
    print("计算加权平均情感比率...")
    wavg_readnum = df_comp_daily.groupby(['Date', 'Period']).apply(
        weighted_average, 'Sent_Ratio', 'Readnum' # 传入要平均的列和权重列名
    ).rename('Sent_Ratio_wavg_Readnum') # 重命名 Series

    wavg_commentnum = df_comp_daily.groupby(['Date', 'Period']).apply(
        weighted_average, 'Sent_Ratio', 'Commentnum'
    ).rename('Sent_Ratio_wavg_Commentnum')

    # --- 第四步：将所有聚合结果合并 ---
    print("合并聚合结果...")
    # 将加权平均结果 (Series) 合并到初步聚合的 DataFrame
    daily_agg_comp = pd.merge(daily_agg_comp_simple, wavg_readnum, on=['Date', 'Period'], how='left')
    daily_agg_comp = pd.merge(daily_agg_comp, wavg_commentnum, on=['Date', 'Period'], how='left')

    # --- 第五步：计算整体情感比率 (基于总和) ---
    print("计算整体情感比率...")
    daily_agg_comp['Sent_Ratio_overall'] = (daily_agg_comp['Pospostnum_sum'] - daily_agg_comp['Negpostnum_sum']) / (daily_agg_comp['Pospostnum_sum'] + daily_agg_comp['Negpostnum_sum'] + 1e-6)
    daily_agg_comp['Sent_Ratio_overall'] = daily_agg_comp['Sent_Ratio_overall'].clip(-1, 1)

    print("\n上市公司每日聚合数据预览:")
    print(daily_agg_comp.head())

# --- 绘制上市公司总体情绪与关注度时间序列图 (使用 Plotly) ---
fig_comp_agg = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=("每日总发帖量", "每日总阅读数", "每日聚合情感比率 (Overall & 加权阅读数)"))

# 添加发帖量轨迹
fig_comp_agg.add_trace(go.Scatter(x=daily_agg_comp['Date'], y=daily_agg_comp['Tpostnum_sum'],
                                    mode='lines', name='总发帖量'), row=1, col=1)

# 添加阅读数轨迹
fig_comp_agg.add_trace(go.Scatter(x=daily_agg_comp['Date'], y=daily_agg_comp['Readnum_sum'],
                                    mode='lines', name='总阅读数'), row=2, col=1)

# 添加情感比率轨迹
fig_comp_agg.add_trace(go.Scatter(x=daily_agg_comp['Date'], y=daily_agg_comp['Sent_Ratio_overall'],
                                    mode='lines', name='Overall Sent Ratio'), row=3, col=1)
fig_comp_agg.add_trace(go.Scatter(x=daily_agg_comp['Date'], y=daily_agg_comp['Sent_Ratio_wavg_Readnum'],
                                    mode='lines', name='WAvg Sent Ratio (Readnum)', line=dict(dash='dot')), row=3, col=1)
# 添加零线，方便观察正负
fig_comp_agg.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=3, col=1)


# 更新布局
fig_comp_agg.update_layout(title_text="上市公司股吧总体情绪与关注度时序变化", height=1000)
fig_comp_agg.update_xaxes(title_text="日期", row=3, col=1)
fig_comp_agg.update_yaxes(title_text="数量", row=1, col=1)
fig_comp_agg.update_yaxes(title_text="数量", row=2, col=1)
fig_comp_agg.update_yaxes(title_text="情感比率 (-1 to 1)", range=[-1, 1], row=3, col=1) # 固定y轴范围

# # 添加时期背景色 (可选)
# for period_name, (start_date, end_date) in periods_of_interest.items():
#     color = 'rgba(255, 0, 0, 0.1)' if period_name == '疫情期' else 'rgba(0, 0, 255, 0.1)'
#     fig_comp_agg.add_vrect(x0=start_date, x1=end_date, fillcolor=color, layer="below", line_width=0, row="all", col=1)


fig_comp_agg.show()
# fig_comp_agg.write_image("company_sentiment_attention_timeseries.png") # 保存为静态图片

# --- 示例：对比不同类型股吧的情绪 (上市公司整体 vs 其他股吧整体) ---
if 'other_daily' in dfs_filtered and 'company_daily' in daily_agg_comp:
    df_other_daily = dfs_filtered['other_daily']
    # 按日期聚合其他股吧数据
    daily_agg_other = df_other_daily.groupby(['Date', 'Period']).agg(
         Pospostnum_sum=('Pospostnum', 'sum'),
         Negpostnum_sum=('Negpostnum', 'sum'),
         # Sent_Ratio_wavg_Readnum=lambda x: np.average(x['Sent_Ratio'], weights=x['Readnum'] + 1e-6) if x['Readnum'].sum() > 0 else 0,
    ).reset_index()
    daily_agg_other['Sent_Ratio_overall'] = (daily_agg_other['Pospostnum_sum'] - daily_agg_other['Negpostnum_sum']) / \
                                            (daily_agg_other['Pospostnum_sum'] + daily_agg_other['Negpostnum_sum'] + 1e-6)
    daily_agg_other['Sent_Ratio_overall'] = daily_agg_other['Sent_Ratio_overall'].clip(-1, 1)

    # 合并两种类型的聚合数据
    daily_agg_comp_subset = daily_agg_comp[['Date', 'Sent_Ratio_overall']].rename(columns={'Sent_Ratio_overall': 'Company_Sent_Ratio'})
    daily_agg_other_subset = daily_agg_other[['Date', 'Sent_Ratio_overall']].rename(columns={'Sent_Ratio_overall': 'Other_Sent_Ratio'})
    merged_sent = pd.merge(daily_agg_comp_subset, daily_agg_other_subset, on='Date', how='outer')

    # 绘图对比
    fig_compare_type = px.line(merged_sent.melt(id_vars='Date', var_name='Guba_Type', value_name='Sentiment_Ratio'),
                               x='Date', y='Sentiment_Ratio', color='Guba_Type',
                               title='上市公司股吧 vs 其他股吧 每日总体情感比率对比')
    fig_compare_type.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
    fig_compare_type.update_yaxes(range=[-1, 1])
    fig_compare_type.show()
    # fig_compare_type.write_image("guba_type_sentiment_comparison.png")

# --- 示例：对比单个核心公司 vs 其他股吧 (以山东黄金为例) ---
if 'company_daily' in dfs_filtered and 'other_daily' in dfs_filtered:
    target_company = '山东黄金' # 选择一个公司
    df_target_comp = dfs_filtered['company_daily'][dfs_filtered['company_daily']['Coname'] == target_company].copy()
    # 这里直接用 Sent_Ratio_overall 计算起来麻烦，用计算好的 Sent_Ratio
    df_target_comp = df_target_comp[['Date', 'Sent_Ratio']].rename(columns={'Sent_Ratio': f'{target_company}_Sent_Ratio'})

    # 使用上面聚合好的 other 数据
    # daily_agg_other_subset ...
    df_other_daily = dfs_filtered['other_daily']
    daily_agg_other = df_other_daily.groupby(['Date', 'Period']).agg(
         Pospostnum_sum=('Pospostnum', 'sum'),
         Negpostnum_sum=('Negpostnum', 'sum'),
         # Sent_Ratio_wavg_Readnum=lambda x: np.average(x['Sent_Ratio'], weights=x['Readnum'] + 1e-6) if x['Readnum'].sum() > 0 else 0,
    ).reset_index()
    daily_agg_other['Sent_Ratio_overall'] = (daily_agg_other['Pospostnum_sum'] - daily_agg_other['Negpostnum_sum']) / \
                                            (daily_agg_other['Pospostnum_sum'] + daily_agg_other['Negpostnum_sum'] + 1e-6)
    daily_agg_other['Sent_Ratio_overall'] = daily_agg_other['Sent_Ratio_overall'].clip(-1, 1)
    
    daily_agg_other_subset = daily_agg_other[['Date', 'Sent_Ratio_overall']].rename(columns={'Sent_Ratio_overall': 'Other_Sent_Ratio'})

    merged_comp_other = pd.merge(df_target_comp, daily_agg_other_subset, on='Date', how='outer')

    fig_compare_sdhj_other = px.line(merged_comp_other.melt(id_vars='Date', var_name='Source', value_name='Sentiment_Ratio'),
                                     x='Date', y='Sentiment_Ratio', color='Source',
                                     title=f'{target_company} vs 其他股吧 每日情感比率对比')
    fig_compare_sdhj_other.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
    fig_compare_sdhj_other.update_yaxes(range=[-1, 1])
    fig_compare_sdhj_other.show()
    # fig_compare_sdhj_other.write_image("sdhj_vs_other_sentiment.png")


    print("\n--- 开始交易/非交易时间段情绪分析 ---")
# 主要使用*_trading 数据集

if 'company_trading' in dfs_filtered:
    df_comp_trade = dfs_filtered['company_trading'] 

    # 确保 Tradetime_Dum 列是数值类型(0或1)
    df_comp_trade['Tradetime_Dum']= pd.to_numeric(df_comp_trade['Tradetime_Dum'],errors='coerce') #.fillna(0).astype(int) # 确保是整数0或1
    
    # 方法一：按是否交易时间段分组，计算平均情感
    avg_sent_by_tradetime = df_comp_trade.groupby(['Period','Tradetime_Dum'])['Sent_Ratio'].mean().unstack()

    # 重命名列名（非交易时间段/交易时间段），使其更清晰
    avg_sent_by_tradetime.columns = ['NonTrading_AvgSent', 'Trading_AvgSent']
    print('\n按时期和是否交易时段划分的平均情感比率:')
    print(avg_sent_by_tradetime)


# 可视化对比(柱状图)
avg_sent_by_tradetime.plot(kind='bar', figsize=(10,6))
plt.title('交易时段 vs 非交易时段 平均情感比率对比')
plt.xlabel('时期')
plt.ylabel('平均情感比率')
plt.xticks(rotation=0)
plt.axhline(0,color='grey',linewidth=0.8,linestyle='--') # 添加零线

# 观察交易日前后情绪变化（更复杂，需要按日期和时间排序处理）
# 例如，可以比较 交易日9:00-15:00的情绪 与 前一天15：00-当天9：00的情绪 差异
# 需要更复杂的情绪处理，例如将日期和时间段合并或排序

# 简化：查看交易时段的帖子/阅读量占比，了解信息发布集中度
volume_by_tradetime = df_comp_trade.groupby(['Period','Tradetime_Dum'])[['Tpostnum','Readnum','Commentnum']].sum()
# 正确处理：使用transform() 方法
# groupby(level='Period').transform('sum') 会返回一个与 volume_by_tradetime 形状相同、索引相同的DataFrame,
# 其中每个单元格的值是其所属 Period 分组的总和。
period_sums = volume_by_tradetime.groupby(level='Period').transform('sum')

# 直接用原始总量除以同形状的时期总和，得到占比，索引保持不变 (Period, Tradetime_Dum)
volume_by_tradetime_pct = 100 * volume_by_tradetime / (period_sums + 1e-6) # 加 epsilon 防除零

print("\n按时期和是否交易时段划分的发帖/阅读/评论量占比 (%) [使用 Transform]:")
# 打印前检查一下索引，确认结构是 (Period, Tradetime_Dum)
print(volume_by_tradetime_pct.head())
print("\n索引结构:")
print(volume_by_tradetime_pct.index)

p=volume_by_tradetime_pct[['Tpostnum','Readnum']].unstack(level='Tradetime_Dum')
p.rename(columns={0:'非交易时段',1:'交易时段'},inplace=True)
print(p)

# --- 4. 可视化占比 (这部分代码可以保持不变) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

# 绘制帖子数占比图
p['Tpostnum'].plot(kind='bar', stacked=True, ax=axes[0], colormap='tab10')
axes[0].set_title('交易时段 vs 非交易时段 帖子数占比')
axes[0].set_xlabel('时期')
axes[0].set_ylabel('占比 (%)')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='时段类型')
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.1f%%', label_type='center', color='white', weight='bold', fontsize=9)
# 绘制阅读量占比图
p['Readnum'].plot(kind='bar', stacked=True, ax=axes[1], colormap='tab10')
axes[1].set_title('交易时段 vs 非交易时段 阅读量占比')
axes[1].set_xlabel('时期')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='时段类型')
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.1f%%', label_type='center', color='white', weight='bold', fontsize=9)

plt.suptitle('交易时段 vs 非交易时段 帖子与阅读量占比分析', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('tradetime_volume_percentage_subplots_transform.png')
plt.show()


print("\n--- 开始关联性分析 (需要外部黄金价格数据) ---")

# --- 1. 加载黄金价格数据 ---

# 确保日期格式一致，并设为索引
gold_price_df = pd.read_csv('gold_price.csv')
gold_price_df['Date'] = pd.to_datetime(gold_price_df['Date'])
gold_price_df = gold_price_df.set_index('Date')
# 仅保留收盘价，并重命名
gold_price_df = gold_price_df[['Close_Price']].rename(columns={'Close_Price': 'Gold_Price'})
print("黄金价格数据加载成功。")

# --- 2. 合并情感数据与价格数据 ---
    # 以每日聚合的公司数据为例
if 'daily_agg_comp' in locals():
    # 确保 daily_agg_comp 的 Date 是索引或普通列，以便合并
    if not isinstance(daily_agg_comp.index, pd.DatetimeIndex):
        daily_agg_comp = daily_agg_comp.set_index('Date')

    # 使用 merge 或 join 合并
    merged_data = daily_agg_comp.join(gold_price_df, how='inner') # inner join 只保留双方都有的日期
    print("\n已合并情感数据与黄金价格数据。")
    print(merged_data[['Sent_Ratio_overall', 'Sent_Ratio_wavg_Readnum', 'Tpostnum_sum', 'Gold_Price']].head())

# --- 3. 计算相关系数 ---
# 选择要分析的列
columns_to_correlate = ['Sent_Ratio_overall', 'Sent_Ratio_wavg_Readnum', 'Tpostnum_sum', 'Readnum_sum', 'Commentnum_sum', 'Gold_Price']
correlation_matrix = merged_data[columns_to_correlate].corr()
print("\n情感/关注度指标与黄金价格的相关系数矩阵:")
print(correlation_matrix)

# 可视化相关系数矩阵 (热力图)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('情绪/关注度指标与黄金价格相关性热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap_sentiment_price.png')
plt.show()

# --- 4. 可视化对比情感与价格走势 ---
fig_sent_price = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("每日聚合情感比率 (Overall)", "黄金价格 (收盘价)"))

fig_sent_price.add_trace(go.Scatter(x=merged_data.index, y=merged_data['Sent_Ratio_overall'], name='Sentiment Ratio'), row=1, col=1)
fig_sent_price.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=1, col=1)

fig_sent_price.add_trace(go.Scatter(x=merged_data.index, y=merged_data['Gold_Price'], name='Gold Price'), row=2, col=1)

fig_sent_price.update_layout(title_text="每日总体情感比率与黄金价格走势对比", height=600)
fig_sent_price.update_yaxes(title_text="情感比率 (-1 to 1)", range=[-1, 1], row=1, col=1)
fig_sent_price.update_yaxes(title_text="黄金价格", row=2, col=1)
fig_sent_price.update_xaxes(title_text="日期", row=2, col=1)
fig_sent_price.show()
# fig_sent_price.write_image("sentiment_vs_gold_price.png")

# (高级) 考虑格兰杰因果检验 (Granger Causality)
# 查看情绪指标是否能预测价格变化（或反之）
from statsmodels.tsa.stattools import grangercausalitytests
# 需要数据平稳，可能需要差分
data_for_granger = merged_data[['Gold_Price', 'Sent_Ratio_overall']].dropna()
# 差分示例
data_diff = data_for_granger.diff().dropna()
try:
    gc_res = grangercausalitytests(data_diff[['Gold_Price', 'Sent_Ratio_overall']], maxlag=4) # 测试情绪是否格兰杰引起价格
    print("\n格兰杰因果检验 (情绪 -> 价格):")
    # 解析结果比较复杂，通常看p值
    print(gc_res)
except Exception as gc_e:
     print(f"格兰杰因果检验失败: {gc_e}")    


