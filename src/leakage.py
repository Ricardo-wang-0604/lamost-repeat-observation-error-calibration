# -*- coding: utf-8 -*-
"""
标签泄漏实验：观测级随机划分 vs 恒星级分组划分

机制
----
LAMOST 对同一颗恒星（同一 gp_id / designation）会观测多次，每次产生一行样本。
同一颗星的测光特征（PS1 griz y、Gaia G）在多次观测之间是**完全相同**的常数，
但恒星参数标签（teff/logg/feh）因每次曝光信噪比不同而有涨落。

于是一个星的特征向量能同时在训练集与验证集出现，两种划分因此回答**两个不同的问题**：

  随机划分（按观测）  → 测试集的每颗星在训练集里都见过。
                        模型对该星最可靠的策略是给出它的均值，
                        于是 RMSE 收敛到**组内离散度**（= 复观池化 σ）。
                        它回答的是：「重复预测一颗已见过的星，误差多大？」
  分组划分（按恒星）  → 每颗星只出现在一侧。
                        它回答的是：「预测一颗从未见过的星，误差多大？」

把前者当作后者来汇报，就是**标签泄漏**：报告的精度不是泛化精度。

这与风电功率预测中"多场站 NWP 输入重复导致交叉验证泄漏"属于同一类问题，
只是分组键从『场站』换成了『恒星』。

自洽校验
--------
若随机划分测得的是组内离散，则应有  RMSE_random ≈ pooled σ。
本模块的调用方（scripts/04_leakage.py）会做这一交叉验证。

评测口径
--------
  RMSE_random  : 随机划分（已见过的星）下的 RMSE
  RMSE_grouped : 分组划分（未见过的新星）下的 RMSE
  factor       : RMSE_grouped / RMSE_random
  rel_inflation: (RMSE_random - RMSE_grouped) / RMSE_grouped * 100
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 特征

# 测光是"恒星级常数"特征（复观之间不变）—— 泄漏的来源
STAR_LEVEL_FEATURES = ['mag_ps_g', 'mag_ps_r', 'mag_ps_i', 'mag_ps_z', 'mag_ps_y',
                       'gaia_g_mean_mag', 'g_r', 'r_i', 'i_z', 'z_y', 'g_i', 'r_z']

# 观测级特征（复观之间会变）—— 不构成恒星级泄漏
OBS_LEVEL_FEATURES = ['snrg', 'snrr', 'snru', 'snri', 'snrz']


def build_matrix(df, feature_set='photometry', target='teff'):
    """构造 (X, y, groups) 矩阵。groups 为 gp_id，用于分组划分。"""
    if feature_set == 'photometry':
        cols = [c for c in STAR_LEVEL_FEATURES if c in df.columns]
    elif feature_set == 'obs':
        cols = [c for c in OBS_LEVEL_FEATURES if c in df.columns]
    elif feature_set == 'all':
        cols = [c for c in STAR_LEVEL_FEATURES + OBS_LEVEL_FEATURES if c in df.columns]
    else:
        raise ValueError('未知 feature_set: %s' % feature_set)

    need = cols + [target, 'gp_id']
    sub = df.dropna(subset=need).copy()
    sub = sub[np.isfinite(sub[target])]
    X = sub[cols].to_numpy(dtype=np.float64)
    y = sub[target].to_numpy(dtype=np.float64)
    g = sub['gp_id'].to_numpy()
    return X, y, g, cols, sub


# ---------------------------------------------------------------- 模型


def make_model(kind='lgbm', seed=42):
    if kind == 'lgbm':
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.08, num_leaves=63,
            min_child_samples=40, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, random_state=seed, n_jobs=-1, verbose=-1)
    if kind == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=4,
                                     random_state=seed, n_jobs=-1)
    if kind == 'linear':
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0, random_state=seed)
    raise ValueError('未知模型: %s' % kind)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _r2(y, p):
    from sklearn.metrics import r2_score
    return float(r2_score(y, p))


# ---------------------------------------------------------------- 两种划分


def split_random(X, y, g, test_size=0.2, seed=42):
    """按『观测』随机划分 —— 会泄漏。"""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    te, tr = idx[:n_test], idx[n_test:]
    return tr, te


def split_grouped(g, test_size=0.2, seed=42):
    """按『恒星』分组划分 —— 正确。同 gp_id 只在训练或测试之一。"""
    rng = np.random.RandomState(seed)
    uniq = np.unique(g)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_size)))
    test_groups = set(uniq[:n_test].tolist())
    te = np.array([i for i, gv in enumerate(g) if gv in test_groups])
    tr = np.array([i for i, gv in enumerate(g) if gv not in test_groups])
    return tr, te


def run_split(X, y, g, tr, te, kind='lgbm', seed=42, tag=''):
    model = make_model(kind, seed)
    model.fit(X[tr], y[tr])
    p = model.predict(X[te])
    return {
        'tag': tag,
        'model': kind,
        'n_train': int(len(tr)),
        'n_test': int(len(te)),
        'n_train_groups': int(len(np.unique(g[tr]))),
        'n_test_groups': int(len(np.unique(g[te]))),
        'rmse': _rmse(y[te], p),
        'r2': _r2(y[te], p),
    }


def leakage_experiment(df, target='teff', feature_set='photometry',
                       kind='lgbm', test_size=0.2, repeats=3, seed=42, matrix=None):
    """
    跑随机划分与分组划分各 repeats 次，返回结果表与汇总。

    matrix: 可选，`build_matrix()` 的返回值 (X, y, g, cols, sub)。
            传入可避免在 745 万行上重复执行 dropna —— 调用方常常已经算过一次。
    """
    if matrix is not None:
        X, y, g, cols, sub = matrix
    else:
        X, y, g, cols, sub = build_matrix(df, feature_set=feature_set, target=target)
    rows = []
    for r in range(repeats):
        s = seed + r
        tr, te = split_random(X, y, g, test_size, s)
        res = run_split(X, y, g, tr, te, kind, s, tag='random')
        res['repeat'] = r
        rows.append(res)

        tr, te = split_grouped(g, test_size, s)
        res = run_split(X, y, g, tr, te, kind, s, tag='grouped')
        res['repeat'] = r
        rows.append(res)

    tab = pd.DataFrame(rows)
    summary = tab.groupby('tag')['rmse'].agg(['mean', 'std', 'min', 'max']).reset_index()
    mr = float(summary.loc[summary['tag'] == 'random', 'mean'].iloc[0])
    mg = float(summary.loc[summary['tag'] == 'grouped', 'mean'].iloc[0])
    inflation = {
        'target': target,
        'feature_set': feature_set,
        'n_samples': int(len(y)),
        'n_groups': int(len(np.unique(g))),
        'features': len(cols),
        'rmse_random': mr,
        'rmse_grouped': mg,
        'abs_inflation': mg - mr,
        # 两个百分比口径**分母不同**，必须分开命名，否则同一个量会有两个值。
        #   rel_increase_over_random_pct : (RMSE_g - RMSE_r) / RMSE_r × 100
        #                                  —— "随机划分把 RMSE 压低了多少百分比"
        #                                     （README 用的是这个口径）
        #   rel_inflation_pct            : (RMSE_g - RMSE_r) / RMSE_g × 100
        #                                  —— 历史口径，保留以兼容旧结果表
        'rel_increase_over_random_pct': 100.0 * (mg - mr) / mr if mr else np.nan,
        'rel_inflation_pct': 100.0 * (mg - mr) / mg if mg else np.nan,
        'factor': mg / mr if mr else np.nan,
        'r2_random': float(tab.loc[tab['tag'] == 'random', 'r2'].mean()),
        'r2_grouped': float(tab.loc[tab['tag'] == 'grouped', 'r2'].mean()),
    }
    return tab, summary, inflation, sub


def feature_importance(df, target='teff', feature_set='photometry', kind='lgbm',
                       seed=42, matrix=None):
    """特征重要性。matrix 同上，传入可复用以避免重复 dropna。"""
    if matrix is not None:
        X, y, g, cols, _ = matrix
    else:
        X, y, g, cols, _ = build_matrix(df, feature_set=feature_set, target=target)
    m = make_model(kind, seed)
    m.fit(X, y)
    imp = getattr(m, 'feature_importances_', None)
    if imp is None:
        return pd.DataFrame()
    return (pd.DataFrame({'feature': cols, 'importance': imp})
            .sort_values('importance', ascending=False).reset_index(drop=True))


if __name__ == '__main__':
    print('leakage.py 就绪；请用 scripts/ 下的脚本调用。')
