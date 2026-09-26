# -*- coding: utf-8 -*-
"""
复观一致性与官方误差标定

科学问题
--------
同一颗恒星多次被 LAMOST 观测，其 Teff / logg / [Fe/H] 在物理上是常数。
因此同一颗星多次观测之间的离散度，就是管线内禀的观测重复性（repeatability），
可与官方给出的随机误差 *_err 直接比较：

    若 实测离散度 >> 官方随机误差  ⇒  官方误差低估了真实不确定度

三个互补口径
------------
1) pooled 重复性（不依赖官方误差）
     sigma_pooled^2 = Σ_g Σ_i (x_i - xbar_g)^2 / Σ_g (n_g - 1)

2) 归一化卡方（把官方误差当权重）
     chi2_g = Σ_i (x_i - wmean_g)^2 / sigma_i^2 ,  dof_g = n_g - 1
     chi2_red = Σ chi2_g / Σ dof_g
   ≈1 → 官方误差自洽；>>1 → 官方误差偏小。
   均值必须用误差加权均值，否则 sigma_i 不等时会引入人为偏差。

3) 误差比  ratio = sigma_pooled / median(官方 err)

实现要点（性能）
----------------
分组数可达 5×10^6。**禁止使用 groupby.apply(自定义函数)** —— 那会是小时级。
本模块全部用 transform + 向量化聚合实现，可在千万行规模上秒级完成。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ['teff', 'logg', 'feh']
ERR_COL = {'teff': 'teff_err', 'logg': 'logg_err', 'feh': 'feh_err'}


def group_stats(df, key='gp_id', value='teff', err='teff_err',
                min_obs=2, snr_col='snrr'):
    """
    按 key 聚合的逐组统计（全向量化）。

    返回 DataFrame（index=key），列：
      n_obs, mean, wmean, pooled_ss, dof, std, chi2, err_mean, err_median, snr_median
    """
    use = df.loc[:, [key, value, err, snr_col]].copy()
    use = use[np.isfinite(use[value].to_numpy(dtype=np.float64, na_value=np.nan))]
    use = use[np.isfinite(use[err].to_numpy(dtype=np.float64, na_value=np.nan))]
    use = use[use[err] > 0]
    if use.empty:
        return pd.DataFrame()

    gb = use.groupby(key, observed=True)

    n = gb[value].size()
    keep = n[n >= min_obs].index
    use = use[use[key].isin(keep)]
    gb = use.groupby(key, observed=True)

    v = use[value].to_numpy(dtype=np.float64)
    e = use[err].to_numpy(dtype=np.float64)
    w = 1.0 / (e * e)
    use['_v'] = v
    use['_w'] = w
    use['_wv'] = w * v

    # 加权均值：Σwv / Σw
    sw = use.groupby(key, observed=True)['_w'].transform('sum')
    swv = use.groupby(key, observed=True)['_wv'].transform('sum')
    use['_wmean'] = swv / sw

    # 普通均值
    use['_mean'] = use.groupby(key, observed=True)['_v'].transform('mean')

    # 组内离差平方和
    use['_dev2'] = (use['_v'] - use['_mean']) ** 2
    use['_chi2t'] = use['_w'] * (use['_v'] - use['_wmean']) ** 2

    agg = use.groupby(key, observed=True).agg(
        n_obs=('_v', 'size'),
        mean=('_v', 'mean'),
        wmean=('_wmean', 'first'),
        pooled_ss=('_dev2', 'sum'),
        chi2=('_chi2t', 'sum'),
        err_mean=(err, 'mean'),
        err_median=(err, 'median'),
    )
    if snr_col in use.columns:
        agg['snr_median'] = use.groupby(key, observed=True)[snr_col].median()
    agg['dof'] = agg['n_obs'] - 1
    agg['std'] = np.sqrt(agg['pooled_ss'] / agg['dof'].replace(0, np.nan))
    return agg


def pooled_repeatability(stats):
    """把逐组统计池化成总体指标。"""
    if stats is None or stats.empty:
        return {}
    ss = float(stats['pooled_ss'].sum())
    dof = float(stats['dof'].sum())
    ok = stats['chi2'].notna()
    chi2 = float(stats.loc[ok, 'chi2'].sum())
    dof_chi = float(stats.loc[ok, 'dof'].sum())
    sigma = float(np.sqrt(ss / dof)) if dof > 0 else np.nan
    err_med = float(stats['err_median'].median())
    return {
        'n_groups': int(len(stats)),
        'n_obs': int(stats['n_obs'].sum()),
        'dof': int(dof),
        'pooled_sigma': sigma,
        'chi2_red': (chi2 / dof_chi) if dof_chi > 0 else np.nan,
        'err_median': err_med,
        'err_median_of_medians': float(stats['err_median'].median()),
        'ratio_sigma_over_err': (sigma / err_med) if err_med else np.nan,
    }


def error_calibration(df, key='gp_id', value='teff', err='teff_err',
                      min_obs=2, snr_bins=8):
    """单参数：整体标定 + 按 S/N 分箱。返回 (overall, binned, stats)。"""
    st = group_stats(df, key=key, value=value, err=err, min_obs=min_obs)
    if st.empty:
        return {}, pd.DataFrame(), st
    overall = pooled_repeatability(st)
    overall['value'] = value

    binned = []
    s = st[st['snr_median'].notna()] if 'snr_median' in st.columns else pd.DataFrame()
    if len(s) > 100:
        s = s.copy()
        try:
            s['snr_bin'] = pd.qcut(s['snr_median'], snr_bins, duplicates='drop')
            for b, sub in s.groupby('snr_bin', observed=True):
                if len(sub) < 20:
                    continue
                r = pooled_repeatability(sub)
                r['snr_lo'] = float(b.left)
                r['snr_hi'] = float(b.right)
                r['value'] = value
                binned.append(r)
        except Exception as exc:  # noqa: BLE001
            print('  [S/N 分箱跳过] %s' % exc)
    return overall, pd.DataFrame(binned), st


def calibration_table(df, key='gp_id', min_obs=2):
    """三个参数汇总成表。"""
    rows, bins = [], []
    for v in TARGETS:
        ov, bd, _ = error_calibration(df, key=key, value=v, err=ERR_COL[v], min_obs=min_obs)
        if ov:
            rows.append(ov)
        if len(bd):
            bd = bd.copy()
            bd['param'] = v
            bins.append(bd)
    return pd.DataFrame(rows), (pd.concat(bins, ignore_index=True) if bins else pd.DataFrame())


def repeatability_by_subset(df, key='gp_id', value='teff', err='teff_err',
                            group_col='plan_prefix', min_obs=2):
    """按某个子集列（如 plan_prefix / subclass / mjd 年代）分别算重复性。"""
    out = []
    for name, sub in df.groupby(group_col, observed=True):
        if len(sub) < 5000:
            continue
        ov, _, _ = error_calibration(sub, key=key, value=value, err=err, min_obs=min_obs)
        if ov:
            ov[group_col] = name
            out.append(ov)
    return pd.DataFrame(out)


if __name__ == '__main__':
    print('repeat.py 就绪（全向量化）；请用 scripts/ 下的脚本调用。')
